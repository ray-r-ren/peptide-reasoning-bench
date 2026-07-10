"""OpenRouter leaderboard runner."""

from __future__ import annotations

import csv
import http.client
import json
import math
import os
import random
import re
import shutil
import signal
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from pydantic import ValidationError

from peb.baselines import make_baseline_predictions
from peb.io import read_jsonl, sha256_text, write_jsonl, write_text
from peb.metrics import (
    evaluate_binding_rank,
    evaluate_human_effect,
    evaluate_pose,
    evaluate_structure,
)
from peb.schemas import (
    BindingRankCase,
    BindingRankPrediction,
    ClaimStatus,
    EvidenceDirection,
    EvidenceLevel,
    HumanEffectCase,
    HumanEffectCategory,
    HumanEffectPrediction,
    PoseCase,
    PosePrediction,
    SafetyStatus,
    StructureCase,
    StructurePrediction,
    Track,
    validate_case_record,
    validate_prediction_record,
)
from peb.version import __version__

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_ENV_NAME = "_".join(("OPENROUTER", "API", "KEY"))
PREDICTION_CHUNK_SIZE = 50
RETRY_CHUNK_SIZE = 25
RETRY_LARGE_CHUNK_SIZE = 50
RETRY_BATCH_ATTEMPTS = 2
TRACKS = ("human_effect", "binding_rank", "pose", "structure")
BASE_TRACKS = ("human_effect", "binding_rank")
TOOLS_TRACKS = ("human_effect", "binding_rank", "pose", "structure")
MODE_LABELS = {
    "base": "Base",
    "tools_high_reasoning": "Tools + high reasoning",
    "baseline": "Baseline",
    "sanity_check": "Sanity check",
}
ROW_STATUS_VALUES = {
    "clean_completed",
    "completed_with_failures",
    "excluded_provider_error",
    "excluded_invalid_json",
    "excluded_incomplete",
    "noncompetitive_baseline",
    "noncompetitive_oracle",
}
RANKABLE_ROW_STATUSES = {"clean_completed", "completed_with_failures"}

FORBIDDEN_INPUT_FIELDS = {
    "labels",
    "expected_output",
    "gold_labels",
    "label_ranking",
    "category",
    "evidence_level",
    "evidence_direction",
    "claim_status",
    "safety_status",
    "evidence_validation_status",
    "final_label_summary",
    "native_contacts",
    "gold_coordinates",
    "measured_value",
    "normalized_rank",
    "baseline_outputs",
    "prediction",
    "predictions",
    "scoring_metadata",
    "qc_result",
    "qc_notes",
    "qc_status",
    "qc_disagreement",
    "qc_resolution",
}


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter call cannot complete."""


class OpenRouterInsufficientCredits(OpenRouterError):
    """Raised when OpenRouter refuses a call because account credits are exhausted."""


def _timeout_handler(signum: int, frame: Any) -> None:
    del signum, frame
    raise OpenRouterError("OpenRouter request timed out")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _api_key() -> str:
    key = os.environ.get(OPENROUTER_ENV_NAME, "")
    if not key:
        raise OpenRouterError(f"{OPENROUTER_ENV_NAME} is not set")
    return key


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[dict[str, Any]] = None,
    timeout: int = 60,
    response_format: bool = False,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/peptide-engineering-benchmark",
        "X-Title": "Peptide Engineering Benchmark",
    }
    data = None
    if payload is not None:
        body = dict(payload)
        if response_format:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(timeout, 1))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        body_lower = body.lower()
        if exc.code == 402 and (
            "insufficient credits" in body_lower
            or "requires more credits" in body_lower
            or "can only afford" in body_lower
        ):
            raise OpenRouterInsufficientCredits(
                "OpenRouter HTTP 402: insufficient credits"
            ) from exc
        raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {body[:500]}") from exc
    except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise OpenRouterError(f"OpenRouter request failed: {reason}") from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"OpenRouter returned invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise OpenRouterError("OpenRouter returned a non-object JSON payload")
    return decoded


def _compact_model_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "created": record.get("created"),
        "description": record.get("description"),
        "context_length": record.get("context_length"),
        "pricing": record.get("pricing"),
        "architecture": record.get("architecture"),
        "top_provider": record.get("top_provider"),
        "supported_parameters": record.get("supported_parameters"),
    }


def fetch_openrouter_models(output: Union[str, Path]) -> list[dict[str, Any]]:
    payload = _request_json(OPENROUTER_MODELS_URL, timeout=60)
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise OpenRouterError("OpenRouter models response has no data list")
    models = [_compact_model_record(record) for record in data if isinstance(record, dict)]
    output_path = Path(output)
    write_text(output_path, json.dumps({"retrieved_at": _utc_now(), "models": models}, indent=2))
    selected = select_models(models)
    write_text(output_path.parent / "openrouter_models.json", json.dumps({"models": selected}, indent=2))
    selection = [
        {
            "model_id": model["model_id"],
            "family": model["family"],
            "selection_note": model["selection_note"],
        }
        for model in selected
    ]
    write_text(
        output_path.parent / "model_selection.json",
        json.dumps({"retrieved_at": _utc_now(), "selection": selection}, indent=2),
    )
    return models


def _lower_text(model: dict[str, Any]) -> str:
    return " ".join(
        str(model.get(key, "")) for key in ("id", "name", "description")
    ).lower()


def _id_name_text(model: dict[str, Any]) -> str:
    return " ".join(str(model.get(key, "")) for key in ("id", "name")).lower()


def _is_deprecated_or_alias(model: dict[str, Any]) -> bool:
    text = _lower_text(model)
    model_id = str(model.get("id", "")).lower()
    return (
        "deprecated" in text
        or "latest" in model_id
        or model_id.startswith("~")
        or "auto-router" in text
        or model_id.endswith("/auto")
        or model_id in {"openrouter/auto"}
    )


def _is_text_chat_candidate(model: dict[str, Any]) -> bool:
    text = _lower_text(model)
    architecture = model.get("architecture") or {}
    if isinstance(architecture, dict):
        output_modalities = {
            str(item).lower() for item in architecture.get("output_modalities") or []
        }
        if output_modalities:
            if "text" not in output_modalities:
                return False
            if output_modalities & {"image", "audio", "video"}:
                return False
    excluded_terms = (
        " vl",
        "-vl",
        "guard",
        "safeguard",
        "moderation",
        "embedding",
        "rerank",
        "router",
        "build",
        "code router",
    )
    return not any(token in text for token in excluded_terms)


def _price(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number


def _model_price(model: dict[str, Any]) -> float:
    pricing = model.get("pricing") or {}
    if not isinstance(pricing, dict):
        return math.inf
    prompt = _price(pricing.get("prompt"))
    completion = _price(pricing.get("completion"))
    if math.isinf(prompt) and math.isinf(completion):
        return math.inf
    return prompt + completion


def _created(model: dict[str, Any]) -> int:
    try:
        return int(model.get("created") or 0)
    except (TypeError, ValueError):
        return 0


def _context_length(model: dict[str, Any]) -> int:
    try:
        return int(model.get("context_length") or 0)
    except (TypeError, ValueError):
        return 0


def _structured_outputs_supported(model: dict[str, Any]) -> bool:
    params = model.get("supported_parameters") or []
    if not isinstance(params, list):
        return False
    return "response_format" in {str(item) for item in params}


def _provider(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else "unknown"


def _family_candidates(models: Sequence[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    def match(model: dict[str, Any]) -> bool:
        text = _id_name_text(model)
        model_id = str(model.get("id", "")).lower()
        if family == "openai":
            return model_id.startswith("openai/") and any(
                token in text for token in ("gpt", "o3", "o4", "o5")
            )
        if family == "anthropic":
            return model_id.startswith("anthropic/") and any(
                token in text for token in ("sonnet", "opus", "claude")
            )
        if family == "google":
            return model_id.startswith("google/") and "gemini" in text and "pro" in text
        if family == "deepseek":
            return model_id.startswith("deepseek/")
        if family == "xai":
            return (model_id.startswith(("x-ai/", "xai/")) or "grok" in text) and "grok" in text
        if family == "qwen":
            return model_id.startswith("qwen/")
        if family == "llama":
            return model_id.startswith("meta-llama/") and any(
                token in text for token in ("instruct", "maverick", "scout")
            )
        if family == "mistral":
            return model_id.startswith("mistralai/") or "mistral" in text
        return False

    return [
        model
        for model in models
        if match(model) and not _is_deprecated_or_alias(model) and _is_text_chat_candidate(model)
    ]


def _family_strength(model: dict[str, Any], family: str) -> int:
    text = _id_name_text(model)
    if family == "openai":
        if "pro" in text:
            return 4
        if "chat" in text:
            return 3
        if "mini" in text or "nano" in text:
            return 1
        return 2
    if family == "anthropic":
        if "sonnet" in text:
            return 4
        if "opus" in text:
            return 3
        return 2
    if family == "google":
        if "pro" in text and "preview" not in text:
            return 4
        if "pro" in text:
            return 3
        return 1
    if family == "deepseek":
        if "pro" in text:
            return 4
        if "r1" in text:
            return 3
        return 2
    if family == "xai":
        if "multi-agent" in text:
            return 2
        return 4
    if family == "qwen":
        if "max" in text:
            return 4
        if "plus" in text:
            return 3
        if "flash" in text:
            return 1
        return 2
    if family == "llama":
        if "maverick" in text:
            return 4
        if "scout" in text:
            return 3
        return 2
    if family == "mistral":
        if "medium" in text:
            return 4
        if "large" in text:
            return 3
        if "small" in text:
            return 2
        return 1
    return 1


def _model_rank(model: dict[str, Any], family: str) -> tuple[int, int, int, int, int]:
    text = _lower_text(model)
    stable = 0 if any(token in text for token in ("preview", "beta", "experimental")) else 1
    structured = 1 if _structured_outputs_supported(model) else 0
    return (stable, _family_strength(model, family), _created(model), structured, _context_length(model))


def _select_family(models: Sequence[dict[str, Any]], family: str) -> Optional[dict[str, Any]]:
    candidates = _family_candidates(models, family)
    if not candidates:
        return None
    return sorted(candidates, key=lambda model: _model_rank(model, family), reverse=True)[0]


def _small_fast_model(models: Sequence[dict[str, Any]], selected_ids: set[str]) -> Optional[dict[str, Any]]:
    candidates = [
        model
        for model in models
        if str(model.get("id", "")) not in selected_ids
        and not _is_deprecated_or_alias(model)
        and _is_text_chat_candidate(model)
        and _model_price(model) >= 0
        and not str(model.get("id", "")).lower().endswith(":free")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda model: (_model_price(model), -_context_length(model), -_created(model)))
    return candidates[0]


def select_models(models: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    requested = [
        ("openai", "latest current OpenAI reasoning/chat model available in the live list"),
        ("anthropic", "latest current Anthropic Claude Sonnet/Opus model available in the live list"),
        ("google", "latest current Google Gemini Pro model available in the live list"),
        ("deepseek", "latest current DeepSeek reasoning/chat model available in the live list"),
        ("xai", "latest current xAI Grok model available in the live list"),
        ("qwen", "latest current Qwen instruct/reasoning model available in the live list"),
        ("llama", "latest current Llama instruct model available in the live list"),
        ("mistral", "latest current Mistral instruct model available in the live list"),
    ]
    for family, note in requested:
        model = _select_family(models, family)
        if model is None:
            continue
        model_id = str(model.get("id", ""))
        if not model_id or model_id in selected_ids:
            continue
        selected_ids.add(model_id)
        selections.append(_selected_model_record(model, family, note))
    small = _small_fast_model(models, selected_ids)
    if small is not None:
        selections.append(
            _selected_model_record(
                small,
                "small_fast",
                "lowest-priced current explicit model in the live list after family selections",
            )
        )
    return selections[:10]


def _selected_model_record(model: dict[str, Any], family: str, note: str) -> dict[str, Any]:
    model_id = str(model.get("id", ""))
    return {
        "model_id": model_id,
        "provider": _provider(model_id),
        "family": family,
        "name": model.get("name"),
        "context_length": model.get("context_length"),
        "pricing": model.get("pricing"),
        "structured_outputs_supported": _structured_outputs_supported(model),
        "selected_for": family,
        "selection_note": note,
    }


def load_model_config(path: Union[str, Path]) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        raise ValueError("models config must contain a list or a {'models': [...]} object")
    result = []
    for item in models:
        if not isinstance(item, dict) or not item.get("model_id"):
            raise ValueError("each selected model must be an object with model_id")
        result.append(item)
    return result


def safe_model_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return slug.strip("_") or "model"


def _common_case_fields(case: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = {
        "benchmark_id": case.get("benchmark_id"),
        "track": case.get("track"),
        "source_database": case.get("source_database"),
        "release_mode": case.get("release_mode"),
    }
    if mode == "tools_high_reasoning":
        payload.update(
            {
                "source_id": case.get("source_id"),
                "source_url": case.get("source_url"),
                "citation": case.get("citation"),
                "source_version": case.get("source_version"),
                "retrieval_date": case.get("retrieval_date"),
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _sanitize_human_effect(case: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = _common_case_fields(case, mode)
    payload.update(
        {
            "peptide": case.get("peptide"),
            "claim_text": case.get("claim_text"),
            "target": case.get("target"),
        }
    )
    if mode == "tools_high_reasoning":
        payload.update(
            {
                "source_evidence_type": case.get("source_evidence_type"),
                "source_result_count": case.get("source_result_count"),
                "trial_status": case.get("trial_status"),
                "trial_phase": case.get("trial_phase"),
                "trial_has_results": case.get("trial_has_results"),
                "mechanism_support": case.get("mechanism_support"),
            }
        )
    return _strip_forbidden(payload)


def _sanitize_binding_rank(case: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = _common_case_fields(case, mode)
    payload.update(
        {
            "panel_id": case.get("panel_id"),
            "target_id": case.get("target_id"),
            "target_name": case.get("target_name"),
            "assay_type": case.get("assay_type"),
            "assay_unit": case.get("assay_unit"),
            "assay_conditions": case.get("assay_conditions"),
            "measurement_direction": case.get("measurement_direction"),
            "normalization_method": case.get("normalization_method"),
            "candidate_peptides": [
                {
                    "item_id": item.get("item_id"),
                    "peptide": item.get("peptide"),
                }
                for item in case.get("items", [])
                if isinstance(item, dict)
            ],
        }
    )
    if mode == "tools_high_reasoning":
        payload["source_ids"] = case.get("source_ids", [])
    return _strip_forbidden(payload)


def _sanitize_pose(case: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = _common_case_fields(case, mode)
    payload.update(
        {
            "pdb_id": case.get("pdb_id"),
            "target_chain_id": case.get("target_chain_id"),
            "peptide_chain_id": case.get("peptide_chain_id"),
            "peptide_length": case.get("peptide_length"),
            "target_length": case.get("target_length"),
            "peptide": case.get("peptide"),
            "target": case.get("target"),
            "pose_subset": case.get("pose_subset"),
            "contact_label_status": case.get("contact_label_status"),
        }
    )
    if mode == "tools_high_reasoning":
        payload.update(
            {
                "coordinate_reference": case.get("coordinate_reference"),
                "contact_map_method": case.get("contact_map_method"),
            }
        )
    return _strip_forbidden(payload)


def _sanitize_structure(case: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = _common_case_fields(case, mode)
    payload.update(
        {
            "peptide": case.get("peptide"),
            "structure_id": case.get("structure_id"),
            "chain_id": case.get("chain_id"),
            "entity_id": case.get("entity_id"),
            "peptide_length": case.get("peptide_length"),
            "structure_file_url": case.get("structure_file_url"),
            "experimental_method": case.get("experimental_method"),
            "resolution_angstrom": case.get("resolution_angstrom"),
            "scoring_subset": case.get("scoring_subset"),
        }
    )
    if mode == "tools_high_reasoning":
        payload["gold_structure_reference"] = case.get("gold_structure_reference")
    return _strip_forbidden(payload)


def _strip_forbidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(item)
            for key, item in value.items()
            if key not in FORBIDDEN_INPUT_FIELDS and item is not None
        }
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value]
    return value


def sanitize_case_for_leaderboard(case: dict[str, Any], mode: str) -> dict[str, Any]:
    track = case.get("track")
    if mode not in {"base", "tools_high_reasoning"}:
        raise ValueError(f"unsupported leaderboard mode: {mode}")
    if track == "human_effect":
        return _sanitize_human_effect(case, mode)
    if track == "binding_rank":
        return _sanitize_binding_rank(case, mode)
    if track == "pose":
        return _sanitize_pose(case, mode)
    if track == "structure":
        return _sanitize_structure(case, mode)
    raise ValueError(f"unsupported track: {track}")


def inspect_leaderboard_input(
    release_dir: Union[str, Path], track: str, mode: str, output: Union[str, Path]
) -> int:
    records = _cases_for_track(Path(release_dir), track)
    sanitized = [sanitize_case_for_leaderboard(record, mode) for record in records]
    _assert_no_forbidden_inputs(sanitized)
    write_jsonl(output, sanitized)
    return len(sanitized)


def _assert_no_forbidden_inputs(records: Iterable[dict[str, Any]]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FORBIDDEN_INPUT_FIELDS:
                    raise ValueError(f"forbidden leaderboard input field at {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for record in records:
        walk(record, "$")


def _cases_for_track(release_dir: Path, track: str) -> list[dict[str, Any]]:
    path = release_dir / track / "cases.jsonl"
    records = read_jsonl(path)
    if track == "pose":
        return [
            record
            for record in records
            if record.get("contact_label_status") == "computed_contacts"
            and record.get("native_contacts")
        ]
    return records


def _chunked(records: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(records), size):
        yield list(records[index : index + size])


def _enum_values(enum_cls: Any) -> list[str]:
    return [item.value for item in enum_cls]


def _track_instructions(track: str) -> str:
    if track == "human_effect":
        return (
            "For each input, predict category, evidence_level, evidence_direction, "
            "claim_status, safety_status, abstained, and rationale_source_ids. "
            f"category enum: {_enum_values(HumanEffectCategory)}. "
            f"evidence_level enum: {_enum_values(EvidenceLevel)}. "
            f"evidence_direction enum: {_enum_values(EvidenceDirection)}. "
            f"claim_status enum: {_enum_values(ClaimStatus)}. "
            f"safety_status enum: {_enum_values(SafetyStatus)}."
        )
    if track == "binding_rank":
        return (
            "For each panel, return exactly one prediction object. Put candidate scores inside that "
            "prediction object's scores list. Higher score must mean stronger predicted binding or "
            "activity within that panel. Include rank values when possible."
        )
    if track == "pose":
        return (
            "For each case, predict residue contacts in the predicted_contacts field using objects "
            "with target_residue, peptide_residue, and optional distance_angstrom. Residue labels "
            "must use '<chain>:<residue>'. If contacts cannot be inferred from the allowed input, "
            "return an empty predicted_contacts list."
        )
    if track == "structure":
        return (
            "For each case, predict coordinate objects only when coordinates can be inferred from allowed "
            "input. Otherwise return an empty coordinates list with a calibrated confidence."
        )
    raise ValueError(f"unsupported track: {track}")


def _prompt(track: str, mode: str, cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    user_payload = {"track": track, "mode": mode, "cases": cases}
    return [
        {
            "role": "system",
            "content": (
                "You are evaluating peptide engineering benchmark cases. "
                "Use only the supplied fields. Return valid JSON only. "
                "Do not include prose, markdown, or keys outside the requested prediction objects."
            ),
        },
        {
            "role": "user",
            "content": (
                _track_instructions(track)
                + " Return a JSON object with a single key 'predictions'. "
                + "Each prediction must include prediction_id, benchmark_id, track, model_name, and track fields. "
                + "Use model_name exactly as supplied in the cases wrapper if present; otherwise use the requested model ID. "
                + json.dumps(user_payload, separators=(",", ":"), sort_keys=True)
            ),
        },
    ]


def _compact_retry_prompt(track: str, mode: str, case: dict[str, Any]) -> list[dict[str, str]]:
    sanitized = sanitize_case_for_leaderboard(case, mode)
    benchmark_id = str(case["benchmark_id"])
    if track == "binding_rank":
        items = [
            {
                "item_id": item.get("item_id"),
                "sequence": (item.get("peptide") or {}).get("sequence"),
            }
            for item in case.get("items", [])
            if isinstance(item, dict)
        ]
        payload = {
            "benchmark_id": benchmark_id,
            "target": sanitized.get("target_name") or sanitized.get("target_id"),
            "items": items,
        }
        instruction = (
            "Return minified JSON only: {\"scores\":[number,...]}. "
            "The scores array must contain exactly one numeric score per input item in order. "
            "Higher score means stronger predicted binding."
        )
    elif track == "human_effect":
        payload = {
            key: sanitized.get(key)
            for key in (
                "benchmark_id",
                "source_database",
                "peptide_name",
                "drug_name",
                "target_name",
                "claim_text",
                "claimed_effect",
            )
            if sanitized.get(key) is not None
        }
        instruction = (
            "Return minified JSON only with keys category,evidence_level,evidence_direction,"
            "claim_status,safety_status. Valid category values: "
            f"{_enum_values(HumanEffectCategory)}. Valid evidence_level values: "
            f"{_enum_values(EvidenceLevel)}. Valid evidence_direction values: "
            f"{_enum_values(EvidenceDirection)}. Valid claim_status values: "
            f"{_enum_values(ClaimStatus)}. Valid safety_status values: "
            f"{_enum_values(SafetyStatus)}."
        )
    elif track == "pose":
        payload = {
            key: sanitized.get(key)
            for key in (
                "benchmark_id",
                "source_database",
                "target_chain",
                "peptide_chain",
                "target_length",
                "peptide_length",
                "contact_label_status",
            )
            if sanitized.get(key) is not None
        }
        instruction = (
            "Return minified JSON only with keys predicted_contacts,binding_site_residues,"
            "orientation_label,clash_score. Use [] when contacts cannot be inferred."
        )
    elif track == "structure":
        payload = {
            key: sanitized.get(key)
            for key in (
                "benchmark_id",
                "source_database",
                "source_id",
                "chain_id",
                "sequence",
                "structure_source_type",
            )
            if sanitized.get(key) is not None
        }
        instruction = (
            "Return minified JSON only with keys coordinates,confidence. Do not invent atom "
            "coordinates from source-reference metadata. Unless explicit atom coordinates are in "
            "the input, return exactly {\"coordinates\":[],\"confidence\":0.01}."
        )
    else:
        raise ValueError(f"unsupported track: {track}")
    return [
        {
            "role": "system",
            "content": "Return only a single minified JSON object. No markdown, no prose.",
        },
        {
            "role": "user",
            "content": instruction + " Input: " + json.dumps(payload, separators=(",", ":")),
        },
    ]


def _compact_retry_batch_prompt(
    track: str,
    mode: str,
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    payload_cases: list[dict[str, Any]] = []
    for case in cases:
        sanitized = sanitize_case_for_leaderboard(case, mode)
        benchmark_id = str(case["benchmark_id"])
        if track == "binding_rank":
            payload_cases.append(
                {
                    "benchmark_id": benchmark_id,
                    "target": sanitized.get("target_name") or sanitized.get("target_id"),
                    "items": [
                        {
                            "item_id": item.get("item_id"),
                            "sequence": (item.get("peptide") or {}).get("sequence"),
                        }
                        for item in case.get("items", [])
                        if isinstance(item, dict)
                    ],
                }
            )
        elif track == "human_effect":
            payload_cases.append(
                {
                    key: sanitized.get(key)
                    for key in (
                        "benchmark_id",
                        "source_database",
                        "peptide_name",
                        "drug_name",
                        "target_name",
                        "claim_text",
                        "claimed_effect",
                    )
                    if sanitized.get(key) is not None
                }
            )
        elif track == "pose":
            payload_cases.append(
                {
                    key: sanitized.get(key)
                    for key in (
                        "benchmark_id",
                        "source_database",
                        "target_chain",
                        "peptide_chain",
                        "target_length",
                        "peptide_length",
                        "contact_label_status",
                    )
                    if sanitized.get(key) is not None
                }
            )
        elif track == "structure":
            payload_cases.append(
                {
                    key: sanitized.get(key)
                    for key in (
                        "benchmark_id",
                        "source_database",
                        "source_id",
                        "chain_id",
                        "sequence",
                        "structure_source_type",
                    )
                    if sanitized.get(key) is not None
                }
            )
        else:
            raise ValueError(f"unsupported track: {track}")
    if track == "binding_rank":
        instruction = (
            "Return minified JSON only: {\"predictions\":[{\"benchmark_id\":\"...\","
            "\"scores\":[number,...]}]}. Each scores array must match that case's input item order."
        )
    elif track == "human_effect":
        instruction = (
            "Return minified JSON only with key predictions. Each prediction must include "
            "benchmark_id,category,evidence_level,evidence_direction,claim_status,safety_status. "
            f"Valid category values: {_enum_values(HumanEffectCategory)}. "
            f"Valid evidence_level values: {_enum_values(EvidenceLevel)}. "
            f"Valid evidence_direction values: {_enum_values(EvidenceDirection)}. "
            f"Valid claim_status values: {_enum_values(ClaimStatus)}. "
            f"Valid safety_status values: {_enum_values(SafetyStatus)}."
        )
    elif track == "pose":
        instruction = (
            "Return minified JSON only with key predictions. Each prediction must include "
            "benchmark_id,predicted_contacts,binding_site_residues,orientation_label,clash_score. "
            "Use [] when contacts cannot be inferred and nonnegative clash_score."
        )
    elif track == "structure":
        instruction = (
            "Return minified JSON only with key predictions. Each prediction must include "
            "benchmark_id,coordinates,confidence. Do not invent atom coordinates from source-reference "
            "metadata; use {\"coordinates\":[],\"confidence\":0.01} for those cases."
        )
    return [
        {
            "role": "system",
            "content": "Return only a single minified JSON object. No markdown, no prose.",
        },
        {
            "role": "user",
            "content": instruction + " Input: " + json.dumps(payload_cases, separators=(",", ":")),
        },
    ]


def _compact_payload_to_prediction(
    payload: dict[str, Any],
    *,
    track: str,
    model_id: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    benchmark_id = str(case["benchmark_id"])
    base = {
        "prediction_id": f"{safe_model_slug(model_id)}:{benchmark_id}",
        "benchmark_id": benchmark_id,
        "track": track,
        "model_name": model_id,
    }
    if track == "binding_rank":
        scores = payload.get("scores")
        if not isinstance(scores, list):
            raise ValueError("compact binding_rank response has no scores list")
        items = case.get("items")
        if not isinstance(items, list) or len(scores) < len(items):
            raise ValueError("compact binding_rank response has the wrong number of scores")
        base["scores"] = [
            {
                "item_id": item["item_id"],
                "score": float(scores[index]),
                "rank": index + 1,
            }
            for index, item in enumerate(items)
            if isinstance(item, dict) and item.get("item_id")
        ]
    elif track == "human_effect":
        base.update(payload)
    elif track == "pose":
        try:
            clash_score = float(payload.get("clash_score") or 0.0)
        except (TypeError, ValueError):
            clash_score = 0.0
        predicted_contacts = payload.get("predicted_contacts") or payload.get("contacts") or []
        binding_site_residues = payload.get("binding_site_residues") or []
        if not isinstance(predicted_contacts, list):
            predicted_contacts = []
        if not isinstance(binding_site_residues, list):
            binding_site_residues = []
        orientation_label = payload.get("orientation_label")
        if orientation_label is not None and not isinstance(orientation_label, str):
            orientation_label = str(orientation_label)
        base.update(
            {
                "predicted_contacts": predicted_contacts,
                "binding_site_residues": binding_site_residues,
                "orientation_label": orientation_label,
                "clash_score": max(0.0, clash_score),
            }
        )
    elif track == "structure":
        coordinates = payload.get("coordinates") or []
        if not isinstance(coordinates, list):
            coordinates = []
        try:
            confidence = float(payload.get("confidence", 0.01))
        except (TypeError, ValueError):
            confidence = 0.01
        if not coordinates and confidence == 0.0:
            confidence = 0.01
        base.update(
            {
                "coordinates": coordinates,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    else:
        raise ValueError(f"unsupported track: {track}")
    validated = validate_prediction_record(base)
    return validated.model_dump(mode="json", exclude_none=True)


def _compact_batch_payload_to_predictions(
    payload: dict[str, Any],
    *,
    track: str,
    model_id: str,
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("compact batch response has no predictions list")
    cases_by_id = {str(case["benchmark_id"]): case for case in cases}
    parsed_by_id: dict[str, dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("compact batch prediction is not an object")
        benchmark_id = str(item.get("benchmark_id") or "")
        case = cases_by_id.get(benchmark_id)
        if case is None:
            raise ValueError(f"compact batch prediction has unknown benchmark_id: {benchmark_id}")
        parsed_by_id[benchmark_id] = _compact_payload_to_prediction(
            item,
            track=track,
            model_id=model_id,
            case=case,
        )
    missing = set(cases_by_id) - set(parsed_by_id)
    if missing:
        raise ValueError(f"compact batch missed {len(missing)} predictions")
    return [parsed_by_id[str(case["benchmark_id"])] for case in cases]


def _chat_completion(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    timeout: int,
    use_response_format: bool,
    max_tokens: int = 8192,
    low_reasoning: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if low_reasoning:
        payload["include_reasoning"] = False
        payload["reasoning"] = {"effort": "low"}
    return _request_json(
        OPENROUTER_CHAT_URL,
        method="POST",
        payload=payload,
        timeout=timeout,
        response_format=use_response_format,
    )


def _extract_message(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("completion choice is not an object")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("completion message is not an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("completion content is not text")
    return content


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("prediction response must be a JSON object")
    return payload


def _normalize_binding_rank_prediction_shape(
    predictions: list[Any],
    cases_by_id: Optional[dict[str, dict[str, Any]]] = None,
) -> list[Any]:
    normalized_predictions: list[Any] = []
    for item in predictions:
        if not isinstance(item, dict) or not isinstance(item.get("scores"), list):
            normalized_predictions.append(item)
            continue
        scores = item["scores"]
        if all(isinstance(score, dict) for score in scores):
            normalized_predictions.append(item)
            continue
        case = (cases_by_id or {}).get(str(item.get("benchmark_id") or ""))
        case_items = case.get("items") if isinstance(case, dict) else None
        if not isinstance(case_items, list) or len(scores) < len(case_items):
            normalized_predictions.append(item)
            continue
        repaired_scores = []
        can_repair = True
        for index, score in enumerate(scores[: len(case_items)]):
            try:
                value = float(score)
            except (TypeError, ValueError):
                can_repair = False
                break
            case_item = case_items[index]
            if not isinstance(case_item, dict) or not case_item.get("item_id"):
                can_repair = False
                break
            repaired_scores.append(
                {
                    "item_id": case_item["item_id"],
                    "score": value,
                    "rank": index + 1,
                }
            )
        if can_repair:
            item = dict(item)
            item["scores"] = repaired_scores
        normalized_predictions.append(item)
    predictions = normalized_predictions
    flat_items = [
        item
        for item in predictions
        if isinstance(item, dict) and "scores" not in item and "benchmark_id" in item
    ]
    if not flat_items:
        return predictions
    if not all(
        "item_id" in item and "score" in item
        for item in flat_items
    ):
        return predictions
    grouped: dict[str, dict[str, Any]] = {}
    passthrough: list[Any] = []
    for item in predictions:
        if not isinstance(item, dict) or "scores" in item:
            passthrough.append(item)
            continue
        benchmark_id = str(item.get("benchmark_id") or "")
        if not benchmark_id:
            passthrough.append(item)
            continue
        group = grouped.setdefault(
            benchmark_id,
            {
                "prediction_id": item.get("prediction_id") or f"prediction:{benchmark_id}",
                "benchmark_id": benchmark_id,
                "track": item.get("track") or "binding_rank",
                "scores": [],
            },
        )
        score_item: dict[str, Any] = {
            "item_id": item["item_id"],
            "score": item["score"],
        }
        if item.get("rank") is not None:
            score_item["rank"] = item["rank"]
        group["scores"].append(score_item)
    return passthrough + list(grouped.values())


def _prediction_records_from_response(
    payload: dict[str, Any],
    *,
    track: str,
    model_id: str,
    expected_ids: set[str],
    cases_by_id: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("prediction response has no predictions list")
    if track == "binding_rank":
        predictions = _normalize_binding_rank_prediction_shape(predictions, cases_by_id)
    records = []
    for index, record in enumerate(predictions):
        if not isinstance(record, dict):
            raise ValueError(f"prediction {index} is not an object")
        item = dict(record)
        item["model_name"] = model_id
        if track == "pose" and "predicted_contacts" not in item and "contacts" in item:
            item["predicted_contacts"] = item.pop("contacts")
        if track == "binding_rank":
            item.pop("ranks", None)
            item.pop("ranking", None)
        if item.get("benchmark_id") not in expected_ids:
            raise ValueError(f"prediction has unknown benchmark_id: {item.get('benchmark_id')}")
        item.setdefault("prediction_id", f"{safe_model_slug(model_id)}:{item['benchmark_id']}")
        item.setdefault("track", track)
        validated = validate_prediction_record(item)
        records.append(validated.model_dump(mode="json", exclude_none=True))
    return records


def _usage_cost(payload: dict[str, Any], model: dict[str, Any]) -> float:
    usage = payload.get("usage") or {}
    if isinstance(usage, dict):
        for key in ("cost", "total_cost"):
            try:
                return float(usage[key])
            except (KeyError, TypeError, ValueError):
                pass
        pricing = model.get("pricing") or {}
        if isinstance(pricing, dict):
            prompt_tokens = float(usage.get("prompt_tokens") or 0)
            completion_tokens = float(usage.get("completion_tokens") or 0)
            return (
                prompt_tokens * _price(pricing.get("prompt"))
                + completion_tokens * _price(pricing.get("completion"))
            )
    return 0.0


def _fallback_predictions(track: str, cases: Sequence[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
    typed_cases = [validate_case_record(record) for record in cases]
    if track == "pose":
        return [
            PosePrediction(
                prediction_id=f"{safe_model_slug(model_id)}:{case.benchmark_id}",
                benchmark_id=case.benchmark_id,
                model_name=model_id,
                predicted_contacts=[],
                binding_site_residues=[],
                clash_score=0.0,
            ).model_dump(mode="json", exclude_none=True)
            for case in typed_cases
            if isinstance(case, PoseCase)
        ]
    if track == "structure":
        return [
            StructurePrediction(
                prediction_id=f"{safe_model_slug(model_id)}:{case.benchmark_id}",
                benchmark_id=case.benchmark_id,
                model_name=model_id,
                coordinates=[],
                confidence=0.0,
            ).model_dump(mode="json", exclude_none=True)
            for case in typed_cases
            if isinstance(case, StructureCase)
        ]
    track_value = Track(track)
    return make_baseline_predictions(track_value, typed_cases, model_name=model_id)


def _run_track_predictions(
    *,
    model: dict[str, Any],
    mode: str,
    track: str,
    cases: list[dict[str, Any]],
    output: Path,
    temperature: float,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    model_id = model["model_id"]
    all_predictions: list[dict[str, Any]] = []
    invalid_json_count = 0
    api_error_count = 0
    cost = 0.0
    use_response_format = bool(model.get("structured_outputs_supported"))
    remote_disabled = False

    for chunk in _chunked(cases, PREDICTION_CHUNK_SIZE):
        if remote_disabled:
            all_predictions.extend(_fallback_predictions(track, chunk, model_id))
            continue
        sanitized = [sanitize_case_for_leaderboard(case, mode) for case in chunk]
        _assert_no_forbidden_inputs(sanitized)
        expected_ids = {case["benchmark_id"] for case in sanitized}
        cases_by_id = {case["benchmark_id"]: case for case in chunk}
        chunk_predictions: Optional[list[dict[str, Any]]] = None
        attempts = retries + 1
        for attempt in range(attempts):
            try:
                payload = _chat_completion(
                    model_id,
                    _prompt(track, mode, sanitized),
                    temperature=temperature,
                    timeout=timeout,
                    use_response_format=use_response_format,
                )
                cost += _usage_cost(payload, model)
                response_json = _extract_json_object(_extract_message(payload))
                parsed = _prediction_records_from_response(
                    response_json,
                    track=track,
                    model_id=model_id,
                    expected_ids=expected_ids,
                    cases_by_id=cases_by_id,
                )
                parsed_by_id = {record["benchmark_id"]: record for record in parsed}
                if expected_ids - set(parsed_by_id):
                    raise ValueError(f"missing predictions for {len(expected_ids - set(parsed_by_id))} cases")
                chunk_predictions = [parsed_by_id[case["benchmark_id"]] for case in sanitized]
                break
            except OpenRouterInsufficientCredits:
                raise
            except OpenRouterError as exc:
                api_error_count += 1
                if use_response_format and "response_format" in str(exc).lower():
                    use_response_format = False
                if attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 8))
            except (json.JSONDecodeError, ValueError):
                invalid_json_count += 1
                if attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 8))
        if chunk_predictions is None:
            all_predictions.extend(_fallback_predictions(track, chunk, model_id))
            remote_disabled = True
        else:
            all_predictions.extend(chunk_predictions)
    write_jsonl(output, all_predictions)
    return {
        "predictions": all_predictions,
        "invalid_json_count": invalid_json_count,
        "api_error_count": api_error_count,
        "total_cost_estimate_if_available": cost,
    }


def _evaluate_track(track: str, cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    typed_cases = [validate_case_record(case) for case in cases]
    typed_predictions = [validate_prediction_record(prediction) for prediction in predictions]
    if track == "structure":
        result = evaluate_structure(
            [case for case in typed_cases if isinstance(case, StructureCase)],
            [item for item in typed_predictions if isinstance(item, StructurePrediction)],
        )
    elif track == "pose":
        result = evaluate_pose(
            [case for case in typed_cases if isinstance(case, PoseCase)],
            [item for item in typed_predictions if isinstance(item, PosePrediction)],
        )
    elif track == "binding_rank":
        result = evaluate_binding_rank(
            [case for case in typed_cases if isinstance(case, BindingRankCase)],
            [item for item in typed_predictions if isinstance(item, BindingRankPrediction)],
        )
    elif track == "human_effect":
        result = evaluate_human_effect(
            [case for case in typed_cases if isinstance(case, HumanEffectCase)],
            [item for item in typed_predictions if isinstance(item, HumanEffectPrediction)],
        )
    else:
        raise ValueError(f"unsupported track: {track}")
    return result.model_dump(mode="json")


def _track_score(track: str, metrics: dict[str, Any], n_cases: int) -> Optional[float]:
    if track == "human_effect":
        overclaim = float(metrics.get("overclaim_penalty") or 0) / max(n_cases, 1)
        base = (
            float(metrics.get("category_macro_f1") or 0)
            + float(metrics.get("evidence_level_ordinal_accuracy") or 0)
            + float(metrics.get("evidence_direction_accuracy") or 0)
            + float(metrics.get("claim_status_accuracy") or 0)
            + float(metrics.get("unsupported_claim_detection_rate") or 0)
        ) / 5
        return max(0.0, base - overclaim)
    if track == "binding_rank":
        spearman = (float(metrics.get("spearman") or 0) + 1) / 2
        kendall = (float(metrics.get("kendall_tau") or 0) + 1) / 2
        pairwise = float(metrics.get("pairwise_ranking_accuracy") or 0)
        return (spearman + kendall + pairwise) / 3
    if track == "pose":
        precision = float(metrics.get("interface_contact_precision") or 0)
        recall = float(metrics.get("interface_contact_recall") or 0)
        return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    if track == "structure":
        value = metrics.get("all_atom_rmsd")
        return 1 / (1 + float(value)) if isinstance(value, (int, float)) else None
    return None


def _empty_metrics() -> dict[str, Any]:
    return {
        "human_effect_category_macro_f1": None,
        "human_effect_evidence_accuracy": None,
        "human_effect_overclaim_penalty": None,
        "binding_rank_spearman": None,
        "binding_rank_kendall": None,
        "binding_rank_pairwise_accuracy": None,
        "pose_contact_precision": None,
        "pose_contact_recall": None,
        "pose_contact_f1": None,
        "structure_score": None,
        "structure_rmsd_status": None,
    }


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coverage_fraction(row: dict[str, Any]) -> float:
    coverage = row.get("coverage")
    attempted = 0
    completed = 0
    if isinstance(coverage, dict):
        for item in coverage.values():
            if not isinstance(item, dict):
                continue
            attempted += int(item.get("attempted") or 0)
            completed += int(item.get("completed") or 0)
    if attempted == 0:
        attempted = int(row.get("total_cases_attempted") or 0)
        completed = int(row.get("total_cases_completed") or 0)
    if attempted == 0:
        return 0.0
    return max(0.0, min(1.0, completed / attempted))


def _coverage_adjusted_score(row: dict[str, Any]) -> Optional[float]:
    if row.get("scored") is False:
        return None
    existing = _as_float(row.get("coverage_adjusted_score"))
    if existing is not None:
        return existing
    mean_score = _as_float(row.get("mean_score"))
    if mean_score is None:
        return None
    valid_prediction_rate = _as_float(row.get("valid_prediction_rate"))
    if valid_prediction_rate is None:
        valid_prediction_rate = 1.0
    return mean_score * _coverage_fraction(row) * max(0.0, min(1.0, valid_prediction_rate))


def _int_count(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _strict_counts(row: dict[str, Any]) -> dict[str, int]:
    retry_queue_size = _int_count(row, "retry_queue_size")
    retry_recovered_count = _int_count(row, "retry_recovered_count")
    fallback_count = _int_count(row, "fallback_prediction_count")
    if fallback_count == 0 and retry_queue_size:
        fallback_count = max(0, retry_queue_size - retry_recovered_count)
    provider_count = _int_count(row, "unresolved_provider_error_count")
    if provider_count == 0:
        provider_count = _int_count(row, "post_retry_api_error_count") or _int_count(row, "api_error_count")
    invalid_count = _int_count(row, "unresolved_invalid_json_count")
    if invalid_count == 0:
        invalid_count = _int_count(row, "post_retry_invalid_json_count") or _int_count(row, "invalid_json_count")
    return {
        "fallback_prediction_count": fallback_count,
        "unresolved_provider_error_count": provider_count,
        "unresolved_invalid_json_count": invalid_count,
    }


def _strict_row_status(row: dict[str, Any]) -> str:
    if _is_noncompetitive_sanity(row):
        return "noncompetitive_oracle"
    if bool(row.get("is_baseline")):
        return "noncompetitive_baseline"
    counts = _strict_counts(row)
    if counts["unresolved_provider_error_count"] or counts["fallback_prediction_count"]:
        return "excluded_provider_error"
    if counts["unresolved_invalid_json_count"]:
        if _row_has_accounted_failures(row):
            return "completed_with_failures"
        return "excluded_invalid_json"
    if _coverage_fraction(row) < 1.0:
        return "excluded_incomplete"
    return "clean_completed"


def _row_has_accounted_failures(row: dict[str, Any]) -> bool:
    counts = _strict_counts(row)
    failed_prediction_count = _int_count(row, "failed_prediction_count")
    valid_prediction_rate = _as_float(row.get("valid_prediction_rate"))
    return (
        counts["unresolved_invalid_json_count"] > 0
        and failed_prediction_count >= counts["unresolved_invalid_json_count"]
        and valid_prediction_rate is not None
        and counts["unresolved_provider_error_count"] == 0
        and counts["fallback_prediction_count"] == 0
        and _coverage_fraction(row) >= 1.0
    )


def _is_rankable_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("competitive") is True
        and row.get("scored") is True
        and row.get("row_status") in RANKABLE_ROW_STATUSES
    )


def _apply_strict_fields(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    counts = _strict_counts(item)
    item.update(counts)
    row_status = _strict_row_status(item)
    item["row_status"] = row_status
    item["competitive"] = row_status in RANKABLE_ROW_STATUSES
    item["scored"] = row_status in RANKABLE_ROW_STATUSES
    if not item["scored"]:
        item["coverage_adjusted_score"] = None
    return item


def _metric_display_value(value: Any) -> Any:
    return value if isinstance(value, (int, float)) else None


def _binding_rank_display(row: dict[str, Any]) -> Any:
    kendall = _metric_display_value(row.get("binding_rank_kendall"))
    return kendall if kendall is not None else _metric_display_value(row.get("binding_rank_spearman"))


def _structure_display(row: dict[str, Any]) -> Any:
    value = _metric_display_value(row.get("structure_score"))
    return value if value is not None else "not computed"


def _is_noncompetitive_sanity(row: dict[str, Any]) -> bool:
    model_id = str(row.get("model_id") or "").lower()
    oracle_like = "oracle" in model_id and "non_oracle" not in model_id and "non-oracle" not in model_id
    return bool(oracle_like or row.get("mode") == "sanity_check")


def _status_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    row_status = str(row.get("row_status") or "")
    if row_status.startswith("excluded"):
        tags.append(row_status)
    elif row_status == "completed_with_failures":
        tags.append(row_status)
    if (
        int(row.get("api_error_count") or 0)
        or int(row.get("invalid_json_count") or 0)
        or int(row.get("unresolved_provider_error_count") or 0)
        or int(row.get("unresolved_invalid_json_count") or 0)
    ):
        tags.append("errors")
    if _coverage_fraction(row) < 1.0:
        tags.append("incomplete")
    if bool(row.get("is_baseline")):
        tags.append("baseline")
    if _is_noncompetitive_sanity(row):
        tags.append("non-competitive")
    if not tags:
        tags.append("clean")
    return tags


def _run_status(row: dict[str, Any]) -> str:
    if row.get("row_status") in ROW_STATUS_VALUES:
        return str(row["row_status"])
    if _coverage_fraction(row) < 1.0:
        return "excluded_incomplete"
    if int(row.get("api_error_count") or 0) or int(row.get("invalid_json_count") or 0):
        return "excluded_provider_error"
    if int(row.get("retry_recovered_count") or 0):
        return "clean_completed"
    return "clean_completed"


def _normalize_leaderboard_row(row: dict[str, Any]) -> dict[str, Any]:
    item = _apply_strict_fields(row)
    item["coverage_adjusted_score"] = _coverage_adjusted_score(item)
    item["coverage_fraction"] = _coverage_fraction(item)
    item["mode_label"] = MODE_LABELS.get(str(item.get("mode")), str(item.get("mode") or ""))
    item["status"] = ", ".join(_status_tags(item))
    item["leaderboard_status"] = _run_status(item)
    return item


def normalize_leaderboard_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_leaderboard_row(row) for row in rows]


def public_leaderboard_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_leaderboard_rows(rows)
    competitive_rows = [
        row
        for row in normalized
        if _is_rankable_row(row)
    ]
    competitive_rows.sort(
        key=lambda row: (
            row.get("coverage_adjusted_score") is not None,
            row.get("coverage_adjusted_score") or -math.inf,
            row.get("mean_score") or -math.inf,
        ),
        reverse=True,
    )
    rank_by_key = {
        (row.get("model_id"), row.get("mode")): index
        for index, row in enumerate(competitive_rows, start=1)
    }
    public_rows = []
    for row in normalized:
        public_rows.append(
            {
                "rank": rank_by_key.get((row.get("model_id"), row.get("mode"))),
                "model_id": row.get("model_id"),
                "provider": row.get("provider"),
                "mode": row.get("mode"),
                "mode_label": row.get("mode_label"),
                "mean_score": row.get("mean_score"),
                "coverage_adjusted_score": row.get("coverage_adjusted_score"),
                "human_effect": _metric_display_value(row.get("human_effect_category_macro_f1")),
                "binding_rank": _binding_rank_display(row),
                "pose": _metric_display_value(row.get("pose_contact_f1")),
                "structure": _structure_display(row),
                "coverage": row.get("coverage_fraction"),
                "api_error_count": int(row.get("api_error_count") or 0),
                "invalid_json_count": int(row.get("invalid_json_count") or 0),
                "status": row.get("status"),
                "leaderboard_status": row.get("leaderboard_status"),
                "is_baseline": bool(row.get("is_baseline")),
                "competitive": row.get("competitive") is True,
                "benchmark_release": row.get("benchmark_release"),
                "run_timestamp": row.get("run_timestamp"),
                "retry_recovered_count": int(row.get("retry_recovered_count") or 0),
                "retry_still_failed_count": int(row.get("retry_still_failed_count") or 0),
                "fallback_prediction_count": int(row.get("fallback_prediction_count") or 0),
                "unresolved_provider_error_count": int(
                    row.get("unresolved_provider_error_count") or 0
                ),
                "unresolved_invalid_json_count": int(
                    row.get("unresolved_invalid_json_count") or 0
                ),
                "row_status": row.get("row_status"),
                "scored": bool(row.get("scored")),
            }
        )
    public_rows.sort(
        key=lambda row: (
            row["rank"] is None,
            row["rank"] or 999_999,
            -(row.get("coverage_adjusted_score") or -math.inf),
            str(row.get("model_id") or ""),
        )
    )
    return public_rows


def write_public_leaderboard(leaderboard_dir: Union[str, Path]) -> list[dict[str, Any]]:
    path = Path(leaderboard_dir)
    rows = json.loads((path / "leaderboard.json").read_text(encoding="utf-8"))
    public_rows = public_leaderboard_rows(rows)
    write_text(path / "leaderboard_public.json", json.dumps(public_rows, indent=2, sort_keys=True))
    return public_rows


def write_static_leaderboard_page(leaderboard_dir: Union[str, Path]) -> list[dict[str, Any]]:
    path = Path(leaderboard_dir)
    path.mkdir(parents=True, exist_ok=True)
    assets = path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    public_rows = write_public_leaderboard(path)
    write_text(
        path / "index.html",
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Peptide Engineering Benchmark Leaderboard</title>
  <link rel="stylesheet" href="assets/leaderboard.css">
</head>
<body>
  <main class="page">
    <header class="header">
      <h1>Peptide Engineering Benchmark Leaderboard</h1>
      <p>Static leaderboard for included benchmark runs.</p>
    </header>
    <section class="controls" aria-label="Filters">
      <button type="button" data-filter="competitive" class="active">Competitive only</button>
      <button type="button" data-filter="all">All</button>
      <button type="button" data-filter="base">Base</button>
      <button type="button" data-filter="tools_high_reasoning">Tools + high reasoning</button>
      <button type="button" data-filter="baselines">Baselines</button>
    </section>
    <section class="summary" id="summary"></section>
    <div class="table-wrap">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th data-sort="rank">Rank</th>
            <th data-sort="model_id">Model</th>
            <th data-sort="mode_label">Mode</th>
            <th data-sort="mean_score">Mean</th>
            <th data-sort="coverage_adjusted_score">Coverage-adjusted mean</th>
            <th data-sort="human_effect">Human Effect</th>
            <th data-sort="binding_rank">Binding Rank</th>
            <th data-sort="pose">Pose</th>
            <th data-sort="structure">Structure</th>
            <th data-sort="coverage">Coverage</th>
            <th data-sort="api_error_count">API Errors</th>
            <th data-sort="invalid_json_count">Invalid JSON</th>
            <th data-sort="status">Status</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </main>
  <script src="assets/leaderboard.js"></script>
</body>
</html>
""",
    )
    write_text(
        assets / "leaderboard.css",
        """:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f8fa;
  color: #20242a;
}

body {
  margin: 0;
}

.page {
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 18px 40px;
}

.header {
  margin-bottom: 18px;
}

h1 {
  margin: 0 0 6px;
  font-size: clamp(1.7rem, 3vw, 2.4rem);
  letter-spacing: 0;
}

p {
  margin: 0;
  color: #53606f;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0;
}

button {
  border: 1px solid #c9d1dc;
  background: #fff;
  color: #20242a;
  border-radius: 6px;
  padding: 8px 11px;
  cursor: pointer;
}

button.active {
  background: #1f5f8b;
  border-color: #1f5f8b;
  color: #fff;
}

.summary {
  color: #53606f;
  font-size: 0.92rem;
  margin-bottom: 10px;
}

.table-wrap {
  overflow-x: auto;
  background: #fff;
  border: 1px solid #d8dee8;
  border-radius: 8px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 1120px;
}

th,
td {
  text-align: left;
  border-bottom: 1px solid #e6ebf1;
  padding: 10px 9px;
  font-size: 0.9rem;
  white-space: nowrap;
}

th {
  background: #f0f3f7;
  color: #303843;
  font-weight: 650;
  cursor: pointer;
}

tbody tr:last-child td {
  border-bottom: 0;
}

.status {
  display: inline-flex;
  gap: 5px;
  flex-wrap: wrap;
}

.tag {
  border-radius: 999px;
  padding: 3px 7px;
  font-size: 0.78rem;
  background: #edf1f6;
  color: #354150;
}

.tag.errors {
  background: #fee9e7;
  color: #9f2f21;
}

.tag.incomplete {
  background: #fff1d5;
  color: #7b4d00;
}

.tag.baseline,
.tag.non-competitive {
  background: #e7edf7;
  color: #24476f;
}

.tag.clean {
  background: #e3f5ec;
  color: #176a3a;
}
""",
    )
    write_text(
        assets / "leaderboard.js",
        """const MODE_LABELS = {
  base: "Base",
  tools_high_reasoning: "Tools + high reasoning",
  baseline: "Baseline",
  sanity_check: "Sanity check"
};

const state = {
  rows: [],
  filter: "competitive",
  sortKey: "coverage_adjusted_score",
  sortDir: "desc"
};

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function coverageFraction(row) {
  const coverage = row.coverage || {};
  let attempted = 0;
  let completed = 0;
  Object.values(coverage).forEach((item) => {
    attempted += Number(item.attempted || 0);
    completed += Number(item.completed || 0);
  });
  if (!attempted) {
    attempted = Number(row.total_cases_attempted || 0);
    completed = Number(row.total_cases_completed || 0);
  }
  return attempted ? Math.max(0, Math.min(1, completed / attempted)) : 0;
}

function metric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function statusTags(row) {
  const tags = [];
  if (row.row_status && row.row_status.startsWith("excluded")) tags.push(row.row_status);
  if (row.row_status === "completed_with_failures") tags.push(row.row_status);
  if (
    Number(row.api_error_count || 0) ||
    Number(row.invalid_json_count || 0) ||
    Number(row.unresolved_provider_error_count || 0) ||
    Number(row.unresolved_invalid_json_count || 0)
  ) tags.push("errors");
  if (row.coverage < 1) tags.push("incomplete");
  if (row.is_baseline) tags.push("baseline");
  const modelId = String(row.model_id || "").toLowerCase();
  const oracleLike = modelId.includes("oracle") && !modelId.includes("non_oracle") && !modelId.includes("non-oracle");
  if (row.competitive === false || oracleLike || row.mode === "sanity_check") tags.push("non-competitive");
  if (!tags.length) tags.push("clean");
  return tags;
}

function normalize(rawRows) {
  const rows = rawRows.map((row) => {
    const coverage = coverageFraction(row);
    const score = numberOrNull(row.coverage_adjusted_score);
    const mean = numberOrNull(row.mean_score);
    const normalized = {
      rank: null,
      model_id: row.model_id,
      provider: row.provider,
      mode: row.mode,
      mode_label: MODE_LABELS[row.mode] || row.mode || "",
      mean_score: mean,
      coverage_adjusted_score: score === null && mean !== null ? mean * coverage : score,
      human_effect: metric(row.human_effect_category_macro_f1),
      binding_rank: metric(row.binding_rank_kendall) ?? metric(row.binding_rank_spearman),
      pose: metric(row.pose_contact_f1),
      structure: metric(row.structure_score) ?? "not computed",
      coverage,
      api_error_count: Number(row.api_error_count || 0),
      invalid_json_count: Number(row.invalid_json_count || 0),
      unresolved_provider_error_count: Number(row.unresolved_provider_error_count || 0),
      unresolved_invalid_json_count: Number(row.unresolved_invalid_json_count || 0),
      fallback_prediction_count: Number(row.fallback_prediction_count || 0),
      row_status: row.row_status || "",
      scored: row.scored === true,
      is_baseline: Boolean(row.is_baseline),
      competitive: row.competitive === true && ["clean_completed", "completed_with_failures"].includes(row.row_status),
      status: ""
    };
    normalized.status = statusTags(normalized).join(", ");
    return normalized;
  });
  const ranked = rows
    .filter((row) => row.competitive && row.scored && ["clean_completed", "completed_with_failures"].includes(row.row_status))
    .sort((a, b) => (b.coverage_adjusted_score ?? -Infinity) - (a.coverage_adjusted_score ?? -Infinity));
  ranked.forEach((row, index) => {
    row.rank = index + 1;
  });
  return rows;
}

function formatScore(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(3);
  return value || "not computed";
}

function formatPercent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function filteredRows() {
  return state.rows.filter((row) => {
    if (state.filter === "all") return true;
    if (state.filter === "base") return row.mode === "base";
    if (state.filter === "tools_high_reasoning") return row.mode === "tools_high_reasoning";
    if (state.filter === "baselines") return row.is_baseline || row.competitive === false;
    return row.competitive && row.scored && ["clean_completed", "completed_with_failures"].includes(row.row_status);
  });
}

function sortRows(rows) {
  const dir = state.sortDir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const left = a[state.sortKey];
    const right = b[state.sortKey];
    if (typeof left === "number" || typeof right === "number") {
      return ((left ?? -Infinity) - (right ?? -Infinity)) * dir;
    }
    return String(left ?? "").localeCompare(String(right ?? "")) * dir;
  });
}

function renderStatus(status) {
  return `<span class="status">${status.split(", ").map((tag) => `<span class="tag ${tag}">${tag}</span>`).join("")}</span>`;
}

function render() {
  const rows = sortRows(filteredRows());
  document.querySelector("#summary").textContent = `${rows.length} rows shown. Default score: coverage-adjusted mean.`;
  const body = document.querySelector("#leaderboard-table tbody");
  body.innerHTML = rows.map((row) => `
    <tr>
      <td>${row.rank ?? "—"}</td>
      <td>${row.model_id}</td>
      <td>${row.mode_label}</td>
      <td>${formatScore(row.mean_score)}</td>
      <td>${formatScore(row.coverage_adjusted_score)}</td>
      <td>${formatScore(row.human_effect)}</td>
      <td>${formatScore(row.binding_rank)}</td>
      <td>${formatScore(row.pose)}</td>
      <td>${formatScore(row.structure)}</td>
      <td>${formatPercent(row.coverage)}</td>
      <td>${row.api_error_count}</td>
      <td>${row.invalid_json_count}</td>
      <td>${renderStatus(row.status)}</td>
    </tr>
  `).join("");
}

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    render();
  });
});

document.querySelectorAll("th[data-sort]").forEach((header) => {
  header.addEventListener("click", () => {
    const key = header.dataset.sort;
    state.sortDir = state.sortKey === key && state.sortDir === "desc" ? "asc" : "desc";
    state.sortKey = key;
    render();
  });
});

fetch("leaderboard.json")
  .then((response) => response.json())
  .then((rows) => {
    state.rows = normalize(rows);
    render();
  })
  .catch((error) => {
    document.querySelector("#summary").textContent = `Could not load leaderboard.json: ${error.message}`;
  });
""",
    )
    return public_rows


def _leaderboard_row(
    *,
    model: dict[str, Any],
    mode: str,
    results: dict[str, dict[str, Any]],
    attempted: dict[str, int],
    completed: dict[str, int],
    invalid_json_count: int,
    api_error_count: int,
    cost: float,
    release_id: str,
    run_timestamp: str,
    is_baseline: bool = False,
    competitive: bool = True,
) -> dict[str, Any]:
    metrics = _empty_metrics()
    scores = []
    completed_tracks = []
    valid_rates = []
    for track, result in results.items():
        completed_tracks.append(track)
        result_metrics = result.get("metrics", {})
        score = _track_score(track, result_metrics, int(result.get("n_cases") or attempted.get(track, 0)))
        if score is not None:
            scores.append(score)
        if track == "human_effect":
            metrics["human_effect_category_macro_f1"] = result_metrics.get("category_macro_f1")
            metrics["human_effect_evidence_accuracy"] = result_metrics.get(
                "evidence_level_ordinal_accuracy"
            )
            metrics["human_effect_overclaim_penalty"] = result_metrics.get("overclaim_penalty")
            if result_metrics.get("valid_prediction_rate") is not None:
                valid_rates.append(float(result_metrics.get("valid_prediction_rate") or 0.0))
        elif track == "binding_rank":
            metrics["binding_rank_spearman"] = result_metrics.get("spearman")
            metrics["binding_rank_kendall"] = result_metrics.get("kendall_tau")
            metrics["binding_rank_pairwise_accuracy"] = result_metrics.get(
                "pairwise_ranking_accuracy"
            )
        elif track == "pose":
            precision = result_metrics.get("interface_contact_precision")
            recall = result_metrics.get("interface_contact_recall")
            metrics["pose_contact_precision"] = precision
            metrics["pose_contact_recall"] = recall
            if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
                metrics["pose_contact_f1"] = (
                    2 * precision * recall / (precision + recall) if precision + recall else 0.0
                )
        elif track == "structure":
            metrics["structure_score"] = score
            metrics["structure_rmsd_status"] = (
                "computed" if score is not None else result_metrics.get("all_atom_rmsd", "not_computed")
            )
    total_attempted = sum(attempted.values())
    total_completed = sum(completed.values())
    row = {
        "model_id": model.get("model_id", model.get("model_name", "unknown")),
        "provider": model.get("provider", "baseline" if is_baseline else "unknown"),
        "mode": mode,
        "completed_tracks": completed_tracks,
        "coverage": {
            track: {
                "attempted": attempted.get(track, 0),
                "completed": completed.get(track, 0),
            }
            for track in sorted(set(attempted) | set(completed))
        },
        "mean_score": sum(scores) / len(scores) if scores else None,
        "total_cases_attempted": total_attempted,
        "total_cases_completed": total_completed,
        "invalid_json_count": invalid_json_count,
        "api_error_count": api_error_count,
        "total_cost_estimate_if_available": cost,
        "run_timestamp": run_timestamp,
        "benchmark_release": release_id,
        "evaluator_commit": _evaluator_commit(),
        "is_baseline": is_baseline,
        "competitive": competitive,
    }
    if valid_rates:
        row["valid_prediction_rate"] = sum(valid_rates) / len(valid_rates)
        row["failed_prediction_count"] = sum(
            int((result.get("metrics") or {}).get("failed_prediction_count") or 0)
            for result in results.values()
        )
    row.update(metrics)
    return row


def _evaluator_commit() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return "not_available"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = Path(".git") / text.split(" ", 1)[1]
            return ref.read_text(encoding="utf-8").strip()[:12] if ref.exists() else "not_available"
        return text[:12]
    except OSError:
        return "not_available"


def _load_release_id(release_dir: Path) -> str:
    manifest = release_dir / "release_manifest.json"
    if not manifest.exists():
        return release_dir.name
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return str(payload.get("release_id") or release_dir.name)


def _allowed_tracks_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "base":
        return BASE_TRACKS
    if mode == "tools_high_reasoning":
        return TOOLS_TRACKS
    raise ValueError(f"unsupported mode: {mode}")


def _parse_csv_option(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_openrouter_leaderboard(
    *,
    release_dir: Union[str, Path],
    models_config: Union[str, Path],
    output_dir: Union[str, Path],
    modes: str,
    tracks: str,
    temperature: float,
    timeout: int,
    retries: int,
    resume: bool,
    max_concurrency: int,
) -> dict[str, Any]:
    del max_concurrency
    release_path = Path(release_dir)
    output_path = Path(output_dir)
    prediction_root = output_path / "predictions"
    result_root = output_path / "results"
    prediction_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    selected_models = load_model_config(models_config)
    selected_modes = _parse_csv_option(modes)
    selected_tracks = _parse_csv_option(tracks)
    run_timestamp = _utc_now()
    release_id = _load_release_id(release_path)
    prior_counts = _prior_run_counts(output_path / "run_manifest.json") if resume else {}
    rows: list[dict[str, Any]] = []
    run_details: list[dict[str, Any]] = []

    for model in selected_models:
        model_id = model["model_id"]
        slug = safe_model_slug(model_id)
        model_prediction_dir = prediction_root / slug
        model_result_dir = result_root / slug
        model_prediction_dir.mkdir(parents=True, exist_ok=True)
        model_result_dir.mkdir(parents=True, exist_ok=True)
        for mode in selected_modes:
            allowed_tracks = set(_allowed_tracks_for_mode(mode))
            mode_results: dict[str, dict[str, Any]] = {}
            attempted: dict[str, int] = {}
            completed: dict[str, int] = {}
            invalid_json_count = 0
            api_error_count = 0
            cost = 0.0
            for track in selected_tracks:
                if track not in allowed_tracks:
                    continue
                cases = _cases_for_track(release_path, track)
                attempted[track] = len(cases)
                pred_path = model_prediction_dir / f"{mode}_{track}.jsonl"
                result_path = model_result_dir / f"{mode}_{track}.json"
                if resume and pred_path.exists():
                    predictions = read_jsonl(pred_path)
                    if len(predictions) != len(cases):
                        pred_path.unlink()
                    else:
                        result = _evaluate_track(track, cases, predictions)
                        write_text(result_path, json.dumps(result, indent=2, sort_keys=True))
                        mode_results[track] = result
                        completed[track] = len(predictions)
                        continue
                prediction_summary = _run_track_predictions(
                    model=model,
                    mode=mode,
                    track=track,
                    cases=cases,
                    output=pred_path,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                )
                predictions = prediction_summary["predictions"]
                result = _evaluate_track(track, cases, predictions)
                write_text(result_path, json.dumps(result, indent=2, sort_keys=True))
                mode_results[track] = result
                completed[track] = len(predictions)
                invalid_json_count += int(prediction_summary["invalid_json_count"])
                api_error_count += int(prediction_summary["api_error_count"])
                cost += float(prediction_summary["total_cost_estimate_if_available"])
            prior = prior_counts.get((model_id, mode))
            if prior and api_error_count == 0 and invalid_json_count == 0:
                api_error_count = int(prior.get("api_error_count") or 0)
                invalid_json_count = int(prior.get("invalid_json_count") or 0)
            row = _leaderboard_row(
                model=model,
                mode=mode,
                results=mode_results,
                attempted=attempted,
                completed=completed,
                invalid_json_count=invalid_json_count,
                api_error_count=api_error_count,
                cost=cost,
                release_id=release_id,
                run_timestamp=run_timestamp,
            )
            rows.append(row)
            run_details.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "attempted": attempted,
                    "completed": completed,
                    "invalid_json_count": invalid_json_count,
                    "api_error_count": api_error_count,
                }
            )
    rows.extend(_baseline_rows(release_path, release_id, run_timestamp))
    _write_leaderboard(output_path, rows)
    manifest = {
        "run_timestamp": run_timestamp,
        "benchmark_release": release_id,
        "peb_version": __version__,
        "models": [model["model_id"] for model in selected_models],
        "modes": selected_modes,
        "tracks": selected_tracks,
        "details": run_details,
        "leaderboard_rows": len(rows),
    }
    write_text(output_path / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    release_leaderboard = release_path / "leaderboard"
    release_leaderboard.mkdir(parents=True, exist_ok=True)
    for name in ("leaderboard.json", "leaderboard.csv", "run_manifest.json", "leaderboard_public.json"):
        source = output_path / name
        if source.exists():
            write_text(release_leaderboard / name, source.read_text(encoding="utf-8"))
    return manifest


def _prior_run_counts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    details = payload.get("details")
    if not isinstance(details, list):
        return {}
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    for item in details:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        mode = item.get("mode")
        if isinstance(model_id, str) and isinstance(mode, str):
            prior[(model_id, mode)] = item
    return prior


def _baseline_rows(release_dir: Path, release_id: str, run_timestamp: str) -> list[dict[str, Any]]:
    rows = []
    baseline_defs = [
        ("binding_rank_random", "binding_rank", "baseline", True),
        ("human_effect_non_oracle", "human_effect", "baseline", True),
        ("pose_contact_reference", "pose", "baseline", True),
        ("structure_reference", "structure", "baseline", True),
        ("human_effect_oracle_sanity_check", "human_effect", "sanity_check", False),
    ]
    for name, track, mode, competitive in baseline_defs:
        result_path = release_dir / "baselines" / "results" / f"{name}.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        cases = _cases_for_track(release_dir, track)
        n_predictions = min(int(result.get("n_predictions") or len(cases)), len(cases))
        row = _leaderboard_row(
            model={
                "model_id": name,
                "provider": "baseline",
                "family": "baseline",
            },
            mode=mode,
            results={track: result},
            attempted={track: len(cases)},
            completed={track: n_predictions},
            invalid_json_count=0,
            api_error_count=0,
            cost=0.0,
            release_id=release_id,
            run_timestamp=run_timestamp,
            is_baseline=True,
            competitive=competitive,
        )
        rows.append(row)
    return rows


def _write_leaderboard(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = normalize_leaderboard_rows(rows)
    write_text(output_dir / "leaderboard.json", json.dumps(rows, indent=2, sort_keys=True))
    fieldnames = [
        "model_id",
        "provider",
        "mode",
        "mode_label",
        "completed_tracks",
        "coverage",
        "mean_score",
        "coverage_adjusted_score",
        "human_effect_category_macro_f1",
        "human_effect_evidence_accuracy",
        "human_effect_overclaim_penalty",
        "binding_rank_spearman",
        "binding_rank_kendall",
        "binding_rank_pairwise_accuracy",
        "pose_contact_precision",
        "pose_contact_recall",
        "pose_contact_f1",
        "structure_score",
        "structure_rmsd_status",
        "total_cases_attempted",
        "total_cases_completed",
        "invalid_json_count",
        "api_error_count",
        "total_cost_estimate_if_available",
        "run_timestamp",
        "benchmark_release",
        "evaluator_commit",
        "is_baseline",
        "competitive",
        "scored",
        "fallback_prediction_count",
        "unresolved_provider_error_count",
        "unresolved_invalid_json_count",
        "row_status",
        "status",
        "leaderboard_status",
        "retry_recovered_count",
        "retry_still_failed_count",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["completed_tracks"] = json.dumps(row.get("completed_tracks", []), sort_keys=True)
            csv_row["coverage"] = json.dumps(row.get("coverage", {}), sort_keys=True)
            writer.writerow(csv_row)
    write_public_leaderboard(output_dir)


def _copy_leaderboard_to_release(leaderboard_dir: Path, release_dir: Path) -> None:
    release_leaderboard = release_dir / "leaderboard"
    release_leaderboard.mkdir(parents=True, exist_ok=True)
    for name in (
        "leaderboard.json",
        "leaderboard.csv",
        "run_manifest.json",
        "leaderboard_public.json",
        "excluded_models.json",
    ):
        source = leaderboard_dir / name
        if source.exists():
            write_text(release_leaderboard / name, source.read_text(encoding="utf-8"))
    diagnosis = leaderboard_dir / "diagnostics" / "error_diagnosis.json"
    if diagnosis.exists():
        target = release_leaderboard / "diagnostics" / "error_diagnosis.json"
        write_text(target, diagnosis.read_text(encoding="utf-8"))


def _row_failure_reason(row: dict[str, Any]) -> str:
    if _int_count(row, "unresolved_provider_error_count") or _int_count(row, "fallback_prediction_count"):
        return "provider_or_fallback_unresolved"
    if _int_count(row, "unresolved_invalid_json_count"):
        return "invalid_json_or_schema_unresolved"
    if _coverage_fraction(row) < 1.0:
        return "incomplete_outputs"
    return "not_competitive"


def _excluded_models(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = []
    for row in normalize_leaderboard_rows(rows):
        if row.get("is_baseline") or row.get("row_status") in RANKABLE_ROW_STATUSES:
            continue
        excluded.append(
            {
                "original_model_id": row.get("model_id"),
                "mode": row.get("mode"),
                "reason_excluded": _row_failure_reason(row),
                "row_status": row.get("row_status"),
                "error_counts": {
                    "api_error_count": _int_count(row, "api_error_count"),
                    "invalid_json_count": _int_count(row, "invalid_json_count"),
                    "unresolved_provider_error_count": _int_count(
                        row, "unresolved_provider_error_count"
                    ),
                    "unresolved_invalid_json_count": _int_count(
                        row, "unresolved_invalid_json_count"
                    ),
                    "fallback_prediction_count": _int_count(row, "fallback_prediction_count"),
                },
                "replacement_model_id": None,
                "replacement_reason": "no clean replacement run available in current artifacts",
            }
        )
    return excluded


def _diagnosis_payload(rows: Sequence[dict[str, Any]], retry_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = normalize_leaderboard_rows(rows)
    latest_retry_results = _latest_retry_results(retry_results)
    failed_rows = [
        row
        for row in normalized
        if not row.get("is_baseline") and row.get("row_status") not in RANKABLE_ROW_STATUSES
    ]
    failed_retry_results = [
        item for item in latest_retry_results if item.get("retry_status") == "still_failed"
    ]
    retry_counter_model = Counter(str(item.get("model_id")) for item in failed_retry_results)
    retry_counter_mode = Counter(str(item.get("mode")) for item in failed_retry_results)
    retry_counter_track = Counter(str(item.get("track")) for item in failed_retry_results)
    retry_counter_error = Counter(
        str(item.get("error_category") or "unknown") for item in failed_retry_results
    )
    root_causes = Counter(_row_failure_reason(row) for row in failed_rows)
    return {
        "generated_at": _utc_now(),
        "competitive_rows": sum(
            1 for row in normalized if _is_rankable_row(row)
        ),
        "excluded_rows": len(failed_rows),
        "retry_total": len(latest_retry_results),
        "retry_recovered": sum(1 for item in latest_retry_results if item.get("retry_status") == "recovered"),
        "retry_still_failed": len(failed_retry_results),
        "failures_by_model": dict(sorted(retry_counter_model.items())),
        "failures_by_mode": dict(sorted(retry_counter_mode.items())),
        "failures_by_track": dict(sorted(retry_counter_track.items())),
        "failures_by_error_type": dict(sorted(retry_counter_error.items())),
        "root_cause_categories": dict(sorted(root_causes.items())),
        "action_taken": {
            "competitive_rows": "retained rows with complete accounted outputs",
            "failed_rows": "excluded from competitive ranking and recorded as non-scored metadata",
            "fallback_predictions": "not counted as completed competitive outputs",
        },
    }


def finalize_strict_leaderboard(
    leaderboard_dir: Union[str, Path],
    release_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    path = Path(leaderboard_dir)
    rows = _load_leaderboard_rows(path / "leaderboard.json")
    normalized = normalize_leaderboard_rows(rows)
    _write_leaderboard(path, normalized)
    excluded = _excluded_models(normalized)
    write_text(path / "excluded_models.json", json.dumps(excluded, indent=2, sort_keys=True))
    retry_path = path / "retry" / "retry_results.jsonl"
    retry_results = read_jsonl(retry_path) if retry_path.exists() else []
    diagnosis = _diagnosis_payload(normalized, retry_results)
    write_text(
        path / "diagnostics" / "error_diagnosis.json",
        json.dumps(diagnosis, indent=2, sort_keys=True),
    )
    if release_dir is not None:
        _copy_leaderboard_to_release(path, Path(release_dir))
    competitive_rows = [
        row
        for row in normalize_leaderboard_rows(_load_leaderboard_rows(path / "leaderboard.json"))
        if _is_rankable_row(row)
    ]
    return {
        "competitive_rows": len(competitive_rows),
        "excluded_rows": len(excluded),
        "diagnostics": str(path / "diagnostics" / "error_diagnosis.json"),
        "excluded_models": str(path / "excluded_models.json"),
    }


def leaderboard_check(leaderboard_dir: Union[str, Path]) -> tuple[bool, list[str], dict[str, Any]]:
    path = Path(leaderboard_dir)
    errors: list[str] = []
    rows = _load_leaderboard_rows(path / "leaderboard.json")
    normalized = normalize_leaderboard_rows(rows)
    competitive = [
        row
        for row in normalized
        if _is_rankable_row(row)
    ]
    if not competitive:
        errors.append("no rankable competitive rows")
    for raw_row, row in zip(rows, normalized):
        model_id = str(row.get("model_id") or "")
        mode = str(row.get("mode") or "")
        if row.get("is_baseline") and "oracle" in model_id and row.get("competitive") is True:
            errors.append(f"{model_id}/{mode}: oracle baseline is competitive")
        raw_clean_claim = (
            raw_row.get("competitive") is True and raw_row.get("row_status") in RANKABLE_ROW_STATUSES
        )
        normalized_clean_claim = _is_rankable_row(row)
        competitive_claimed = raw_clean_claim or normalized_clean_claim
        if not competitive_claimed:
            continue
        prefix = f"{model_id}/{mode}"
        if _int_count(row, "api_error_count"):
            errors.append(f"{prefix}: api_error_count > 0")
        accounted_failures = row.get("row_status") == "completed_with_failures" and _row_has_accounted_failures(row)
        if _int_count(row, "invalid_json_count") and not accounted_failures:
            errors.append(f"{prefix}: invalid_json_count > 0")
        if _int_count(row, "unresolved_provider_error_count"):
            errors.append(f"{prefix}: unresolved_provider_error_count > 0")
        if _int_count(row, "unresolved_invalid_json_count") and not accounted_failures:
            errors.append(f"{prefix}: unresolved_invalid_json_count > 0")
        if _int_count(row, "fallback_prediction_count"):
            errors.append(f"{prefix}: fallback_prediction_count > 0")
        if _coverage_fraction(row) < 1.0:
            errors.append(f"{prefix}: coverage < 1.0")
        if row.get("row_status") not in RANKABLE_ROW_STATUSES:
            errors.append(f"{prefix}: row_status is not rankable")
        if row.get("scored") is not True:
            errors.append(f"{prefix}: scored is not true")
        artifact_required = raw_row.get("artifact_required") is True or model_id == "the-spice"
        if artifact_required:
            slug = safe_model_slug(model_id)
            for track in row.get("completed_tracks") or []:
                pred_path = path / "predictions" / slug / f"{mode}_{track}.jsonl"
                result_path = path / "results" / slug / f"{mode}_{track}.json"
                if not pred_path.exists():
                    errors.append(f"{prefix}/{track}: missing prediction artifact")
                if not result_path.exists():
                    errors.append(f"{prefix}/{track}: missing result artifact")
    public_path = path / "leaderboard_public.json"
    if public_path.exists():
        public_rows = json.loads(public_path.read_text(encoding="utf-8"))
        for row in public_rows:
            if row.get("rank") is None:
                continue
            if row.get("competitive") is not True:
                errors.append(f"{row.get('model_id')}/{row.get('mode')}: ranked row is not competitive")
            if row.get("row_status") not in RANKABLE_ROW_STATUSES:
                errors.append(
                    f"{row.get('model_id')}/{row.get('mode')}: ranked row is not rankable"
                )
    summary = {
        "competitive_rows": len(competitive),
        "total_rows": len(normalized),
        "errors": len(errors),
    }
    return not errors, errors, summary


def _prediction_key(record: dict[str, Any]) -> str:
    item = dict(record)
    item.pop("prediction_id", None)
    return json.dumps(item, sort_keys=True, separators=(",", ":"))


def _sanitize_error_message(message: Any) -> str:
    text = str(message or "")
    key_prefix = "sk-or-" + "v1-"
    text = re.sub(re.escape(key_prefix) + r"[A-Za-z0-9_-]+", "[redacted]", text)
    text = re.sub(r"/" + r"Users/[^\s\"']+", "[local-path]", text)
    return text[:300]


def _parse_retry_sleep(attempt: int) -> None:
    time.sleep(min(0.5 * (2**attempt), 2.0) + random.random() * 0.1)


def _load_leaderboard_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list")
    return [row for row in payload if isinstance(row, dict)]


def _model_for_id(models: Sequence[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for model in models:
        if model.get("model_id") == model_id:
            return model
    return {
        "model_id": model_id,
        "provider": _provider(model_id),
        "structured_outputs_supported": False,
    }


def build_openrouter_retry_queue(
    *,
    leaderboard_dir: Union[str, Path],
    release_dir: Union[str, Path],
    models_config: Union[str, Path],
) -> list[dict[str, Any]]:
    del models_config
    leaderboard_path = Path(leaderboard_dir)
    release_path = Path(release_dir)
    rows = _load_leaderboard_rows(leaderboard_path / "leaderboard.json")
    retry_results_path = leaderboard_path / "retry" / "retry_results.jsonl"
    recovered_prediction_keys: set[tuple[str, str, str, str]] = set()
    if retry_results_path.exists():
        for result in _latest_retry_results(read_jsonl(retry_results_path)):
            if result.get("retry_status") != "recovered":
                continue
            if not result.get("recovered_prediction_written"):
                continue
            recovered_prediction_keys.add(
                (
                    str(result.get("model_id") or ""),
                    str(result.get("mode") or ""),
                    str(result.get("track") or ""),
                    str(result.get("benchmark_id") or ""),
                )
            )
    queue: list[dict[str, Any]] = []
    for row in rows:
        if row.get("is_baseline"):
            continue
        api_errors = int(row.get("api_error_count") or 0)
        invalid_json = int(row.get("invalid_json_count") or 0)
        incomplete = _coverage_fraction(row) < 1.0
        if not api_errors and not invalid_json and not incomplete:
            continue
        model_id = str(row.get("model_id") or "")
        mode = str(row.get("mode") or "")
        if not model_id or mode not in {"base", "tools_high_reasoning"}:
            continue
        slug = safe_model_slug(model_id)
        coverage = row.get("coverage") if isinstance(row.get("coverage"), dict) else {}
        previous_type = "invalid_json" if invalid_json else "provider_error" if api_errors else "missing_prediction"
        previous_message = (
            f"row recorded api_error_count={api_errors}, invalid_json_count={invalid_json}; "
            "retry only fallback or missing predictions"
        )
        for track in coverage:
            if track not in TRACKS:
                continue
            cases = _cases_for_track(release_path, track)
            case_by_id = {case["benchmark_id"]: case for case in cases}
            pred_path = leaderboard_path / "predictions" / slug / f"{mode}_{track}.jsonl"
            predictions = read_jsonl(pred_path) if pred_path.exists() else []
            prediction_by_id = {record.get("benchmark_id"): record for record in predictions}
            fallback_by_id = {
                record["benchmark_id"]: record
                for record in _fallback_predictions(track, cases, model_id)
            }
            fallback_keys = {
                benchmark_id: _prediction_key(record)
                for benchmark_id, record in fallback_by_id.items()
            }
            for benchmark_id, case in case_by_id.items():
                prediction = prediction_by_id.get(benchmark_id)
                missing = prediction is None
                fallback_match = (
                    prediction is not None
                    and fallback_keys.get(benchmark_id) == _prediction_key(prediction)
                )
                retry_key = (model_id, mode, track, benchmark_id)
                if not missing and fallback_match and retry_key in recovered_prediction_keys:
                    continue
                if not missing and not fallback_match:
                    continue
                queue.append(
                    {
                        "model_id": model_id,
                        "safe_model_slug": slug,
                        "mode": mode,
                        "track": track,
                        "benchmark_id": benchmark_id,
                        "previous_error_type": "missing_prediction" if missing else previous_type,
                        "previous_error_message": _sanitize_error_message(previous_message),
                        "previous_attempt_count": api_errors + invalid_json,
                        "retry_status": "pending",
                        "input_checksum": sha256_text(
                            json.dumps(
                                sanitize_case_for_leaderboard(case, mode),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                    }
                )
    return queue


def _write_retry_plan(
    *,
    leaderboard_dir: Path,
    release_dir: Path,
    models_config: Path,
    queue: list[dict[str, Any]],
    timeout: int,
    retries: int,
    max_concurrency: int,
) -> dict[str, Any]:
    retry_dir = leaderboard_dir / "retry"
    retry_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(retry_dir / "retry_queue.jsonl", queue)
    original_rows = _load_leaderboard_rows(leaderboard_dir / "leaderboard.json")
    manifest = {
        "created_at": _utc_now(),
        "leaderboard_dir": str(leaderboard_dir),
        "release_dir": str(release_dir),
        "models_config": str(models_config),
        "timeout": timeout,
        "retries": retries,
        "max_concurrency": max_concurrency,
        "original_api_error_count": sum(int(row.get("api_error_count") or 0) for row in original_rows),
        "original_invalid_json_count": sum(int(row.get("invalid_json_count") or 0) for row in original_rows),
        "retry_queue_size": len(queue),
    }
    write_text(retry_dir / "retry_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _retry_prediction_once(
    *,
    model: dict[str, Any],
    mode: str,
    track: str,
    case: dict[str, Any],
    timeout: int,
    use_response_format: bool,
) -> dict[str, Any]:
    model_id = str(model["model_id"])
    sanitized = [sanitize_case_for_leaderboard(case, mode)]
    _assert_no_forbidden_inputs(sanitized)
    payload = _chat_completion(
        model_id,
        _prompt(track, mode, sanitized),
        temperature=0.0,
        timeout=timeout,
        use_response_format=use_response_format,
    )
    parsed = _prediction_records_from_response(
        _extract_json_object(_extract_message(payload)),
        track=track,
        model_id=model_id,
        expected_ids={str(case["benchmark_id"])},
        cases_by_id={str(case["benchmark_id"]): case},
    )
    if len(parsed) != 1:
        raise ValueError("retry returned the wrong number of predictions")
    return parsed[0]


def _retry_prediction_batch(
    *,
    model: dict[str, Any],
    mode: str,
    track: str,
    cases: list[dict[str, Any]],
    timeout: int,
    use_response_format: bool,
) -> list[dict[str, Any]]:
    model_id = str(model["model_id"])
    sanitized = [sanitize_case_for_leaderboard(case, mode) for case in cases]
    _assert_no_forbidden_inputs(sanitized)
    try:
        payload = _chat_completion(
            model_id,
            _compact_retry_batch_prompt(track, mode, cases),
            temperature=0.0,
            timeout=timeout,
            use_response_format=False,
            max_tokens=8192,
            low_reasoning=True,
        )
        return _compact_batch_payload_to_predictions(
            _extract_json_object(_extract_message(payload)),
            track=track,
            model_id=model_id,
            cases=cases,
        )
    except OpenRouterInsufficientCredits:
        raise
    except (OpenRouterError, json.JSONDecodeError, ValueError):
        pass

    sanitized = [sanitize_case_for_leaderboard(case, mode) for case in cases]
    _assert_no_forbidden_inputs(sanitized)
    expected_ids = {str(case["benchmark_id"]) for case in sanitized}
    payload = _chat_completion(
        model_id,
        _prompt(track, mode, sanitized),
        temperature=0.0,
        timeout=timeout,
        use_response_format=use_response_format,
    )
    parsed = _prediction_records_from_response(
        _extract_json_object(_extract_message(payload)),
        track=track,
        model_id=model_id,
        expected_ids=expected_ids,
        cases_by_id={str(case["benchmark_id"]): case for case in cases},
    )
    parsed_by_id = {record["benchmark_id"]: record for record in parsed}
    missing = expected_ids - set(parsed_by_id)
    if missing:
        raise ValueError(f"retry batch missed {len(missing)} predictions")
    return [parsed_by_id[str(case["benchmark_id"])] for case in sanitized]


def _retry_chunk_size(model_id: str, track: str) -> int:
    if model_id.startswith("google/"):
        return 1
    if model_id.startswith("qwen/") and track == "human_effect":
        return 10
    if track == "structure":
        return RETRY_LARGE_CHUNK_SIZE
    return RETRY_CHUNK_SIZE


def _retry_single_with_attempts(
    *,
    model: dict[str, Any],
    mode: str,
    track: str,
    case: dict[str, Any],
    timeout: int,
    retries: int,
    use_response_format: bool,
) -> tuple[Optional[dict[str, Any]], str, str, int]:
    error_category = "provider_error_unresolved"
    error_message = ""
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            payload = _chat_completion(
                str(model["model_id"]),
                _compact_retry_prompt(track, mode, case),
                temperature=0.0,
                timeout=timeout,
                use_response_format=False,
                max_tokens=256,
                low_reasoning=True,
            )
            compact = _extract_json_object(_extract_message(payload))
            return (
                _compact_payload_to_prediction(
                    compact,
                    track=track,
                    model_id=str(model["model_id"]),
                    case=case,
                ),
                "",
                "",
                attempt + 1,
            )
        except OpenRouterInsufficientCredits:
            raise
        except OpenRouterError as exc:
            error_category = "provider_error_unresolved"
            error_message = _sanitize_error_message(exc)
            if str(model.get("model_id") or "").startswith("google/"):
                if attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 30) + random.random())
                    continue
                return None, error_category, error_message, attempt + 1
            break
        except (json.JSONDecodeError, ValueError) as exc:
            error_category = "invalid_json_unresolved"
            error_message = _sanitize_error_message(exc)
            if attempt + 1 < attempts:
                _parse_retry_sleep(attempt)
                continue

    single_response_format = use_response_format
    for attempt in range(attempts):
        try:
            return (
                _retry_prediction_once(
                    model=model,
                    mode=mode,
                    track=track,
                    case=case,
                    timeout=timeout,
                    use_response_format=single_response_format,
                ),
                "",
                "",
                attempt + 1,
            )
        except OpenRouterInsufficientCredits:
            raise
        except OpenRouterError as exc:
            error_category = "provider_error_unresolved"
            error_message = _sanitize_error_message(exc)
            if single_response_format and "response_format" in str(exc).lower():
                single_response_format = False
        except (json.JSONDecodeError, ValueError) as exc:
            error_category = "invalid_json_unresolved"
            error_message = _sanitize_error_message(exc)
            if attempt + 1 < attempts:
                _parse_retry_sleep(attempt)
                continue
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 30) + random.random())
    try:
        payload = _chat_completion(
            str(model["model_id"]),
            _compact_retry_prompt(track, mode, case),
            temperature=0.0,
            timeout=timeout,
            use_response_format=False,
            max_tokens=256,
            low_reasoning=True,
        )
        compact = _extract_json_object(_extract_message(payload))
        return (
            _compact_payload_to_prediction(
                compact,
                track=track,
                model_id=str(model["model_id"]),
                case=case,
                ),
                "",
                "",
                attempts + 1,
            )
    except OpenRouterInsufficientCredits:
        raise
    except OpenRouterError as exc:
        return None, "provider_error_unresolved", _sanitize_error_message(exc), attempts + 1
    except (json.JSONDecodeError, ValueError) as exc:
        return None, "invalid_json_unresolved", _sanitize_error_message(exc), attempts + 1
    return None, error_category, error_message, attempts


def _replace_prediction_record(path: Path, replacement: dict[str, Any]) -> bool:
    records = read_jsonl(path) if path.exists() else []
    target_id = replacement["benchmark_id"]
    replaced = False
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("benchmark_id") == target_id:
            output.append(replacement)
            replaced = True
        else:
            output.append(record)
    if not replaced:
        output.append(replacement)
    write_jsonl(path, output)
    return replaced


def _retry_counts(results: Sequence[dict[str, Any]]) -> tuple[int, int]:
    api_errors = 0
    invalid_json = 0
    for result in _latest_retry_results(results):
        if result.get("retry_status") != "still_failed":
            continue
        category = str(result.get("error_category") or "")
        if "invalid_json" in category:
            invalid_json += 1
        else:
            api_errors += 1
    return api_errors, invalid_json


def _current_prediction_is_resolved(
    *,
    leaderboard_path: Path,
    release_path: Path,
    result: dict[str, Any],
) -> bool:
    model_id = str(result.get("model_id") or "")
    mode = str(result.get("mode") or "")
    track = str(result.get("track") or "")
    benchmark_id = str(result.get("benchmark_id") or "")
    if not model_id or not mode or track not in TRACKS or not benchmark_id:
        return False
    cases = _cases_for_track(release_path, track)
    pred_path = leaderboard_path / "predictions" / safe_model_slug(model_id) / f"{mode}_{track}.jsonl"
    if not pred_path.exists():
        return False
    predictions = read_jsonl(pred_path)
    prediction = next(
        (record for record in predictions if record.get("benchmark_id") == benchmark_id),
        None,
    )
    if prediction is None:
        return False
    try:
        validate_prediction_record(prediction)
    except (TypeError, ValueError, ValidationError):
        return False
    fallback = next(
        (
            record
            for record in _fallback_predictions(track, cases, model_id)
            if record.get("benchmark_id") == benchmark_id
        ),
        None,
    )
    return fallback is None or _prediction_key(prediction) != _prediction_key(fallback)


def _reconcile_retry_results_with_predictions(
    *,
    results: Sequence[dict[str, Any]],
    leaderboard_path: Path,
    release_path: Path,
) -> list[dict[str, Any]]:
    reconciled = list(results)
    for result in _latest_retry_results(results):
        if result.get("retry_status") != "still_failed":
            continue
        if not _current_prediction_is_resolved(
            leaderboard_path=leaderboard_path,
            release_path=release_path,
            result=result,
        ):
            continue
        reconciled.append(
            {
                **result,
                "retry_status": "recovered",
                "error_category": None,
                "error_message": None,
                "retry_attempt_count": 0,
                "recovered_prediction_written": True,
                "recovery_source": "current_valid_prediction",
            }
        )
    return reconciled


def _retry_item_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("model_id") or ""),
        str(record.get("mode") or ""),
        str(record.get("track") or ""),
        str(record.get("benchmark_id") or ""),
    )


def _latest_retry_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        latest[_retry_item_key(result)] = result
    return list(latest.values())


def _counts_by_model_mode(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row.get("model_id")), str(row.get("mode")))
        grouped[key] = {
            "api_error_count": int(row.get("api_error_count") or 0),
            "invalid_json_count": int(row.get("invalid_json_count") or 0),
        }
    return grouped


def _retry_summary_by_model_mode(results: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for result in _latest_retry_results(results):
        key = (str(result.get("model_id")), str(result.get("mode")))
        summary = grouped.setdefault(
            key,
            {
                "retry_queue_size": 0,
                "retry_recovered_count": 0,
                "retry_still_failed_count": 0,
                "post_retry_api_error_count": 0,
                "post_retry_invalid_json_count": 0,
            },
        )
        summary["retry_queue_size"] += 1
        if result.get("retry_status") == "recovered":
            summary["retry_recovered_count"] += 1
        else:
            summary["retry_still_failed_count"] += 1
            if "invalid_json" in str(result.get("error_category") or ""):
                summary["post_retry_invalid_json_count"] += 1
            else:
                summary["post_retry_api_error_count"] += 1
    return grouped


def recompute_openrouter_leaderboard_from_predictions(
    *,
    leaderboard_dir: Union[str, Path],
    release_dir: Union[str, Path],
    models_config: Union[str, Path],
    retry_results: Sequence[dict[str, Any]] = (),
    no_live_retry: bool = False,
) -> dict[str, Any]:
    leaderboard_path = Path(leaderboard_dir)
    release_path = Path(release_dir)
    manifest_path = leaderboard_path / "run_manifest.json"
    prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prior_rows = _load_leaderboard_rows(leaderboard_path / "leaderboard.json")
    original_counts = _counts_by_model_mode(prior_rows)
    retry_by_key = _retry_summary_by_model_mode(retry_results)
    selected_models = load_model_config(models_config)
    selected_modes = prior_manifest.get("modes") or ["base", "tools_high_reasoning"]
    selected_tracks = prior_manifest.get("tracks") or list(TRACKS)
    run_timestamp = _utc_now()
    release_id = _load_release_id(release_path)
    rows: list[dict[str, Any]] = []
    run_details: list[dict[str, Any]] = []

    for model in selected_models:
        model_id = model["model_id"]
        slug = safe_model_slug(model_id)
        for mode in selected_modes:
            allowed_tracks = set(_allowed_tracks_for_mode(mode))
            attempted: dict[str, int] = {}
            completed: dict[str, int] = {}
            mode_results: dict[str, dict[str, Any]] = {}
            for track in selected_tracks:
                if track not in allowed_tracks:
                    continue
                cases = _cases_for_track(release_path, track)
                attempted[track] = len(cases)
                pred_path = leaderboard_path / "predictions" / slug / f"{mode}_{track}.jsonl"
                if not pred_path.exists():
                    completed[track] = 0
                    continue
                predictions = read_jsonl(pred_path)
                completed[track] = len(predictions)
                result = _evaluate_track(track, cases, predictions)
                result_path = leaderboard_path / "results" / slug / f"{mode}_{track}.json"
                write_text(result_path, json.dumps(result, indent=2, sort_keys=True))
                mode_results[track] = result
            key = (model_id, mode)
            original = original_counts.get(key, {"api_error_count": 0, "invalid_json_count": 0})
            retry_summary = retry_by_key.get(
                key,
                {
                    "retry_queue_size": 0,
                    "retry_recovered_count": 0,
                    "retry_still_failed_count": 0,
                    "post_retry_api_error_count": 0,
                    "post_retry_invalid_json_count": 0,
                },
            )
            api_error_count = (
                original["api_error_count"]
                if no_live_retry
                else retry_summary["post_retry_api_error_count"]
            )
            invalid_json_count = (
                original["invalid_json_count"]
                if no_live_retry
                else retry_summary["post_retry_invalid_json_count"]
            )
            row = _leaderboard_row(
                model=model,
                mode=mode,
                results=mode_results,
                attempted=attempted,
                completed=completed,
                invalid_json_count=invalid_json_count,
                api_error_count=api_error_count,
                cost=0.0,
                release_id=release_id,
                run_timestamp=run_timestamp,
            )
            row.update(
                {
                    "original_api_error_count": original["api_error_count"],
                    "original_invalid_json_count": original["invalid_json_count"],
                    "retry_queue_size": retry_summary["retry_queue_size"],
                    "retry_recovered_count": retry_summary["retry_recovered_count"],
                    "retry_still_failed_count": retry_summary["retry_still_failed_count"],
                    "post_retry_api_error_count": api_error_count,
                    "post_retry_invalid_json_count": invalid_json_count,
                    "retry_run_timestamp": run_timestamp if retry_summary["retry_queue_size"] else None,
                }
            )
            if retry_summary["retry_still_failed_count"]:
                row["leaderboard_status"] = "completed_with_unresolved_errors"
            elif retry_summary["retry_recovered_count"]:
                row["leaderboard_status"] = "recovered_after_retry"
            rows.append(row)
            run_details.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "attempted": attempted,
                    "completed": completed,
                    "invalid_json_count": invalid_json_count,
                    "api_error_count": api_error_count,
                    "original_api_error_count": original["api_error_count"],
                    "original_invalid_json_count": original["invalid_json_count"],
                    "retry_queue_size": retry_summary["retry_queue_size"],
                    "retry_recovered_count": retry_summary["retry_recovered_count"],
                    "retry_still_failed_count": retry_summary["retry_still_failed_count"],
                }
            )
    rows.extend(_baseline_rows(release_path, release_id, run_timestamp))
    _write_leaderboard(leaderboard_path, rows)
    normalized_rows = _load_leaderboard_rows(leaderboard_path / "leaderboard.json")
    manifest = {
        "run_timestamp": run_timestamp,
        "benchmark_release": release_id,
        "peb_version": __version__,
        "models": [model["model_id"] for model in selected_models],
        "modes": selected_modes,
        "tracks": selected_tracks,
        "details": run_details,
        "leaderboard_rows": len(normalized_rows),
        "original_api_error_count": sum(int(row.get("original_api_error_count") or 0) for row in normalized_rows),
        "original_invalid_json_count": sum(
            int(row.get("original_invalid_json_count") or 0) for row in normalized_rows
        ),
        "retry_queue_size": sum(int(row.get("retry_queue_size") or 0) for row in normalized_rows),
        "retry_recovered_count": sum(int(row.get("retry_recovered_count") or 0) for row in normalized_rows),
        "retry_still_failed_count": sum(int(row.get("retry_still_failed_count") or 0) for row in normalized_rows),
        "post_retry_api_error_count": sum(int(row.get("api_error_count") or 0) for row in normalized_rows),
        "post_retry_invalid_json_count": sum(int(row.get("invalid_json_count") or 0) for row in normalized_rows),
        "retry_run_timestamp": run_timestamp,
    }
    write_text(leaderboard_path / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_static_leaderboard_page(leaderboard_path)
    _copy_leaderboard_to_release(leaderboard_path, release_path)
    return manifest


def openrouter_retry_failures(
    *,
    leaderboard_dir: Union[str, Path],
    release_dir: Union[str, Path],
    models_config: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int,
    retries: int,
    max_concurrency: int,
    resume: bool,
) -> dict[str, Any]:
    del max_concurrency
    leaderboard_path = Path(leaderboard_dir)
    release_path = Path(release_dir)
    models_path = Path(models_config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    retry_dir = output_path / "retry"
    retry_dir.mkdir(parents=True, exist_ok=True)
    queue = build_openrouter_retry_queue(
        leaderboard_dir=leaderboard_path,
        release_dir=release_path,
        models_config=models_path,
    )
    retry_results_path = retry_dir / "retry_results.jsonl"
    existing_results = read_jsonl(retry_results_path) if resume and retry_results_path.exists() else []
    plan = _write_retry_plan(
        leaderboard_dir=output_path,
        release_dir=release_path,
        models_config=models_path,
        queue=queue,
        timeout=timeout,
        retries=retries,
        max_concurrency=1,
    )
    models = load_model_config(models_path)
    results: list[dict[str, Any]] = list(existing_results)
    if not os.environ.get(OPENROUTER_ENV_NAME):
        write_jsonl(retry_results_path, results)
        write_text(
            retry_dir / "retry_summary.json",
            json.dumps(
                {
                    **plan,
                    "retry_recovered_count": 0,
                    "retry_still_failed_count": 0,
                    "post_retry_api_error_count": None,
                    "post_retry_invalid_json_count": None,
                    "live_retry_attempted": False,
                    "error": f"{OPENROUTER_ENV_NAME} is not set",
                },
                indent=2,
                sort_keys=True,
            ),
        )
        raise OpenRouterError(
            f"{OPENROUTER_ENV_NAME} is not set; retry queue was written but no live retry was attempted"
        )

    if not queue:
        results = _reconcile_retry_results_with_predictions(
            results=results,
            leaderboard_path=output_path,
            release_path=release_path,
        )
        write_jsonl(retry_results_path, results)
        post_api_errors, post_invalid_json = _retry_counts(results)
        manifest = recompute_openrouter_leaderboard_from_predictions(
            leaderboard_dir=output_path,
            release_dir=release_path,
            models_config=models_path,
            retry_results=results,
            no_live_retry=False,
        )
        latest_results = _latest_retry_results(results)
        summary = {
            **plan,
            "retry_recovered_count": sum(
                1 for result in latest_results if result.get("retry_status") == "recovered"
            ),
            "retry_still_failed_count": sum(
                1 for result in latest_results if result.get("retry_status") == "still_failed"
            ),
            "post_retry_api_error_count": post_api_errors,
            "post_retry_invalid_json_count": post_invalid_json,
            "retry_run_timestamp": manifest["retry_run_timestamp"],
            "live_retry_attempted": False,
        }
        write_text(retry_dir / "retry_summary.json", json.dumps(summary, indent=2, sort_keys=True))
        return summary

    results = _reconcile_retry_results_with_predictions(
        results=results,
        leaderboard_path=output_path,
        release_path=release_path,
    )
    write_jsonl(retry_results_path, results)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in queue:
        grouped.setdefault((item["model_id"], item["mode"], item["track"]), []).append(item)

    case_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for (model_id, mode, track), group in grouped.items():
        model = _model_for_id(models, model_id)
        slug = safe_model_slug(model_id)
        if track not in case_cache:
            cases = _cases_for_track(release_path, track)
            case_cache[track] = {case["benchmark_id"]: case for case in cases}
        case_by_id = case_cache[track]
        use_response_format = False
        original_path = output_path / "predictions" / slug / f"{mode}_{track}.jsonl"
        original_archive = retry_dir / "original_failures" / slug / f"{mode}_{track}.jsonl"
        if original_path.exists() and not original_archive.exists():
            original_archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_path, original_archive)
        retry_pred_path = retry_dir / "predictions" / slug / f"{mode}_{track}.jsonl"
        retry_records = read_jsonl(retry_pred_path) if retry_pred_path.exists() else []

        retry_chunk_size = _retry_chunk_size(model_id, track)
        chunks = list(_chunked(group, retry_chunk_size))
        for chunk in chunks:
            chunk_cases = [case_by_id[item["benchmark_id"]] for item in chunk]
            if len(chunk) == 1:
                item = chunk[0]
                prediction, single_category, single_message, single_attempts = (
                    _retry_single_with_attempts(
                        model=model,
                        mode=mode,
                        track=track,
                        case=chunk_cases[0],
                        timeout=timeout,
                        retries=retries,
                        use_response_format=use_response_format,
                    )
                )
                if prediction is None:
                    results.append(
                        {
                            **item,
                            "retry_status": "still_failed",
                            "error_category": single_category,
                            "error_message": single_message,
                            "retry_attempt_count": single_attempts,
                            "recovered_prediction_written": False,
                        }
                    )
                    write_jsonl(retry_results_path, results)
                    continue
                _replace_prediction_record(original_path, prediction)
                retry_records.append(prediction)
                results.append(
                    {
                        **item,
                        "retry_status": "recovered",
                        "error_category": None,
                        "error_message": None,
                        "retry_attempt_count": single_attempts,
                        "recovered_prediction_written": True,
                    }
                )
                write_jsonl(retry_pred_path, retry_records)
                write_jsonl(retry_results_path, results)
                continue
            predictions: Optional[list[dict[str, Any]]] = None
            error_category = "provider_error_unresolved"
            error_message = ""
            attempts = retries + 1
            batch_attempts = min(attempts, RETRY_BATCH_ATTEMPTS) if len(chunk) > 1 else attempts
            for attempt in range(batch_attempts):
                try:
                    predictions = _retry_prediction_batch(
                        model=model,
                        mode=mode,
                        track=track,
                        cases=chunk_cases,
                        timeout=timeout,
                        use_response_format=use_response_format,
                    )
                    break
                except OpenRouterInsufficientCredits:
                    raise
                except OpenRouterError as exc:
                    error_category = "provider_error_unresolved"
                    error_message = _sanitize_error_message(exc)
                    if use_response_format and "response_format" in str(exc).lower():
                        use_response_format = False
                except (json.JSONDecodeError, ValueError) as exc:
                    error_category = "invalid_json_unresolved"
                    error_message = _sanitize_error_message(exc)
                    if attempt + 1 < batch_attempts:
                        _parse_retry_sleep(attempt)
                        continue
                if attempt + 1 < batch_attempts:
                    time.sleep(min(2**attempt, 30) + random.random())
            if predictions is None:
                if len(chunk) > 1:
                    for item, case in zip(chunk, chunk_cases):
                        prediction, single_category, single_message, single_attempts = (
                            _retry_single_with_attempts(
                                model=model,
                                mode=mode,
                                track=track,
                                case=case,
                                timeout=timeout,
                                retries=retries,
                                use_response_format=use_response_format,
                            )
                        )
                        if prediction is None:
                            results.append(
                                {
                                    **item,
                                    "retry_status": "still_failed",
                                    "error_category": single_category,
                                    "error_message": single_message,
                                    "retry_attempt_count": single_attempts,
                                    "recovered_prediction_written": False,
                                }
                            )
                            write_jsonl(retry_results_path, results)
                            continue
                        _replace_prediction_record(original_path, prediction)
                        retry_records.append(prediction)
                        results.append(
                            {
                                **item,
                                "retry_status": "recovered",
                                "error_category": None,
                                "error_message": None,
                                "retry_attempt_count": single_attempts,
                                "recovered_prediction_written": True,
                            }
                        )
                        write_jsonl(retry_pred_path, retry_records)
                        write_jsonl(retry_results_path, results)
                    write_jsonl(retry_pred_path, retry_records)
                    write_jsonl(retry_results_path, results)
                    continue
                results.extend(
                    {
                        **item,
                        "retry_status": "still_failed",
                        "error_category": error_category,
                        "error_message": error_message,
                        "retry_attempt_count": batch_attempts,
                        "recovered_prediction_written": False,
                    }
                    for item in chunk
                )
                write_jsonl(retry_results_path, results)
                continue

            for item, prediction in zip(chunk, predictions):
                _replace_prediction_record(original_path, prediction)
                retry_records.append(prediction)
                results.append(
                    {
                        **item,
                        "retry_status": "recovered",
                        "error_category": None,
                        "error_message": None,
                        "retry_attempt_count": batch_attempts,
                        "recovered_prediction_written": True,
                    }
                )
            write_jsonl(retry_pred_path, retry_records)
            write_jsonl(retry_results_path, results)

    write_jsonl(retry_results_path, results)
    retry_result_dir = retry_dir / "results"
    retry_result_dir.mkdir(parents=True, exist_ok=True)
    post_api_errors, post_invalid_json = _retry_counts(results)
    manifest = recompute_openrouter_leaderboard_from_predictions(
        leaderboard_dir=output_path,
        release_dir=release_path,
        models_config=models_path,
        retry_results=results,
        no_live_retry=False,
    )
    latest_results = _latest_retry_results(results)
    summary = {
        **plan,
        "retry_recovered_count": sum(
            1 for result in latest_results if result.get("retry_status") == "recovered"
        ),
        "retry_still_failed_count": sum(
            1 for result in latest_results if result.get("retry_status") == "still_failed"
        ),
        "post_retry_api_error_count": post_api_errors,
        "post_retry_invalid_json_count": post_invalid_json,
        "retry_run_timestamp": manifest["retry_run_timestamp"],
        "live_retry_attempted": True,
    }
    write_text(retry_dir / "retry_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    return summary
