"""Run PEB-format inference for the trained Tinker sampler."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from training_common import (
    DEFAULT_RELEASE_DIR,
    DEFAULT_RUN_DIR,
    SECRET_PATTERNS,
    TINKER_ENV_NAME,
    stable_hash,
    strict_response_template,
    write_json,
    write_jsonl,
)

from peb.io import read_jsonl
from peb.openrouter_leaderboard import (
    BASE_TRACKS,
    TOOLS_TRACKS,
    _assert_no_forbidden_inputs,
    _cases_for_track,
    _prediction_records_from_response,
    _prompt,
    safe_model_slug,
    sanitize_case_for_leaderboard,
)

DEFAULT_MODEL_NAME = "the-spice"
DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_CHUNK_SIZE = 1
DEFAULT_MAX_TOKENS = 1536
JSON_START_TAG = "<json>"
JSON_END_TAG = "</json>"
MAX_PREDICTED_CONTACTS = 50
MODES = ("base", "tools_high_reasoning")
CONFIDENCE_VALUES = {"low", "medium", "high"}
DEVELOPABILITY_KEYS = {
    "stability_risk",
    "solubility_risk",
    "toxicity_risk",
    "hemolysis_risk",
    "cytotoxicity_risk",
    "synthesis_complexity",
}
REPEATED_IDENTIFIER_RE = re.compile(r"(?:PEB-)?(?:STRUCT|POSE)-\d{3,}|(?:STRUCT|POSE)-\d{3,}", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z]+-\d+[A-Za-z0-9_:-]*|[A-Z]{3,}[A-Z0-9_:-]*")


class InferenceBlocked(RuntimeError):
    """Raised for an exact, non-secret inference blocker."""

    def __init__(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sanitize_error(value: Any) -> str:
    text = f"{type(value).__name__}: {value}" if isinstance(value, BaseException) else str(value or "")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = text.replace(str(Path.home()), "[home]")
    return text[:500]


def _local_adapter_files(run_dir: Path) -> list[Path]:
    upload_dir = run_dir / "hf_upload"
    files: list[Path] = []
    for root in (run_dir, upload_dir):
        if not root.exists():
            continue
        for pattern in ("*.safetensors", "*.bin"):
            files.extend(root.glob(pattern))
    return sorted(set(files))


def _manifest_candidates(run_dir: Path) -> list[dict[str, Any]]:
    manifests = [
        _read_json(run_dir / "hf_upload" / "adapter_export_manifest.json"),
        _read_json(run_dir / "adapter_export_manifest.json"),
        _read_json(run_dir / "hf_upload_manifest.json"),
        _read_json(run_dir / "checkpoints.json"),
    ]
    return [item for item in manifests if item]


def _sampler_path(run_dir: Path) -> Optional[str]:
    for manifest in _manifest_candidates(run_dir):
        for key in ("remote_sampler_path", "tinker_path", "sampler_path"):
            value = manifest.get(key)
            if isinstance(value, str) and value.startswith("tinker://"):
                return value
        latest = manifest.get("latest_checkpoint")
        if isinstance(latest, dict):
            for key in ("sampler_path", "remote_sampler_path", "tinker_path", "path"):
                value = latest.get(key)
                if isinstance(value, str) and value.startswith("tinker://"):
                    return value
        checkpoints = manifest.get("checkpoints")
        if isinstance(checkpoints, list):
            for item in reversed(checkpoints):
                if not isinstance(item, dict):
                    continue
                for key in ("sampler_path", "remote_sampler_path", "tinker_path", "path"):
                    value = item.get(key)
                    if isinstance(value, str) and value.startswith("tinker://"):
                        return value
    return None


def _allowed_tracks(mode: str) -> tuple[str, ...]:
    if mode == "base":
        return BASE_TRACKS
    if mode == "tools_high_reasoning":
        return TOOLS_TRACKS
    raise ValueError(f"unsupported mode: {mode}")


def _artifact_slugs(model_name: str) -> list[str]:
    legacy = model_name.replace("/", "_").replace("-", "_")
    strict = safe_model_slug(model_name)
    return list(dict.fromkeys([legacy, strict]))


def _chunked(records: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(records), size):
        yield records[index : index + size]


def _future_result(value: Any, timeout: int) -> Any:
    if hasattr(value, "result"):
        try:
            return value.result(timeout=timeout)
        except TypeError:
            return value.result()
    return value


def _import_tinker_module() -> Any:
    try:
        return importlib.import_module("tinker")
    except ImportError as exc:
        raise InferenceBlocked("tinker_sdk_not_installed", {"error": _sanitize_error(exc)}) from exc


def _load_tokenizer(base_model: str) -> Any:
    try:
        module = importlib.import_module("tinker.lib.public_interfaces.sampling_client")
        loader = module._load_tokenizer_from_model_info
        return loader(base_model)
    except Exception as exc:
        raise InferenceBlocked("tokenizer_load_failed", {"error": _sanitize_error(exc)}) from exc


def _create_sampler(run_dir: Path, base_model: str) -> tuple[Any, Any, str, Any]:
    if not os.environ.get(TINKER_ENV_NAME):
        raise InferenceBlocked("adapter_requires_tinker_auth_env")
    sampler_path = _sampler_path(run_dir)
    if not sampler_path:
        raise InferenceBlocked("remote_sampler_path_missing")
    module = _import_tinker_module()
    try:
        service = module.ServiceClient(user_metadata={"project": "peb", "model": DEFAULT_MODEL_NAME})
    except TypeError:
        try:
            service = module.ServiceClient()
        except Exception as exc:
            raise InferenceBlocked("remote_sampler_service_failed", {"error": _sanitize_error(exc)}) from exc
    except Exception as exc:
        raise InferenceBlocked("remote_sampler_service_failed", {"error": _sanitize_error(exc)}) from exc
    try:
        sampler = service.create_sampling_client(model_path=sampler_path)
    except TypeError:
        sampler = service.create_sampling_client(sampler_path)
    except Exception as exc:
        raise InferenceBlocked("remote_sampler_client_failed", {"error": _sanitize_error(exc)}) from exc
    tokenizer = _load_tokenizer(base_model)
    return module, sampler, sampler_path, tokenizer


def _encode_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="low",
            )
            if isinstance(rendered, str) and hasattr(tokenizer, "encode"):
                return list(
                    tokenizer.encode(
                        rendered + "<|start|>assistant<|channel|>final<|message|>"
                    )
                )
        except Exception:
            pass
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            if isinstance(encoded, Mapping) and "input_ids" in encoded:
                encoded = encoded["input_ids"]
            elif hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            elif hasattr(encoded, "ids"):
                encoded = encoded.ids
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if encoded and hasattr(encoded[0], "ids"):
                encoded = encoded[0].ids
            return list(encoded)
        except Exception:
            pass
    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    text += "\nassistant:"
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text))
    raise TypeError("tokenizer has neither apply_chat_template nor encode")


def _decode_tokens(tokenizer: Any, tokens: Any) -> str:
    token_list = list(tokens)
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(token_list, skip_special_tokens=True))
        except TypeError:
            return str(tokenizer.decode(token_list))
    return "".join(chr(int(token)) for token in token_list)


def _json_output_contract(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    instruction = (
        "Return exactly one JSON object wrapped as <json>{...}</json>. "
        "Do not write prose, markdown, analysis, or any text after the closing brace. "
        "Stop immediately after the JSON object."
    )
    if not messages:
        return [{"role": "system", "content": instruction}]
    wrapped = [dict(message) for message in messages]
    system_index = next((index for index, item in enumerate(wrapped) if item.get("role") == "system"), None)
    if system_index is None:
        wrapped.insert(0, {"role": "system", "content": instruction})
    else:
        wrapped[system_index]["content"] = f"{wrapped[system_index].get('content', '')}\n{instruction}"
    for item in reversed(wrapped):
        if item.get("role") == "user":
            item["content"] = f"{item.get('content', '')}\nOutput format: {JSON_START_TAG}{{...}}{JSON_END_TAG}"
            break
    return wrapped


def _pose_report_contract(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    skeleton = {
        "developability": {},
        "functional_assay_estimate": {},
        "human_effect_estimate": {},
        "known_source_backed_facts": [],
        "missing_evidence": [],
        "overall_confidence": "low|medium|high",
        "pose_contact_assessment": {
            "status": "...",
            "confidence": "low|medium|high",
            "interface_residues": [],
            "predicted_contacts": [
                {
                    "peptide_residue": "...",
                    "target_residue": "...",
                    "distance_angstrom": None,
                }
            ],
            "notes": "",
        },
        "recommended_next_assays": [],
        "structure_source_reference": {},
        "target_binding": {},
        "unsupported_claims": [],
    }
    instruction = (
        "For pose_contact_prediction, use the full report schema below. "
        "If contacts are unknown, return an empty predicted_contacts list. "
        f"Include at most {MAX_PREDICTED_CONTACTS} predicted_contacts. "
        "Each contact may contain only peptide_residue, target_residue, and "
        "distance_angstrom. Schema: "
        + json.dumps(skeleton, sort_keys=True)
    )
    wrapped = [dict(message) for message in messages]
    wrapped.append({"role": "user", "content": instruction})
    return wrapped


def _stability_report_contract(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    skeleton = {
        "developability": {
            "stability_risk": "unknown|low|medium|high",
            "solubility_risk": "unknown|low|medium|high",
            "toxicity_risk": "unknown|low|medium|high",
            "hemolysis_risk": "unknown|low|medium|high",
            "cytotoxicity_risk": "unknown|low|medium|high",
            "synthesis_complexity": "unknown|low|medium|high",
        },
        "functional_assay_estimate": {},
        "human_effect_estimate": {},
        "known_source_backed_facts": [],
        "missing_evidence": [],
        "overall_confidence": "low|medium|high",
        "pose_contact_assessment": {},
        "recommended_next_assays": [],
        "structure_source_reference": {},
        "target_binding": {},
        "unsupported_claims": [],
    }
    instruction = (
        "For stability_solubility_toxicity, return the full report schema below. "
        "Put stability, solubility, toxicity, hemolysis, cytotoxicity, and synthesis "
        "risk estimates only under developability. Keep unrelated sections as empty "
        "or default containers unless the prompt supplies direct source evidence. "
        "Do not add long source fact text. Schema: "
        + json.dumps(skeleton, sort_keys=True)
    )
    wrapped = [dict(message) for message in messages]
    wrapped.append({"role": "user", "content": instruction})
    return wrapped


def _known_report_contract(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    skeleton = {
        "known_source_backed_facts": [],
        "structure_source_reference": {},
        "pose_contact_assessment": {},
        "target_binding": {},
        "functional_assay_estimate": {},
        "developability": {},
        "human_effect_estimate": {},
        "unsupported_claims": [],
        "missing_evidence": [],
        "recommended_next_assays": [],
        "overall_confidence": "low|medium|high",
    }
    instruction = (
        "For known_peptide_report, return exactly one compact JSON object. "
        "Start with { and end after the final }. Do not output <noinput>. "
        "Do not output source identifiers outside JSON. Do not repeat STRUCT or POSE identifiers. "
        "Do not write prose before or after JSON. Use this compact schema: "
        + json.dumps(skeleton, sort_keys=True)
    )
    wrapped = [dict(message) for message in messages]
    wrapped.append({"role": "user", "content": instruction})
    return wrapped


def _known_report_retry_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    benchmark_id = str(row.get("benchmark_id") or "")
    source_id = str(row.get("source_id") or "")
    source_database = str(row.get("source_database") or "")
    skeleton = strict_response_template()
    compact = {
        "benchmark_id": benchmark_id,
        "source_id": source_id,
        "source_database": source_database,
        "schema": skeleton,
    }
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. Start with { and stop after }. "
                "No XML-like tags, no prose, no identifiers outside JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one compact known peptide report for this source-backed case. "
                "Use only the identifiers supplied here. Do not invent assay evidence, "
                "human-effect claims, binding ranks, or contacts. If a field is unknown, "
                "use an empty/default container. "
                + json.dumps(compact, sort_keys=True, separators=(",", ":"))
            ),
        },
    ]


def _human_effect_final_fallback_messages(case: dict[str, Any], mode: str) -> list[dict[str, str]]:
    sanitized = sanitize_case_for_leaderboard(case, mode)
    payload = {
        key: sanitized.get(key)
        for key in (
            "benchmark_id",
            "source_database",
            "source_id",
            "peptide_name",
            "drug_name",
            "target_name",
            "claim_text",
            "claimed_effect",
        )
        if sanitized.get(key) is not None
    }
    schema = {
        "human_effect_estimate": {
            "category": (
                "no_known_human_effect_evidence|metabolic_weight_glucose|endocrine_hormonal|"
                "cardiovascular|reproductive|immune_inflammatory|antimicrobial_antiinfective|"
                "oncology|neurologic_neuroactive|other"
            ),
            "claim_status": "supported|plausible_but_unproven|insufficient_information|unsupported",
            "confidence": "low|medium|high",
            "evidence_direction": "positive|negative|mixed|not_reported|not_applicable",
            "evidence_level": (
                "approved_human_indication|human_clinical_evidence|preclinical_only|"
                "source_reference_only|unsupported_contradicted_or_unsafe_claim"
            ),
        },
        "known_source_backed_facts": [],
        "missing_evidence": [],
        "unsupported_claims": [],
    }
    return [
        {
            "role": "system",
            "content": "Return only one minified JSON object. No prose, markdown, tags, or repeated IDs.",
        },
        {
            "role": "user",
            "content": (
                "Predict only the minimum human-effect fields for this benchmark input. "
                "Use only supplied fields. Do not invent supported claims when evidence is absent. "
                "Required schema: "
                + json.dumps(schema, sort_keys=True, separators=(",", ":"))
                + " Input: "
                + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            ),
        },
    ]


def _apply_task_output_contract(
    messages: list[dict[str, str]], *, track: Optional[str] = None, task_type: Optional[str] = None
) -> list[dict[str, str]]:
    if track == "pose" or task_type == "pose_contact_prediction":
        return _pose_report_contract(messages)
    if task_type == "stability_solubility_toxicity":
        return _stability_report_contract(messages)
    if task_type == "known_peptide_report":
        return _known_report_contract(messages)
    return messages


def _json_tag_candidates(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if JSON_START_TAG in text:
        after_start = text.split(JSON_START_TAG, 1)[1]
        before_end = after_start.split(JSON_END_TAG, 1)[0]
        candidates.append(("tagged_json", before_end))
    candidates.append(("raw", text))
    return candidates


def _balanced_json_texts(text: str) -> Iterable[str]:
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _balanced_json_objects(text: str) -> Iterable[dict[str, Any]]:
    for object_text in _balanced_json_texts(text):
        try:
            payload = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _has_balanced_json_object(text: str) -> bool:
    return next(_balanced_json_texts(text), None) is not None


def _repeated_identifier_reason(text: str) -> Optional[str]:
    identifiers = REPEATED_IDENTIFIER_RE.findall(text)
    if len(identifiers) >= 12:
        unique_ratio = len(set(identifiers)) / len(identifiers)
        if unique_ratio <= 0.6:
            return "repeated_struct_pose_identifier_loop"
    tokens = TOKEN_RE.findall(text)
    if len(tokens) >= 40:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio <= 0.35:
            return "mostly_repeated_tokens"
    for size in range(12, min(120, max(13, len(text) // 2))):
        fragment = text[:size]
        if fragment and text.count(fragment) >= 4:
            return "repeated_substring_loop"
    return None


def _degenerate_output_reason(text: str) -> Optional[str]:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    if not stripped:
        return "empty_output"
    if lowered.startswith("<noinput"):
        return "noinput_output"
    if lowered.startswith("<no-"):
        repeated = _repeated_identifier_reason(stripped)
        return repeated or "no_prefixed_non_json_output"
    repeated = _repeated_identifier_reason(stripped)
    if repeated and "{" not in stripped:
        return repeated
    if "{" not in stripped:
        return "no_json_object_marker"
    if not _has_balanced_json_object(stripped):
        return "no_balanced_json_object"
    if len(stripped) > 1000 and "{" not in stripped:
        return "large_output_without_json"
    return None


def _repair_json_payload(candidate_text: str) -> Optional[dict[str, Any]]:
    start = candidate_text.find("{")
    end = candidate_text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = candidate_text[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_payload_with_metadata(
    text: str, *, required_key: Optional[str] = None
) -> dict[str, Any]:
    tails = []
    for marker in ("assistantfinal", "<|channel|>final", "final"):
        if marker in text:
            tails.append(text.rsplit(marker, 1)[-1])
    tails.append(text)
    for candidate_text in tails:
        stripped = candidate_text.strip()
        if not stripped:
            continue
        for source, source_text in _json_tag_candidates(stripped):
            source_text = source_text.strip()
            try:
                payload = json.loads(source_text)
                if isinstance(payload, dict) and (required_key is None or required_key in payload):
                    return {
                        "payload": payload,
                        "method": f"{source}:full_json",
                        "json_text": source_text,
                    }
            except json.JSONDecodeError:
                pass
            for object_text in _balanced_json_texts(source_text):
                try:
                    payload = json.loads(object_text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and (required_key is None or required_key in payload):
                    return {
                        "payload": payload,
                        "method": f"{source}:first_balanced_object",
                        "json_text": object_text,
                    }
            repaired = _repair_json_payload(source_text)
            if repaired is not None and (required_key is None or required_key in repaired):
                return {
                    "payload": repaired,
                    "method": f"{source}:local_repair",
                    "json_text": json.dumps(repaired, sort_keys=True, separators=(",", ":")),
                }
    raise json.JSONDecodeError("no matching JSON object found", text, 0)


def _extract_json_payload(text: str, *, required_key: Optional[str] = None) -> dict[str, Any]:
    return _extract_json_payload_with_metadata(text, required_key=required_key)["payload"]


def _track_max_tokens(track: str, requested: int) -> int:
    limits = {
        "human_effect": 768,
        "binding_rank": 1024,
        "structure": 512,
        "developability": 768,
        "pose": 1536,
    }
    return min(requested, limits.get(track, requested))


def _sample_text(
    *,
    module: Any,
    sampler: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
    use_json_wrapper: bool = True,
) -> str:
    output_messages = _json_output_contract(messages) if use_json_wrapper else messages
    input_tokens = _encode_messages(tokenizer, output_messages)
    model_input = module.ModelInput.from_ints(input_tokens)
    params = module.SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        stop=[JSON_END_TAG, "<|return|>", "<|end|>"],
    )
    response = _future_result(
        sampler.sample(prompt=model_input, num_samples=1, sampling_params=params),
        timeout=timeout,
    )
    sequences = getattr(response, "sequences", None) or []
    if not sequences:
        raise RuntimeError("sampler returned no sequences")
    sequence = sequences[0]
    tokens = getattr(sequence, "tokens", sequence)
    return _decode_tokens(tokenizer, tokens)


def _repair_messages(
    *,
    track: str,
    mode: str,
    expected_ids: set[str],
    raw_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return valid JSON only. Do not add prose, markdown, or explanations.",
        },
        {
            "role": "user",
            "content": (
                "Repair the following model output into a JSON object with key 'predictions'. "
                f"Track: {track}. Mode: {mode}. Expected benchmark_ids: {sorted(expected_ids)}. "
                "Keep only valid prediction objects for those ids. Output to repair: "
                + raw_text[:4000]
            ),
        },
    ]


def _confidence_to_float(value: Any) -> float:
    text = str(value or "").lower()
    return {"high": 0.8, "medium": 0.5, "low": 0.2}.get(text, 0.0)


def _clean_contacts(value: Any) -> list[dict[str, Any]]:
    contacts = []
    if not isinstance(value, list):
        return contacts
    for item in value:
        if not isinstance(item, dict):
            continue
        target = item.get("target_residue")
        peptide = item.get("peptide_residue")
        if not target or not peptide:
            continue
        contact = {"target_residue": str(target), "peptide_residue": str(peptide)}
        try:
            if item.get("distance_angstrom") is not None:
                contact["distance_angstrom"] = float(item["distance_angstrom"])
        except (TypeError, ValueError):
            pass
        contacts.append(contact)
        if len(contacts) >= MAX_PREDICTED_CONTACTS:
            break
    return contacts


def _low_confidence_pose() -> dict[str, Any]:
    return {
        "status": "not_assessed",
        "confidence": "low",
        "interface_residues": [],
        "predicted_contacts": [],
        "notes": "",
    }


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _clean_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in CONFIDENCE_VALUES else "low"


def _normalise_pose_contact(value: Any) -> dict[str, Any]:
    pose = _low_confidence_pose()
    if not isinstance(value, dict):
        return pose
    pose["status"] = str(value.get("status") or "not_assessed")
    pose["confidence"] = _clean_confidence(value.get("confidence"))
    pose["interface_residues"] = _clean_string_list(value.get("interface_residues"))
    pose["predicted_contacts"] = _clean_contacts(value.get("predicted_contacts"))
    pose["notes"] = str(value.get("notes") or "")
    return pose


def _normalise_dict_section(value: Any, template: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(template)
    if not isinstance(value, dict):
        return cleaned
    for key, default in template.items():
        incoming = value.get(key)
        if isinstance(default, list):
            cleaned[key] = incoming if isinstance(incoming, list) else []
        elif isinstance(default, dict):
            cleaned[key] = incoming if isinstance(incoming, dict) else {}
        elif isinstance(default, str):
            cleaned[key] = str(incoming) if incoming is not None else default
        else:
            cleaned[key] = incoming if incoming is not None else default
    return cleaned


def _normalise_report_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    template = strict_response_template()
    normalised: dict[str, Any] = {}
    notes: list[str] = []
    if not isinstance(payload, dict):
        payload = {}
        notes.append("non_dict_payload_replaced")

    top_level_developability = {
        key: payload[key]
        for key in DEVELOPABILITY_KEYS
        if key in payload and not isinstance(payload.get("developability"), dict)
    }
    for key, expected in template.items():
        if key == "pose_contact_assessment":
            normalised[key] = _normalise_pose_contact(payload.get(key))
            if key not in payload:
                notes.append("pose_contact_assessment_defaulted")
            continue
        if key == "developability" and top_level_developability:
            normalised[key] = _normalise_dict_section(top_level_developability, expected)
            notes.append("top_level_developability_wrapped")
            continue
        if key == "overall_confidence":
            normalised[key] = _clean_confidence(payload.get(key))
            if key not in payload:
                notes.append("overall_confidence_defaulted")
            continue
        if isinstance(expected, list):
            value = payload.get(key)
            normalised[key] = value if isinstance(value, list) else []
            if key not in payload:
                notes.append(f"{key}_defaulted")
        elif isinstance(expected, dict):
            normalised[key] = _normalise_dict_section(payload.get(key), expected)
            if key not in payload:
                notes.append(f"{key}_defaulted")
        elif isinstance(expected, str):
            value = payload.get(key)
            normalised[key] = str(value) if value is not None else expected
            if key not in payload:
                notes.append(f"{key}_defaulted")
    return normalised, notes


def _engineer_schema_prediction(
    *,
    payload: dict[str, Any],
    track: str,
    model_name: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    benchmark_id = str(case["benchmark_id"])
    base = {
        "prediction_id": f"{safe_model_slug(model_name)}:{benchmark_id}",
        "benchmark_id": benchmark_id,
        "track": track,
        "model_name": model_name,
    }
    if track == "human_effect":
        estimate = payload.get("human_effect_estimate") if isinstance(payload, dict) else {}
        if not isinstance(estimate, dict):
            estimate = {}
        base.update(
            {
                "category": estimate.get("category") or "no_known_human_effect_evidence",
                "evidence_level": estimate.get("evidence_level")
                or "unsupported_contradicted_or_unsafe_claim",
                "evidence_direction": estimate.get("evidence_direction") or "not_applicable",
                "claim_status": estimate.get("claim_status") or "insufficient_information",
                "safety_status": estimate.get("safety_status") or "insufficient_safety_data",
                "abstained": bool(
                    estimate.get("abstained")
                    or estimate.get("claim_status") in {"insufficient_information", "unsupported"}
                ),
                "rationale_source_ids": [],
            }
        )
        return base
    if track == "pose":
        pose = payload.get("pose_contact_assessment") if isinstance(payload, dict) else {}
        if not isinstance(pose, dict):
            pose = {}
        residues = pose.get("interface_residues")
        base.update(
            {
                "predicted_contacts": _clean_contacts(pose.get("predicted_contacts")),
                "binding_site_residues": [
                    str(item) for item in residues if isinstance(item, str)
                ]
                if isinstance(residues, list)
                else [],
                "clash_score": 0.0,
            }
        )
        return base
    if track == "structure":
        structure = payload.get("structure_source_reference") if isinstance(payload, dict) else {}
        if not isinstance(structure, dict):
            structure = {}
        base.update(
            {
                "coordinates": [],
                "confidence": _confidence_to_float(structure.get("confidence")),
            }
        )
        return base
    if track == "binding_rank":
        binding = payload.get("target_binding") if isinstance(payload, dict) else {}
        if not isinstance(binding, dict):
            binding = {}
        raw_ranks = binding.get("relative_rank")
        scores_by_item: dict[str, dict[str, Any]] = {}
        if isinstance(raw_ranks, list):
            for index, item in enumerate(raw_ranks, start=1):
                if not isinstance(item, dict):
                    continue
                item_id = item.get("item_id") or item.get("peptide_id") or item.get("id")
                if not item_id:
                    continue
                try:
                    score = float(
                        item.get("score")
                        if item.get("score") is not None
                        else item.get("relative_score")
                        if item.get("relative_score") is not None
                        else -float(item.get("rank", index))
                    )
                except (TypeError, ValueError):
                    score = 0.0
                try:
                    rank = int(item.get("rank") or index)
                except (TypeError, ValueError):
                    rank = index
                scores_by_item[str(item_id)] = {
                    "item_id": str(item_id),
                    "score": score,
                    "rank": rank,
                }
        scores = []
        for index, item in enumerate(case.get("items") or [], start=1):
            item_id = str(item.get("item_id"))
            scores.append(scores_by_item.get(item_id, {"item_id": item_id, "score": 0.0, "rank": index}))
        base["scores"] = scores
        return base
    raise ValueError(f"unsupported track: {track}")


def _coerce_human_effect_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    category_aliases = {
        "immune_inflammatory": "immune_vaccine_antigen_presentation",
        "oncology": "anticancer_tumor_homing",
        "neurologic_neuroactive": "neuro_cns",
        "other": "no_known_human_effect_evidence",
    }
    evidence_aliases = {
        "preclinical_only": "animal_preclinical_phenotype_evidence",
        "source_reference_only": "mechanistic_pathway_or_similarity_hypothesis",
    }
    estimate = payload.get("human_effect_estimate") if isinstance(payload, dict) else {}
    if not isinstance(estimate, dict):
        estimate = {}
    template = strict_response_template()
    report = dict(template)
    report["human_effect_estimate"] = dict(template["human_effect_estimate"])

    category = str(estimate.get("category") or "no_known_human_effect_evidence")
    report["human_effect_estimate"]["category"] = category_aliases.get(category, category)
    if report["human_effect_estimate"]["category"] not in {
        "metabolic_weight_glucose",
        "endocrine_hormonal",
        "antimicrobial_antiinfective",
        "anticancer_tumor_homing",
        "immune_vaccine_antigen_presentation",
        "neuro_cns",
        "cardiovascular",
        "reproductive",
        "diagnostic_imaging",
        "toxic_adverse_effect_concern",
        "no_known_human_effect_evidence",
    }:
        report["human_effect_estimate"]["category"] = "no_known_human_effect_evidence"

    evidence_level = str(estimate.get("evidence_level") or "unsupported_contradicted_or_unsafe_claim")
    report["human_effect_estimate"]["evidence_level"] = evidence_aliases.get(evidence_level, evidence_level)
    if report["human_effect_estimate"]["evidence_level"] not in {
        "approved_human_indication",
        "human_clinical_evidence",
        "animal_preclinical_phenotype_evidence",
        "in_vitro_target_activity_evidence",
        "mechanistic_pathway_or_similarity_hypothesis",
        "unsupported_contradicted_or_unsafe_claim",
    }:
        report["human_effect_estimate"]["evidence_level"] = "unsupported_contradicted_or_unsafe_claim"

    evidence_direction = str(estimate.get("evidence_direction") or "not_applicable")
    if evidence_direction not in {
        "positive",
        "negative",
        "mixed",
        "inconclusive",
        "not_reported",
        "not_applicable",
    }:
        evidence_direction = "not_applicable"
    report["human_effect_estimate"]["evidence_direction"] = evidence_direction

    claim_status = str(estimate.get("claim_status") or "insufficient_information")
    if claim_status not in {
        "supported",
        "plausible_but_unproven",
        "unsupported",
        "contradicted",
        "unsafe_to_claim",
        "insufficient_information",
    }:
        claim_status = "insufficient_information"
    report["human_effect_estimate"]["claim_status"] = claim_status
    report["human_effect_estimate"]["confidence"] = _clean_confidence(estimate.get("confidence"))
    for key in ("known_source_backed_facts", "missing_evidence", "unsupported_claims"):
        value = payload.get(key) if isinstance(payload, dict) else None
        report[key] = value if isinstance(value, list) else []
    return report


def _predict_human_effect_final_fallback(
    *,
    module: Any,
    sampler: Any,
    tokenizer: Any,
    case: dict[str, Any],
    mode: str,
    model_name: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    attempts = max(1, min(2, retries))
    last_error: Optional[BaseException] = None
    for _attempt in range(attempts):
        raw_text = _sample_text(
            module=module,
            sampler=sampler,
            tokenizer=tokenizer,
            messages=_human_effect_final_fallback_messages(case, mode),
            timeout=timeout,
            max_tokens=384,
            use_json_wrapper=False,
        )
        degenerate_reason = _degenerate_output_reason(raw_text)
        if degenerate_reason:
            last_error = json.JSONDecodeError(
                f"degenerate_non_json_output:{degenerate_reason}", raw_text, 0
            )
            continue
        try:
            payload = _extract_json_payload(raw_text)
            report = _coerce_human_effect_report_payload(payload)
            return _engineer_schema_prediction(
                payload=report,
                track="human_effect",
                model_name=model_name,
                case=case,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError(_sanitize_error(last_error or "human_effect_final_fallback_failed"))


def _parse_predictions(
    *,
    text: str,
    track: str,
    model_name: str,
    expected_ids: set[str],
    cases_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    degenerate_reason = _degenerate_output_reason(text)
    if degenerate_reason:
        raise json.JSONDecodeError(f"degenerate_non_json_output:{degenerate_reason}", text, 0)
    try:
        payload = _extract_json_payload(text, required_key="predictions")
    except json.JSONDecodeError as exc:
        payload = _extract_json_payload(text)
        if len(cases_by_id) != 1:
            raise ValueError("engineer schema postprocessing requires one-case chunks") from exc
        case = next(iter(cases_by_id.values()))
        payload = {
            "predictions": [
                _engineer_schema_prediction(
                    payload=payload,
                    track=track,
                    model_name=model_name,
                    case=case,
                )
            ]
        }
    parsed = _prediction_records_from_response(
        payload,
        track=track,
        model_id=model_name,
        expected_ids=expected_ids,
        cases_by_id=cases_by_id,
    )
    parsed_by_id = {record["benchmark_id"]: record for record in parsed}
    missing = expected_ids - set(parsed_by_id)
    if missing:
        raise ValueError(f"missing predictions for {len(missing)} cases")
    return [parsed_by_id[benchmark_id] for benchmark_id in sorted(expected_ids)]


def _predict_chunk(
    *,
    module: Any,
    sampler: Any,
    tokenizer: Any,
    track: str,
    mode: str,
    cases: list[dict[str, Any]],
    model_name: str,
    timeout: int,
    max_tokens: int,
    retries: int,
) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any]]:
    sanitized = [sanitize_case_for_leaderboard(case, mode) for case in cases]
    _assert_no_forbidden_inputs(sanitized)
    expected_ids = {str(case["benchmark_id"]) for case in sanitized}
    cases_by_id = {str(case["benchmark_id"]): case for case in cases}
    attempts = retries + 1
    transient_api_errors = 0
    transient_invalid_json = 0
    last_error_type = "unknown"
    last_error_message = ""
    for attempt in range(attempts):
        try:
            raw_text = _sample_text(
                module=module,
                sampler=sampler,
                tokenizer=tokenizer,
                messages=_apply_task_output_contract(_prompt(track, mode, sanitized), track=track),
                timeout=timeout,
                max_tokens=max_tokens,
            )
            try:
                predictions = _parse_predictions(
                    text=raw_text,
                    track=track,
                    model_name=model_name,
                    expected_ids=expected_ids,
                    cases_by_id=cases_by_id,
                )
            except (json.JSONDecodeError, ValueError) as parse_exc:
                transient_invalid_json += 1
                if track == "human_effect":
                    predictions = []
                    fallback_failures = 0
                    for case in cases:
                        try:
                            predictions.append(
                                _predict_human_effect_final_fallback(
                                    module=module,
                                    sampler=sampler,
                                    tokenizer=tokenizer,
                                    case=case,
                                    mode=mode,
                                    model_name=model_name,
                                    timeout=timeout,
                                    retries=retries,
                                )
                            )
                        except (json.JSONDecodeError, ValueError) as fallback_exc:
                            fallback_failures += 1
                            predictions.append(
                                _failed_human_effect_prediction(
                                    case,
                                    model_name=model_name,
                                    error_message=_sanitize_error(fallback_exc),
                                )
                            )
                else:
                    repaired = _sample_text(
                        module=module,
                        sampler=sampler,
                        tokenizer=tokenizer,
                        messages=_repair_messages(
                            track=track,
                            mode=mode,
                            expected_ids=expected_ids,
                            raw_text=raw_text,
                        ),
                        timeout=timeout,
                        max_tokens=max_tokens,
                    )
                    predictions = _parse_predictions(
                        text=repaired,
                        track=track,
                        model_name=model_name,
                        expected_ids=expected_ids,
                        cases_by_id=cases_by_id,
                    )
                last_error_type = (
                    "invalid_json_recovered_with_failed_rows"
                    if track == "human_effect" and fallback_failures
                    else "invalid_json_recovered"
                )
                last_error_message = _sanitize_error(parse_exc)
            return predictions, {
                "chunk_status": "completed",
                "transient_api_errors": transient_api_errors,
                "transient_invalid_json": transient_invalid_json,
                "last_error_type": last_error_type,
                "last_error_message": last_error_message,
                "recovered_after_retry": attempt > 0 or transient_invalid_json > 0,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            transient_invalid_json += 1
            last_error_type = "invalid_json_or_schema"
            last_error_message = _sanitize_error(exc)
        except Exception as exc:
            transient_api_errors += 1
            last_error_type = "provider_or_runtime_error"
            last_error_message = _sanitize_error(exc)
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 8) + random.random() * 0.1)
    return None, {
        "chunk_status": "failed",
        "transient_api_errors": transient_api_errors,
        "transient_invalid_json": transient_invalid_json,
        "last_error_type": last_error_type,
        "last_error_message": last_error_message,
        "recovered_after_retry": False,
    }


def _empty_prediction_files(output_dir: Path, mode: str) -> None:
    for track in _allowed_tracks(mode):
        write_jsonl(output_dir / f"{mode}_{track}.jsonl", [])


def _dedupe_predictions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        benchmark_id = row.get("benchmark_id")
        if not benchmark_id:
            continue
        key = str(benchmark_id)
        if key not in by_id:
            order.append(key)
        by_id[key] = row
    return [by_id[key] for key in order]


def _prediction_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["benchmark_id"]) for row in rows if row.get("benchmark_id")}


def _failed_prediction_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("status") == "failed"
        or row.get("json_valid") is False
        or row.get("schema_valid") is False
        or row.get("error_type") == "unresolved_invalid_json"
    )


def _missing_cases(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_ids = _prediction_ids(predictions)
    return [case for case in cases if str(case.get("benchmark_id")) not in completed_ids]


def _order_predictions_for_cases(
    cases: list[dict[str, Any]], predictions: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["benchmark_id"]): row for row in _dedupe_predictions(predictions) if row.get("benchmark_id")}
    return [by_id[str(case["benchmark_id"])] for case in cases if str(case.get("benchmark_id")) in by_id]


def _sync_prediction_files(
    *,
    path: Path,
    leaderboard_prediction_dirs: list[Path],
    mode: str,
    track: str,
    predictions: list[dict[str, Any]],
) -> None:
    write_jsonl(path, predictions)
    for leaderboard_prediction_dir in leaderboard_prediction_dirs:
        write_jsonl(leaderboard_prediction_dir / f"{mode}_{track}.jsonl", predictions)


def _manifest_status(
    *,
    track_summaries: dict[str, Any],
    unresolved_api_errors: int,
    unresolved_invalid_json: int,
) -> tuple[str, Optional[str]]:
    incomplete = [
        track
        for track, summary in track_summaries.items()
        if int(summary.get("completed") or 0) != int(summary.get("attempted") or 0)
    ]
    if unresolved_api_errors or incomplete:
        return "incomplete", "remote_sampler_outputs_incomplete"
    if unresolved_invalid_json:
        return "completed_with_failures", "explicit_failed_prediction_rows"
    return "completed_clean", "completed_clean"


def _failed_human_effect_prediction(
    case: dict[str, Any],
    *,
    model_name: str,
    error_message: str = "unresolved_invalid_json",
) -> dict[str, Any]:
    benchmark_id = str(case["benchmark_id"])
    return {
        "prediction_id": f"{safe_model_slug(model_name)}:{benchmark_id}",
        "benchmark_id": benchmark_id,
        "track": "human_effect",
        "model_name": model_name,
        "category": "no_known_human_effect_evidence",
        "evidence_level": "unsupported_contradicted_or_unsafe_claim",
        "evidence_direction": "not_applicable",
        "claim_status": "insufficient_information",
        "safety_status": "insufficient_safety_data",
        "abstained": True,
        "rationale_source_ids": [],
        "status": "failed",
        "json_valid": False,
        "schema_valid": False,
        "error_type": "unresolved_invalid_json",
        "error_message": _sanitize_error(error_message),
    }


def infer(
    run_dir: Path,
    release_dir: Path,
    output_dir: Path,
    model_name: str,
    mode: str,
    *,
    leaderboard_dir: Optional[Path] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: int = 300,
    retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    adapter_manifest = _read_json(run_dir / "hf_upload" / "adapter_export_manifest.json") or _read_json(
        run_dir / "adapter_export_manifest.json"
    )
    base_model = str(run_manifest.get("model_name") or adapter_manifest.get("base_model") or DEFAULT_BASE_MODEL)
    local_files = _local_adapter_files(run_dir)
    if str(run_manifest.get("status")) != "training_completed":
        _empty_prediction_files(output_dir, mode)
        manifest = {
            "status": "blocked",
            "reason": "training_not_completed",
            "model_id": model_name,
            "mode": mode,
            "tracks": list(_allowed_tracks(mode)),
            "predictions_written": 0,
        }
        write_json(output_dir / f"{mode}_inference_manifest.json", manifest)
        return manifest

    leaderboard_prediction_dirs = (
        [leaderboard_dir / "predictions" / slug for slug in _artifact_slugs(model_name)]
        if leaderboard_dir
        else []
    )
    for leaderboard_prediction_dir in leaderboard_prediction_dirs:
        leaderboard_prediction_dir.mkdir(parents=True, exist_ok=True)
    tracks = list(_allowed_tracks(mode))
    track_summaries: dict[str, Any] = {}
    track_cases: dict[str, list[dict[str, Any]]] = {}
    existing_predictions: dict[str, list[dict[str, Any]]] = {}
    pending_cases: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        try:
            cases = _cases_for_track(release_dir, track)
        except FileNotFoundError:
            if not os.environ.get(TINKER_ENV_NAME):
                sampler_path = _sampler_path(run_dir)
                manifest = {
                    "status": "blocked",
                    "reason": "adapter_requires_tinker_auth_env",
                    "model_id": model_name,
                    "mode": mode,
                    "base_model": base_model,
                    "adapter_status": adapter_manifest.get("adapter_status")
                    or adapter_manifest.get("status", "unknown"),
                    "remote_sampler_path_available": bool(sampler_path),
                    "remote_sampler_path_hash": stable_hash(sampler_path) if sampler_path else None,
                    "local_adapter_files": [path.name for path in local_files],
                    "predictions_written": 0,
                    "tracks": tracks,
                    "track_summaries": {},
                    "missing_ids": {},
                    "unresolved_api_error_count": 0,
                    "unresolved_invalid_json_count": 0,
                    "retry_recovered_count": 0,
                    "run_timestamp": _utc_now(),
                }
                write_json(output_dir / f"{mode}_inference_manifest.json", manifest)
                return manifest
            raise
        track_cases[track] = cases
        pred_path = output_dir / f"{mode}_{track}.jsonl"
        rows = read_jsonl(pred_path) if resume and pred_path.exists() else []
        valid_ids = {str(case.get("benchmark_id")) for case in cases}
        ordered_existing = _order_predictions_for_cases(
            cases,
            [row for row in rows if str(row.get("benchmark_id")) in valid_ids],
        )
        existing_predictions[track] = ordered_existing if resume else []
        pending_cases[track] = _missing_cases(cases, existing_predictions[track])

    pending_total = sum(len(rows) for rows in pending_cases.values())
    sampler_path = _sampler_path(run_dir)
    module = sampler = tokenizer = None
    if pending_total:
        try:
            module, sampler, sampler_path, tokenizer = _create_sampler(run_dir, base_model)
        except InferenceBlocked as exc:
            for track in tracks:
                cases = track_cases[track]
                pred_path = output_dir / f"{mode}_{track}.jsonl"
                predictions = existing_predictions[track]
                missing = _missing_cases(cases, predictions)
                _sync_prediction_files(
                    path=pred_path,
                    leaderboard_prediction_dirs=leaderboard_prediction_dirs,
                    mode=mode,
                    track=track,
                    predictions=predictions,
                )
                track_summaries[track] = {
                    "status": "incomplete" if missing else "completed_from_existing",
                    "attempted": len(cases),
                    "completed": len(predictions),
                    "missing_ids": [str(case["benchmark_id"]) for case in missing],
                    "unresolved_api_errors": len(missing),
                    "unresolved_invalid_json": _failed_prediction_count(predictions),
                    "retry_recovered_count": 0,
                    "chunks": [],
                }
            unresolved_api_errors = sum(
                int(summary.get("unresolved_api_errors") or 0) for summary in track_summaries.values()
            )
            unresolved_invalid_json = sum(
                int(summary.get("unresolved_invalid_json") or 0) for summary in track_summaries.values()
            )
            status, reason = _manifest_status(
                track_summaries=track_summaries,
                unresolved_api_errors=unresolved_api_errors,
                unresolved_invalid_json=unresolved_invalid_json,
            )
            manifest = {
                "status": status,
                "reason": exc.reason if status == "incomplete" else reason,
                "details": exc.details,
                "model_id": model_name,
                "mode": mode,
                "base_model": base_model,
                "adapter_status": adapter_manifest.get("adapter_status")
                or adapter_manifest.get("status", "unknown"),
                "remote_sampler_path_available": bool(sampler_path),
                "remote_sampler_path_hash": stable_hash(sampler_path) if sampler_path else None,
                "local_adapter_files": [path.name for path in local_files],
                "predictions_written": sum(
                    int(summary.get("completed") or 0) for summary in track_summaries.values()
                ),
                "tracks": tracks,
                "track_summaries": track_summaries,
                "missing_ids": {
                    track: summary.get("missing_ids", []) for track, summary in track_summaries.items()
                },
                "unresolved_api_error_count": unresolved_api_errors,
                "unresolved_invalid_json_count": unresolved_invalid_json,
                "retry_recovered_count": 0,
                "run_timestamp": _utc_now(),
            }
            write_json(output_dir / f"{mode}_inference_manifest.json", manifest)
            return manifest

    run_timestamp = _utc_now()

    for track in tracks:
        cases = track_cases[track]
        pred_path = output_dir / f"{mode}_{track}.jsonl"
        predictions: list[dict[str, Any]] = list(existing_predictions[track])
        chunks: list[dict[str, Any]] = []
        if not pending_cases[track]:
            _sync_prediction_files(
                path=pred_path,
                leaderboard_prediction_dirs=leaderboard_prediction_dirs,
                mode=mode,
                track=track,
                predictions=predictions,
            )
            track_summaries[track] = {
                "status": "completed_from_existing",
                "attempted": len(cases),
                "completed": len(predictions),
                "missing_ids": [],
                "unresolved_api_errors": 0,
                "unresolved_invalid_json": _failed_prediction_count(predictions),
                "retry_recovered_count": 0,
                "chunks": [],
            }
            continue
        assert module is not None and sampler is not None and tokenizer is not None
        new_predictions: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(_chunked(pending_cases[track], chunk_size), start=1):
            chunk_predictions, chunk_summary = _predict_chunk(
                module=module,
                sampler=sampler,
                tokenizer=tokenizer,
                track=track,
                mode=mode,
                cases=chunk,
                model_name=model_name,
                timeout=timeout,
                max_tokens=_track_max_tokens(track, max_tokens),
                retries=retries,
            )
            chunk_record = {
                "chunk_index": chunk_index,
                "benchmark_ids": [case["benchmark_id"] for case in chunk],
                "attempted": len(chunk),
                **chunk_summary,
            }
            chunks.append(chunk_record)
            if chunk_predictions is None:
                if track == "human_effect" and chunk_summary["last_error_type"] == "invalid_json_or_schema":
                    new_predictions.extend(
                        _failed_human_effect_prediction(
                            case,
                            model_name=model_name,
                            error_message=chunk_summary.get("last_error_message")
                            or "unresolved_invalid_json",
                        )
                        for case in chunk
                    )
                continue
            new_predictions.extend(chunk_predictions)
        predictions = _order_predictions_for_cases(cases, [*predictions, *new_predictions])
        _sync_prediction_files(
            path=pred_path,
            leaderboard_prediction_dirs=leaderboard_prediction_dirs,
            mode=mode,
            track=track,
            predictions=predictions,
        )
        missing = _missing_cases(cases, predictions)
        track_summaries[track] = {
            "status": "completed_clean" if not missing else "incomplete",
            "attempted": len(cases),
            "completed": len(predictions),
            "missing_ids": [str(case["benchmark_id"]) for case in missing],
            "unresolved_api_errors": sum(
                chunk["attempted"]
                for chunk in chunks
                if chunk["chunk_status"] == "failed"
                and chunk["last_error_type"] != "invalid_json_or_schema"
            ),
            "unresolved_invalid_json": _failed_prediction_count(predictions)
            if track == "human_effect"
            else sum(
                chunk["attempted"]
                for chunk in chunks
                if chunk["chunk_status"] == "failed"
                and chunk["last_error_type"] == "invalid_json_or_schema"
            ),
            "retry_recovered_count": sum(
                chunk["attempted"]
                for chunk in chunks
                if chunk["chunk_status"] == "completed" and chunk.get("recovered_after_retry")
            ),
            "chunks": chunks,
        }

    total_predictions = sum(int(summary.get("completed") or 0) for summary in track_summaries.values())
    unresolved_api_errors = sum(
        int(summary.get("unresolved_api_errors") or 0) for summary in track_summaries.values()
    )
    unresolved_invalid_json = sum(
        int(summary.get("unresolved_invalid_json") or 0) for summary in track_summaries.values()
    )
    retry_recovered_count = sum(
        int(summary.get("retry_recovered_count") or 0) for summary in track_summaries.values()
    )
    status, reason = _manifest_status(
        track_summaries=track_summaries,
        unresolved_api_errors=unresolved_api_errors,
        unresolved_invalid_json=unresolved_invalid_json,
    )
    manifest = {
        "status": status,
        "reason": reason,
        "model_id": model_name,
        "mode": mode,
        "base_model": base_model,
        "adapter_status": adapter_manifest.get("adapter_status") or adapter_manifest.get("status", "unknown"),
        "remote_sampler_path_available": bool(sampler_path),
        "remote_sampler_path_hash": stable_hash(sampler_path) if sampler_path else None,
        "local_adapter_files": [path.name for path in local_files],
        "predictions_written": total_predictions,
        "tracks": tracks,
        "track_summaries": track_summaries,
        "missing_ids": {
            track: summary.get("missing_ids", []) for track, summary in track_summaries.items()
        },
        "unresolved_api_error_count": unresolved_api_errors,
        "unresolved_invalid_json_count": unresolved_invalid_json,
        "retry_recovered_count": retry_recovered_count,
        "run_timestamp": run_timestamp,
    }
    write_json(output_dir / f"{mode}_inference_manifest.json", manifest)
    return manifest


def infer_all(
    run_dir: Path,
    release_dir: Path,
    output_dir: Path,
    model_name: str,
    *,
    leaderboard_dir: Optional[Path] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: int = 300,
    retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    resume: bool = True,
) -> dict[str, Any]:
    manifests = {
        mode: infer(
            run_dir,
            release_dir,
            output_dir,
            model_name,
            mode,
            leaderboard_dir=leaderboard_dir,
            chunk_size=chunk_size,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
            resume=resume,
        )
        for mode in MODES
    }
    complete_statuses = {"completed_clean", "completed_with_failures", "completed"}
    status = "completed_clean" if all(item.get("status") in complete_statuses for item in manifests.values()) else "incomplete"
    if any(item.get("status") == "completed_with_failures" for item in manifests.values()):
        status = "completed_with_failures"
    if any(item.get("status") == "blocked" for item in manifests.values()):
        status = "blocked"
    summary = {
        "status": status,
        "model_id": model_name,
        "modes": manifests,
        "predictions_written": sum(int(item.get("predictions_written") or 0) for item in manifests.values()),
    }
    write_json(output_dir / "inference_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR / "eval" / "peb_predictions")
    parser.add_argument("--leaderboard-dir", type=Path, default=Path("leaderboard"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "all":
        result = infer_all(
            args.run_dir,
            args.release_dir,
            args.output_dir,
            args.model_name,
            leaderboard_dir=args.leaderboard_dir,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
            resume=not args.no_resume,
        )
    else:
        result = infer(
            args.run_dir,
            args.release_dir,
            args.output_dir,
            args.model_name,
            args.mode,
            leaderboard_dir=args.leaderboard_dir,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
            resume=not args.no_resume,
        )
    print(f"Inference status={result['status']}")


if __name__ == "__main__":
    main()
