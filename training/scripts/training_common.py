"""Shared helpers for training, inference, and evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RUN_DIR = Path("training/runs/peb-engineer-v0")
DEFAULT_RELEASE_DIR = Path("data/releases/peb-v1.0-rc")
TINKER_ENV_NAME = "TINKER_API_KEY"

SECRET_PATTERNS = [
    re.compile(r"tml-[A-Za-z0-9_-]+"),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9_-]{20,}"),
    re.compile(r"OPENROUTER_API_KEY\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"TINKER_API_KEY\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"HF_TOKEN\s*=\s*['\"][^'\"]+['\"]"),
]


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA256 hash for strings or JSON-serializable objects."""
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )



def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")


def strict_response_template() -> dict[str, Any]:
    """Canonical compact PEB report skeleton for strict JSON responses."""
    return {
        "developability": {
            "stability_risk": "unknown",
            "solubility_risk": "unknown",
            "toxicity_risk": "unknown",
            "hemolysis_risk": "unknown",
            "cytotoxicity_risk": "unknown",
            "synthesis_complexity": "unknown",
        },
        "functional_assay_estimate": {
            "categories": [],
            "confidence": "low",
            "evidence_basis": "insufficient_information",
        },
        "human_effect_estimate": {
            "category": "no_known_human_effect_evidence",
            "claim_status": "insufficient_information",
            "confidence": "low",
            "evidence_direction": "not_applicable",
            "evidence_level": "unsupported_contradicted_or_unsafe_claim",
        },
        "known_source_backed_facts": [],
        "missing_evidence": [],
        "overall_confidence": "low",
        "pose_contact_assessment": {
            "status": "not_assessed",
            "confidence": "low",
            "interface_residues": [],
            "predicted_contacts": [],
            "notes": "",
        },
        "recommended_next_assays": [],
        "structure_source_reference": {
            "status": "not_applicable",
            "confidence": "low",
            "notes": "",
        },
        "target_binding": {
            "status": "not_assessed",
            "confidence": "low",
            "relative_rank": [],
            "assay_awareness": "",
        },
        "unsupported_claims": [],
    }
