"""Release and real-case audit checks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from peb.io import read_jsonl
from peb.schemas import (
    AuditResult,
    ClaimStatus,
    EvidenceLevel,
    QCStatus,
    SourceBucket,
    Track,
    validate_case_record,
)


def audit_records(records: list[dict]) -> AuditResult:
    errors: list[str] = []
    warnings: list[str] = []
    for index, record in enumerate(records, start=1):
        label = record.get("benchmark_id", f"record-{index}")
        try:
            case = validate_case_record(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: schema validation failed: {exc}")
            continue
        required = [
            case.source_database,
            case.source_id,
            case.source_version,
            case.retrieval_date,
            case.license_or_usage_note,
            case.qc_status.value,
            case.split.value,
        ]
        if not all(required):
            errors.append(f"{label}: missing required provenance field")
        if case.qc_status == QCStatus.unchecked:
            warnings.append(f"{label}: still marked unchecked")
        if case.evidence_quote_or_label and len(case.evidence_quote_or_label.split()) > 35:
            errors.append(f"{label}: copied source text is too long")
        if case.track == Track.human_effect:
            if (
                getattr(case, "evidence_level", None) == EvidenceLevel.approved_human_indication
                and case.source_database.lower() not in {"dailymed", "regulatory_label"}
            ):
                errors.append(f"{label}: approved indication requires label/regulatory source")
            if getattr(case, "trial_has_results", None) is False and getattr(
                case, "evidence_direction", None
            ).value == "positive":
                errors.append(f"{label}: trial existence without results cannot be positive evidence")
            if getattr(case, "claim_status", None) == ClaimStatus.supported and getattr(
                case, "evidence_level", None
            ) == EvidenceLevel.mechanistic_pathway_or_similarity_hypothesis:
                errors.append(f"{label}: pathway-only evidence cannot be marked supported")
        if case.track == Track.binding_rank:
            if not getattr(case, "normalization_method", None):
                errors.append(f"{label}: binding panel missing normalization method")
            if getattr(case, "measurement_direction", None).value == "unknown":
                errors.append(f"{label}: binding panel has unknown measurement direction")
            if not getattr(case, "comparable_panel", True) and not getattr(
                case, "panel_exclusion_reason", None
            ):
                errors.append(f"{label}: incomparable panel needs exclusion reason")
        if case.track == Track.pose:
            if not getattr(case, "target_chain_id", None) or not getattr(case, "peptide_chain_id", None):
                errors.append(f"{label}: pose case needs explicit target and peptide chains")
            if not getattr(case, "native_contacts", []):
                warnings.append(f"{label}: pose case has no contact labels")
    return AuditResult(passed=not errors, errors=errors, warnings=warnings)


def audit_jsonl(path: Union[str, Path]) -> AuditResult:
    return audit_records(read_jsonl(path))


def check_bucket_policy(source_bucket: str, raw_allowed: str) -> Optional[str]:
    if source_bucket in {SourceBucket.B.value, SourceBucket.C.value} and raw_allowed != "allowed":
        return "Bucket B/C source requires nonredistributable index unless license allows raw release"
    return None
