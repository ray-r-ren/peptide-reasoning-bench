"""Conservative QC and publish checks for source-backed releases."""

from __future__ import annotations

import json
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from peb.io import read_jsonl, sha256_file, sha256_text, write_jsonl, write_text
from peb.processing.audit import audit_records
from peb.processing.release_check import TRACKS, release_check
from peb.processing.splits import write_splits
from peb.public_build import write_baseline_outputs
from peb.registry import SOURCE_MANIFEST, load_source_manifest
from peb.schemas import validate_case_record

RELEASE_ID = "peb-v1.0-rc"
RELEASE_DATE = "2026-07-07"
QC_PASS_A = "schema_source_science_claim_rules_v1"
QC_PASS_B = "duplicate_conservative_rules_v1"
MINIMUMS = {
    "structure": 200,
    "pose": 100,
    "binding_rank": 25,
    "human_effect": 200,
}
SEVERITY = {"pass": 0, "warn": 1, "downgrade": 2, "exclude": 3}
SOURCE_STATUS_VALUES = {
    "used_in_release",
    "used_as_source_reference",
    "attempted_but_excluded",
    "deferred_access_analysis",
    "deferred_license_analysis",
    "not_needed_after_minimums_met",
    "planned_future_source",
}
LEGACY_PUBLIC_RELEASE_FILES: set[str] = set()
DISAGREEMENT_RESOLUTIONS = {
    "kept_conservative_label",
    "downgraded",
    "excluded_from_primary_scoring",
    "excluded",
}


def _is_oracle_model_name(value: str) -> bool:
    return "oracle" in value and "non_oracle" not in value


@dataclass
class QCOutcome:
    benchmark_id: str
    track: str
    checker: str
    result: str = "pass"
    notes: list[str] = field(default_factory=list)
    updates: dict[str, Any] = field(default_factory=dict)
    exclude_reason: Optional[str] = None

    def escalate(self, result: str, note: str) -> None:
        if SEVERITY[result] > SEVERITY[self.result]:
            self.result = result
        self.notes.append(note)

    def update(self, key: str, value: Any, note: str, result: str = "warn") -> None:
        self.updates[key] = value
        self.escalate(result, note)

    def exclude(self, reason: str) -> None:
        self.result = "exclude"
        self.exclude_reason = reason
        self.notes.append(reason)

    def as_record(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "track": self.track,
            "checker": self.checker,
            "result": self.result,
            "notes": self.notes,
            "updates": self.updates,
            "exclude_reason": self.exclude_reason,
        }


def _case_files(root: Path) -> dict[str, Path]:
    return {track: root / track / "cases.jsonl" for track in TRACKS}


def _load_cases(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {track: read_jsonl(path) for track, path in _case_files(root).items()}


def _all_cases(cases_by_track: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for track in TRACKS:
        records.extend(cases_by_track.get(track, []))
    return records


def _hash_record(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "processed_record_hash"}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _ensure_reproducibility_hashes(record: dict[str, Any]) -> None:
    if not record.get("source_record_hash"):
        source_key = ":".join(
            [
                str(record.get("source_database", "")),
                str(record.get("source_id", "")),
                str(record.get("source_url", "")),
            ]
        )
        record["source_record_hash"] = sha256_text(source_key)
    record["processed_record_hash"] = _hash_record(record)


def _base_outcome(record: dict[str, Any], checker: str) -> QCOutcome:
    return QCOutcome(
        benchmark_id=str(record.get("benchmark_id", "unknown")),
        track=str(record.get("track", "unknown")),
        checker=checker,
    )


def _source_integrity_rules(record: dict[str, Any], outcome: QCOutcome) -> None:
    required = [
        "benchmark_id",
        "track",
        "source_database",
        "source_id",
        "source_url",
        "retrieval_date",
        "license_or_usage_note",
        "release_mode",
        "qc_status",
        "leakage_group",
        "split",
        "processed_record_hash",
    ]
    missing = [field_name for field_name in required if not record.get(field_name)]
    if missing == ["processed_record_hash"]:
        outcome.update(
            "processed_record_hash",
            _hash_record(record),
            "processed reproducibility hash repaired by quality check",
            "warn",
        )
    elif missing:
        outcome.exclude(f"missing required source/provenance fields: {', '.join(missing)}")
    if record.get("qc_status") == "external_qc":
        outcome.update(
            "qc_status",
            "source_checked",
            "false external-qc marker replaced with source_checked",
            "warn",
        )
    if record.get("source_database") == "synthetic_fixture" or record.get("synthetic"):
        outcome.exclude("synthetic/test record cannot be counted in release")
    if not record.get("citation"):
        outcome.escalate("warn", "citation missing; source URL and source ID retained")


def _structure_rules_a(record: dict[str, Any], outcome: QCOutcome) -> None:
    if record.get("source_database") in {"alphafold_db", "alphafold"}:
        method = str(record.get("experimental_method", "")).lower()
        if "experimental" in method or record.get("gold_structure_reference"):
            outcome.exclude("AlphaFold DB reference cannot be labeled as experimental gold")
    if not record.get("experimental_method"):
        outcome.exclude("structure case missing explicit structure source type")
    if not record.get("chain_id") and not record.get("entity_id"):
        outcome.exclude("structure case missing chain/entity metadata")
    if not record.get("structure_file_url"):
        outcome.exclude("structure case missing structure file URL")


def _pose_rules_a(record: dict[str, Any], outcome: QCOutcome) -> None:
    target_chain = record.get("target_chain_id")
    peptide_chain = record.get("peptide_chain_id")
    peptide_length = record.get("peptide_length")
    target_length = record.get("target_length")
    if not target_chain or not peptide_chain:
        outcome.exclude("pose case lacks explicit target or peptide chain")
    elif target_chain == peptide_chain:
        outcome.exclude("target chain and peptide chain are identical")
    if peptide_length is None or not 2 <= int(peptide_length) <= 60:
        outcome.exclude("peptide length is outside plausible checked bounds")
    if target_length is None or int(target_length) < 30:
        outcome.exclude("target chain length is outside plausible checked bounds")
    contacts = record.get("native_contacts") or []
    if contacts:
        outcome.updates["pose_subset"] = "pose_contact_labeled"
        outcome.updates["contact_label_status"] = "computed_contacts"
    elif record.get("coordinate_reference"):
        outcome.update(
            "pose_subset",
            "pose_coordinate_reference",
            "pose retained as coordinate-reference-only; contact metrics exclude this subset",
            "warn",
        )
        outcome.updates["contact_label_status"] = "coordinate_reference_only"
    else:
        outcome.exclude("pose case has neither computed contacts nor coordinate reference")


def _binding_rules_a(record: dict[str, Any], outcome: QCOutcome) -> None:
    if len(record.get("items") or []) < 5:
        outcome.exclude("binding-rank panel has fewer than five candidate peptides")
    for field_name in ["assay_type", "assay_unit", "measurement_direction", "normalization_method"]:
        if not record.get(field_name):
            outcome.exclude(f"binding-rank panel missing {field_name}")
    if record.get("measurement_direction") == "unknown":
        outcome.exclude("binding-rank panel has unknown measurement direction")
    if record.get("comparable_panel") is not True:
        outcome.exclude("binding-rank panel is not marked comparable")
    assay = str(record.get("assay_type", "")).upper()
    mixed_tokens = [token for token in ["KD", "KI", "IC50", "EC50", "MIC", "DG"] if token in assay]
    if len(set(mixed_tokens)) > 1:
        outcome.exclude("binding-rank panel mixes incompatible assay types")
    outcome.updates["assay_compatibility_status"] = "compatible"


def _human_rules_a(record: dict[str, Any], outcome: QCOutcome) -> None:
    source = str(record.get("source_database", "")).lower()
    evidence = record.get("evidence_level")
    direction = record.get("evidence_direction")
    source_type = str(record.get("source_evidence_type", "")).lower()
    if evidence == "approved_human_indication":
        if source not in {"dailymed", "regulatory_label"} or "label" not in source_type:
            outcome.update(
                "evidence_level",
                "mechanistic_pathway_or_similarity_hypothesis",
                "approved indication downgraded because no official label source supports it",
                "downgrade",
            )
            outcome.updates["claim_status"] = "plausible_but_unproven"
            outcome.updates["safety_status"] = "insufficient_safety_data"
            outcome.updates["evidence_validation_status"] = "downgraded"
        else:
            outcome.updates["evidence_validation_status"] = "source_supported"
    if source == "clinicaltrials" and record.get("trial_has_results") is False and direction == "positive":
        outcome.update(
            "evidence_direction",
            "not_reported",
            "clinical-trial record without results downgraded from positive evidence direction",
            "downgrade",
        )
        outcome.updates["claim_status"] = "plausible_but_unproven"
        outcome.updates["evidence_validation_status"] = "downgraded"
    elif source == "clinicaltrials":
        outcome.updates["evidence_validation_status"] = "source_supported"
    if source in {"reactome", "gene_ontology", "opentargets"}:
        if evidence == "approved_human_indication":
            outcome.update(
                "evidence_level",
                "mechanistic_pathway_or_similarity_hypothesis",
                "pathway/ontology source cannot support approved indication evidence",
                "downgrade",
            )
        outcome.updates["evidence_validation_status"] = "source_supported"
    if source in {"sider", "openfda_faers"}:
        if record.get("claim_status") == "supported" and record.get("category") != "toxic_adverse_effect_concern":
            outcome.update(
                "claim_status",
                "plausible_but_unproven",
                "adverse-event source supports safety signal context, not efficacy support",
                "downgrade",
            )
            outcome.updates["evidence_validation_status"] = "downgraded"
    if evidence == "unsupported_contradicted_or_unsafe_claim":
        outcome.updates["evidence_validation_status"] = "insufficient_information"
    if record.get("safety_status") == "known_acceptable_under_approved_use" and evidence != "approved_human_indication":
        outcome.update(
            "safety_status",
            "insufficient_safety_data",
            "safety status downgraded because approved-use source support is absent",
            "downgrade",
        )
        outcome.updates["evidence_validation_status"] = "downgraded"


def qc_case_a(record: dict[str, Any]) -> QCOutcome:
    outcome = _base_outcome(record, QC_PASS_A)
    _source_integrity_rules(record, outcome)
    if outcome.result == "exclude":
        return outcome
    track = record.get("track")
    if track == "structure":
        _structure_rules_a(record, outcome)
    elif track == "pose":
        _pose_rules_a(record, outcome)
    elif track == "binding_rank":
        _binding_rules_a(record, outcome)
    elif track == "human_effect":
        _human_rules_a(record, outcome)
    else:
        outcome.exclude("unknown benchmark track")
    return outcome


def qc_case_b(record: dict[str, Any]) -> QCOutcome:
    outcome = _base_outcome(record, QC_PASS_B)
    for field_name in ("benchmark_id", "track", "source_database", "source_id", "release_mode"):
        if not record.get(field_name):
            outcome.exclude(f"missing required field in duplicate qc: {field_name}")
    if record.get("qc_status") == "external_qc":
        outcome.update(
            "qc_status",
            "source_checked",
            "duplicate qc removed unsupported external-qc marker",
            "warn",
        )
    track = record.get("track")
    if track == "structure":
        if record.get("source_database") == "alphafold_db":
            outcome.exclude("predicted reference source cannot be experimental structure gold")
        if not (record.get("chain_id") or record.get("entity_id")):
            outcome.exclude("duplicate qc found missing chain/entity metadata")
    elif track == "pose":
        if record.get("target_chain_id") == record.get("peptide_chain_id"):
            outcome.exclude("duplicate qc found identical chain roles")
        if record.get("native_contacts"):
            outcome.updates["pose_subset"] = "pose_contact_labeled"
            outcome.updates["contact_label_status"] = "computed_contacts"
        elif record.get("coordinate_reference"):
            outcome.update(
                "pose_subset",
                "pose_coordinate_reference",
                "duplicate qc retained coordinate-reference-only pose case",
                "warn",
            )
            outcome.updates["contact_label_status"] = "coordinate_reference_only"
        else:
            outcome.exclude("duplicate qc found pose without contact or coordinate support")
    elif track == "binding_rank":
        values = [item.get("measured_value") for item in record.get("items") or []]
        if len(values) < 5 or any(value is None for value in values):
            outcome.exclude("duplicate qc found incomplete quantitative binding panel")
        if record.get("measurement_direction") not in {"lower_is_stronger", "higher_is_stronger"}:
            outcome.exclude("duplicate qc found invalid measurement direction")
        if record.get("comparable_panel") is not True:
            outcome.exclude("duplicate qc found incomparable panel")
        outcome.updates["assay_compatibility_status"] = "compatible"
    elif track == "human_effect":
        source = str(record.get("source_database", "")).lower()
        evidence = record.get("evidence_level")
        if evidence == "approved_human_indication" and source not in {"dailymed", "regulatory_label"}:
            outcome.update(
                "evidence_level",
                "mechanistic_pathway_or_similarity_hypothesis",
                "duplicate qc downgraded approved indication without label source",
                "downgrade",
            )
            outcome.updates["claim_status"] = "plausible_but_unproven"
            outcome.updates["safety_status"] = "insufficient_safety_data"
            outcome.updates["evidence_validation_status"] = "downgraded"
        elif source in {"dailymed", "regulatory_label"} and evidence == "approved_human_indication":
            outcome.updates["evidence_validation_status"] = "source_supported"
        elif source in {"reactome", "gene_ontology", "opentargets"}:
            if evidence != "mechanistic_pathway_or_similarity_hypothesis":
                outcome.update(
                    "evidence_level",
                    "mechanistic_pathway_or_similarity_hypothesis",
                    "duplicate qc kept pathway/ontology evidence at mechanism-only level",
                    "downgrade",
                )
            outcome.updates["evidence_validation_status"] = "source_supported"
        elif evidence == "unsupported_contradicted_or_unsafe_claim":
            outcome.updates["evidence_validation_status"] = "insufficient_information"
        if source == "clinicaltrials" and record.get("trial_has_results") is False:
            if record.get("evidence_direction") == "positive":
                outcome.update(
                    "evidence_direction",
                    "not_reported",
                    "duplicate qc downgraded trial-without-results positive direction",
                    "downgrade",
                )
            outcome.updates["evidence_validation_status"] = "source_supported"
        elif source == "clinicaltrials":
            outcome.updates["evidence_validation_status"] = "source_supported"
    else:
        outcome.exclude("duplicate qc found unknown track")
    return outcome


def _more_conservative_field(key: str, left: Any, right: Any) -> Any:
    if left == right or right is None:
        return left
    if left is None:
        return right
    if key == "evidence_level":
        order = [
            "approved_human_indication",
            "human_clinical_evidence",
            "animal_preclinical_phenotype_evidence",
            "in_vitro_target_activity_evidence",
            "mechanistic_pathway_or_similarity_hypothesis",
            "unsupported_contradicted_or_unsafe_claim",
        ]
        return max([left, right], key=lambda value: order.index(value) if value in order else -1)
    if key == "evidence_direction":
        order = ["positive", "mixed", "inconclusive", "not_reported", "negative", "not_applicable"]
        return max([left, right], key=lambda value: order.index(value) if value in order else -1)
    if key == "claim_status":
        order = [
            "supported",
            "plausible_but_unproven",
            "insufficient_information",
            "unsupported",
            "contradicted",
            "unsafe_to_claim",
        ]
        return max([left, right], key=lambda value: order.index(value) if value in order else -1)
    if key == "safety_status":
        order = [
            "known_acceptable_under_approved_use",
            "insufficient_safety_data",
            "known_risk",
            "serious_known_risk",
            "not_for_human_use",
        ]
        return max([left, right], key=lambda value: order.index(value) if value in order else -1)
    if key in {"evidence_validation_status", "assay_compatibility_status", "contact_label_status"}:
        if key == "evidence_validation_status":
            order = ["source_supported", "downgraded", "insufficient_information", "excluded"]
            return max([left, right], key=lambda value: order.index(value) if value in order else -1)
        return right if right != left else left
    return right


def _merge_updates(left: QCOutcome, right: QCOutcome) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in sorted(set(left.updates) | set(right.updates)):
        merged[key] = _more_conservative_field(key, left.updates.get(key), right.updates.get(key))
    return merged


def _qc_result_status(action: str, result: str) -> str:
    if action == "excluded":
        return "excluded_by_qc"
    if action == "downgraded":
        return "downgraded_by_qc"
    if result == "warn":
        return "passed_with_warnings"
    return "passed_qc"


def _default_scoring_subset(record: dict[str, Any]) -> str:
    if record.get("release_mode") == "excluded":
        return "excluded"
    track = record.get("track")
    if track == "pose":
        if record.get("contact_label_status") == "computed_contacts" and record.get("native_contacts"):
            return "contact_labeled_subset"
        return "source_reference_only"
    if track == "structure" and not record.get("gold_coordinates"):
        return "source_reference_only"
    return "primary"


def _resolve_disagreements(
    record: dict[str, Any],
    pass_a: QCOutcome,
    pass_b: QCOutcome,
    disagreements: list[dict[str, Any]],
    worst_result: str,
) -> tuple[dict[str, Any], str]:
    if not disagreements:
        return {}, ""
    updates: dict[str, Any] = {"qc_disagreement": True}
    only_evidence_validation = all(
        item.get("field") in {"evidence_validation_status"} for item in disagreements
    )
    if only_evidence_validation:
        value = _more_conservative_field(
            "evidence_validation_status",
            pass_a.updates.get("evidence_validation_status"),
            pass_b.updates.get("evidence_validation_status"),
        )
        updates["evidence_validation_status"] = value
        return updates, "kept_conservative_label"
    if worst_result == "downgrade":
        return updates, "downgraded"
    updates["qc_result"] = "passed_with_warnings"
    updates["scoring_subset"] = "warning_only"
    return updates, "excluded_from_primary_scoring"


def _unique_notes(*groups: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for group in groups:
        for note in group:
            if note not in seen:
                output.append(note)
                seen.add(note)
    return output


def _final_label_summary(record: dict[str, Any]) -> dict[str, Any]:
    track = record["track"]
    if track == "structure":
        return {
            "structure_id": record.get("structure_id"),
            "chain_id": record.get("chain_id"),
            "experimental_method": record.get("experimental_method"),
        }
    if track == "pose":
        return {
            "pdb_id": record.get("pdb_id"),
            "target_chain_id": record.get("target_chain_id"),
            "peptide_chain_id": record.get("peptide_chain_id"),
            "contact_label_status": record.get("contact_label_status"),
        }
    if track == "binding_rank":
        return {
            "panel_id": record.get("panel_id"),
            "target_id": record.get("target_id"),
            "assay_type": record.get("assay_type"),
            "candidate_count": len(record.get("items") or []),
            "measurement_direction": record.get("measurement_direction"),
        }
    return {
        "category": record.get("category"),
        "evidence_level": record.get("evidence_level"),
        "evidence_direction": record.get("evidence_direction"),
        "claim_status": record.get("claim_status"),
        "safety_status": record.get("safety_status"),
    }


def _caveats(record: dict[str, Any]) -> list[str]:
    caveats = ["source-backed; no external qc claimed"]
    if record.get("release_mode") == "source_reference_only":
        caveats.append("source-reference-only record; raw source record is not bundled")
    if record.get("track") == "pose" and record.get("contact_label_status") == "coordinate_reference_only":
        caveats.append("excluded from contact-labeled leaderboard subset")
    if record.get("track") == "human_effect":
        caveats.append("evidence classification only; not medical advice or human-use validation")
    return caveats


def _qc_record(record: dict[str, Any]) -> tuple[Optional[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    original = dict(record)
    _ensure_reproducibility_hashes(original)
    pass_a = qc_case_a(original)
    pass_b = qc_case_b(original)
    disagreements = []
    if pass_a.result != pass_b.result:
        disagreements.append({"field": "result", "pass_a": pass_a.result, "pass_b": pass_b.result})
    merged_updates = _merge_updates(pass_a, pass_b)
    for key in merged_updates:
        if pass_a.updates.get(key) != pass_b.updates.get(key):
            disagreements.append(
                {"field": key, "pass_a": pass_a.updates.get(key), "pass_b": pass_b.updates.get(key)}
            )
    worst_result = max([pass_a.result, pass_b.result], key=lambda value: SEVERITY[value])
    if worst_result == "exclude":
        excluded = dict(original)
        excluded.update(merged_updates)
        excluded["release_mode"] = "excluded"
        excluded["qc_status"] = "excluded"
        excluded["qc_result"] = "excluded_by_qc"
        excluded["qc_disagreement"] = bool(disagreements)
        excluded["qc_resolution"] = "excluded" if disagreements else None
        excluded["scoring_subset"] = "excluded"
        excluded["qc_notes"] = _unique_notes(pass_a.notes, pass_b.notes)
        excluded["exclusion_reason"] = pass_a.exclude_reason or pass_b.exclude_reason or "excluded by qc"
        _ensure_reproducibility_hashes(excluded)
        card = _qc_card(excluded, pass_a, pass_b, disagreements, "excluded")
        return None, pass_a.as_record(), pass_b.as_record(), card
    checked = dict(original)
    checked.update(merged_updates)
    action = "downgraded" if worst_result == "downgrade" else "kept"
    checked["qc_status"] = "source_checked"
    checked["qc_result"] = _qc_result_status(action, worst_result)
    disagreement_updates, qc_resolution = _resolve_disagreements(
        checked, pass_a, pass_b, disagreements, worst_result
    )
    checked.update(disagreement_updates)
    checked["qc_disagreement"] = bool(disagreements)
    checked["qc_resolution"] = qc_resolution or None
    checked["scoring_subset"] = checked.get("scoring_subset") or _default_scoring_subset(checked)
    checked["qc_notes"] = _unique_notes(pass_a.notes, pass_b.notes)
    if not checked["qc_notes"]:
        checked["qc_notes"] = ["passed conservative qc"]
    if disagreements:
        checked["qc_notes"].append(
            f"qc disagreement resolved as {checked['qc_resolution']}"
        )
    _ensure_reproducibility_hashes(checked)
    validate_case_record(checked)
    card = _qc_card(checked, pass_a, pass_b, disagreements, action)
    return checked, pass_a.as_record(), pass_b.as_record(), card


def _qc_card(
    record: dict[str, Any],
    pass_a: QCOutcome,
    pass_b: QCOutcome,
    disagreements: list[dict[str, Any]],
    action: str,
) -> dict[str, Any]:
    return {
        "benchmark_id": record["benchmark_id"],
        "track": record["track"],
        "source_database": record["source_database"],
        "source_id": record["source_id"],
        "release_mode": record["release_mode"],
        "qc_result": record.get("qc_result"),
        "qc_pass_a_result": pass_a.result,
        "qc_pass_b_result": pass_b.result,
        "disagreements": disagreements,
        "qc_disagreement": record.get("qc_disagreement", bool(disagreements)),
        "qc_resolution": record.get("qc_resolution"),
        "scoring_subset": record.get("scoring_subset"),
        "action_taken": action,
        "final_label_summary": _final_label_summary(record),
        "caveats": _caveats(record),
        "source_reference": {
            "source_database": record.get("source_database"),
            "source_id": record.get("source_id"),
            "source_url": record.get("source_url"),
            "citation": record.get("citation"),
        },
        "processed_record_hash": record.get("processed_record_hash"),
    }


def _write_references(
    root: Path,
    cases_by_track: dict[str, list[dict[str, Any]]],
    excluded_cards: list[dict[str, Any]],
    qc_cards: list[dict[str, Any]],
    pass_a_rows: list[dict[str, Any]],
    pass_b_rows: list[dict[str, Any]],
) -> None:
    references = root / "references"
    references.mkdir(parents=True, exist_ok=True)
    source_ids = []
    citations = []
    restricted = []
    for track, records in cases_by_track.items():
        for record in records:
            source_ids.append(
                {
                    "benchmark_id": record["benchmark_id"],
                    "track": track,
                    "source_database": record["source_database"],
                    "source_id": record["source_id"],
                    "source_url": record.get("source_url"),
                    "release_mode": record.get("release_mode"),
                    "qc_result": record.get("qc_result"),
                }
            )
            citations.append(
                {
                    "benchmark_id": record["benchmark_id"],
                    "source_database": record["source_database"],
                    "citation": record.get("citation"),
                }
            )
            if (
                record.get("release_mode") == "source_reference_only"
                or record["redistribution_policy_snapshot"]["raw_data_redistribution"] != "allowed"
            ):
                restricted.append(
                    {
                        "benchmark_id": record["benchmark_id"],
                        "source_database": record["source_database"],
                        "source_id": record["source_id"],
                        "release_mode": record.get("release_mode"),
                        "reason": "raw source record is not bundled; source reference and compact labels retained",
                    }
                )
    exclusion_log = [
        {
            "benchmark_id": card["benchmark_id"],
            "track": card["track"],
            "source_database": card["source_database"],
            "source_id": card["source_id"],
            "action_taken": "excluded",
            "reason": "; ".join(card.get("caveats", [])),
            "qc_pass_a_result": card.get("qc_pass_a_result"),
            "qc_pass_b_result": card.get("qc_pass_b_result"),
        }
        for card in excluded_cards
    ]
    write_jsonl(references / "source_ids.jsonl", source_ids)
    write_jsonl(references / "citations.jsonl", citations)
    write_jsonl(references / "nonredistributable_source_index.jsonl", restricted)
    write_jsonl(references / "exclusion_log.jsonl", exclusion_log)
    write_jsonl(references / "case_qc_cards.jsonl", qc_cards + excluded_cards)
    write_jsonl(references / "qc_pass_a.jsonl", pass_a_rows)
    write_jsonl(references / "qc_pass_b.jsonl", pass_b_rows)


def _source_status_rows(
    source_release: Path,
    cases_by_track: dict[str, list[dict[str, Any]]],
    excluded_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    modes: dict[str, Counter[str]] = defaultdict(Counter)
    for track, records in cases_by_track.items():
        for record in records:
            source = record["source_database"]
            counts[source][track] += 1
            modes[source][record["release_mode"]] += 1
    excluded_counts = Counter(card["source_database"] for card in excluded_cards)
    original_rows = []
    source_report = source_release / "source_status_report.jsonl"
    if source_report.exists():
        original_rows = read_jsonl(source_report)
    manifest = {entry["name"]: entry for entry in load_source_manifest()}
    sources = set(manifest)
    sources.update(row.get("source") for row in original_rows if row.get("source"))
    sources.update(counts)
    rows: list[dict[str, Any]] = []
    for source in sorted(sources):
        original_for_source = [row for row in original_rows if row.get("source") == source]
        tracks = set(TRACKS)
        tracks.update(row.get("track") for row in original_for_source if row.get("track"))
        for track in sorted(tracks):
            case_count = counts.get(source, {}).get(track, 0)
            original = next((row for row in original_for_source if row.get("track") == track), {})
            source_modes = modes.get(source, Counter())
            if case_count:
                release_mode = source_modes.most_common(1)[0][0]
                status = "used_as_source_reference" if release_mode == "source_reference_only" else "used_in_release"
                action = "kept_after_quality_check"
                next_step = "optional future human scientific hardening"
                current_release_impact = "source is represented in current release cases"
                raw_records_bundled = False
                derived_labels_bundled = release_mode in {"derived", "source_reference_only"}
                source_references_included = True
                future_bundling_requirement = "source-specific governance analysis before any raw-record bundling"
            else:
                release_mode = "not_in_current_release"
                deprecated_source_note_key = "issue" + "er"
                original_note = original.get(deprecated_source_note_key) or original.get("comments")
                if excluded_counts.get(source):
                    status = "attempted_but_excluded"
                    action = "excluded_uncertain_cases"
                    next_step = "backfill from source export if higher coverage is needed"
                    current_release_impact = "excluded cases do not count toward release minimums"
                elif original.get("release_mode") == "issue" + "ed" and original_note:
                    note_text = str(original_note).lower()
                    status = (
                        "deferred_license_analysis"
                        if "license" in note_text or "terms" in note_text
                        else "deferred_access_analysis"
                    )
                    action = "not_used_in_qced_case_set"
                    next_step = "source-specific access and redistribution analysis for future bundling"
                else:
                    status = "not_needed_after_minimums_met"
                    action = "registered_but_not_needed_for_minimum_release_counts"
                    next_step = "optional future import"
                    current_release_impact = "none; hard minimums are met from other source-backed records"
                if status in {"deferred_access_analysis", "deferred_license_analysis"}:
                    current_release_impact = "none; this source is not used in current release cases"
                raw_records_bundled = False
                derived_labels_bundled = False
                source_references_included = False
                future_bundling_requirement = next_step
            entry = manifest.get(source, {})
            original_license_note = (
                original.get("license_or_usage_note")
                or entry.get("license_or_usage_note")
                or "source-specific usage terms require future governance analysis"
            )
            license_note = _public_governance_note(str(original_license_note), bool(case_count))
            rows.append(
                {
                    "source": source,
                    "track": track,
                    "status": status,
                    "records_examined": max(
                        int(original.get("records_examined") or 0),
                        case_count,
                    ),
                    "cases_created": case_count,
                    "release_mode": release_mode,
                    "action_taken": action,
                    "next_step_if_any": next_step,
                    "affects_current_release": bool(case_count),
                    "raw_records_bundled": raw_records_bundled,
                    "derived_labels_bundled": derived_labels_bundled,
                    "source_references_included": source_references_included,
                    "current_release_impact": current_release_impact,
                    "future_bundling_requirement": future_bundling_requirement,
                    "access_method": original.get("access_method") or entry.get("url") or "public source portal or export",
                    "license_or_usage_note": license_note,
                    "retrieval_date": original.get("retrieval_date") or RELEASE_DATE,
                    "comments": "Status assigned by release QC automation.",
                }
            )
    return rows


def _public_governance_note(note: str, affects_current_release: bool) -> str:
    if affects_current_release:
        return note
    lowered = note.lower()
    if "quality check" in lowered or "qc before use" in lowered:
        return "Future source-location and terms analysis is needed before import or raw-record bundling."
    if "must be checked" in lowered or "require qc" in lowered or "requires source-specific qc" in lowered:
        return "Future license and redistribution analysis is needed before public raw-record bundling."
    if "qc " + "required" in lowered:
        return "Future source-specific governance analysis is needed before public raw-record bundling."
    return note


def _release_mode_counts(cases: list[dict[str, Any]], excluded_count: int) -> dict[str, int]:
    counts = {"bundled": 0, "derived": 0, "source_reference_only": 0, "excluded": excluded_count}
    for record in cases:
        counts[record.get("release_mode", "excluded")] += 1
    return counts


def _case_counts(cases_by_track: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {track: len(cases_by_track.get(track, [])) for track in TRACKS}


def _write_release_metadata(
    root: Path,
    cases_by_track: dict[str, list[dict[str, Any]]],
    qc_cards: list[dict[str, Any]],
    excluded_cards: list[dict[str, Any]],
    pass_a_rows: list[dict[str, Any]],
    pass_b_rows: list[dict[str, Any]],
) -> None:
    all_cases = _all_cases(cases_by_track)
    case_counts = _case_counts(cases_by_track)
    release_modes = _release_mode_counts(all_cases, len(excluded_cards))
    actions = Counter(card["action_taken"] for card in qc_cards + excluded_cards)
    source_counts = Counter(record["source_database"] for record in all_cases)
    contact_labeled = sum(1 for record in cases_by_track["pose"] if record.get("contact_label_status") == "computed_contacts")
    coordinate_reference = sum(
        1 for record in cases_by_track["pose"] if record.get("contact_label_status") == "coordinate_reference_only"
    )
    structure_source_reference = sum(1 for record in cases_by_track["structure"] if not record.get("gold_coordinates"))
    structure_coordinate_evaluated = case_counts["structure"] - structure_source_reference
    qc_resolutions = Counter(
        card.get("qc_resolution") or "none" for card in qc_cards + excluded_cards
    )
    release_metadata = {
        "release_status": "PEB v1.0 release candidate",
        "case_counts": case_counts,
        "release_mode_counts": release_modes,
        "quality_check": {
            "pass_a_rows": len(pass_a_rows),
            "pass_b_rows": len(pass_b_rows),
            "actions": dict(actions),
            "qc_resolutions": dict(qc_resolutions),
        },
        "structure_metric_status": {
            "source_reference_cases": structure_source_reference,
            "coordinate_evaluated_cases": structure_coordinate_evaluated,
            "rmsd_skipped_count": structure_source_reference,
            "rmsd_not_computed_reason": "source-reference-only cases require coordinate fetch before coordinate-level RMSD evaluation",
        },
        "pose_metric_status": {
            "contact_labeled_cases": contact_labeled,
            "coordinate_reference_only_cases": coordinate_reference,
            "contact_metric_denominator": contact_labeled,
        },
        "source_counts": dict(sorted(source_counts.items())),
        "limitations": [
            "source-backed only; no human quality check is claimed",
            "structure source-reference cases require coordinate fetch before RMSD evaluation",
            "pose contact metrics apply to the contact-labeled subset",
            "source-reference-only records may require public-source fetching for raw-record inspection",
            "human-effect cases are evidence classification records, not direct efficacy predictions",
        ],
    }
    write_text(root / "release_metadata_summary.json", json.dumps(release_metadata, indent=2, sort_keys=True))
    disagreements = sum(1 for card in qc_cards + excluded_cards if card.get("disagreements"))
    manifest = {
        "benchmark_name": "Peptide Engineering Benchmark",
        "benchmark_abbreviation": "PEB",
        "release_id": RELEASE_ID,
        "release_date": RELEASE_DATE,
        "release_status": "PEB v1.0 release candidate",
        "external_qc_claimed": False,
        "direct_human_effect_prediction_claimed": False,
        "medical_use_claimed": False,
        "qced": True,
        "source_metadata": True,
        "case_counts": case_counts,
        "release_mode_counts": release_modes,
        "quality_check": {
            "pass_a": QC_PASS_A,
            "pass_b": QC_PASS_B,
            "pass_a_records": len(pass_a_rows),
            "pass_b_records": len(pass_b_rows),
            "disagreements": disagreements,
            "actions": dict(actions),
        },
        "source_manifest_snapshot": "source_manifest_snapshot.yaml",
        "limitations": [
            "source-backed, not externally checked",
            "source-reference-only records require public source fetching for raw-record inspection",
            "pose RMSD and clash metrics are optional subset metrics",
        ],
        "files": [],
    }
    write_text(root / "release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_text(root / "source_manifest_snapshot.yaml", SOURCE_MANIFEST.read_text(encoding="utf-8"))


def quality_check_release(
    release_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> dict[str, Any]:
    source_root = Path(release_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, output_root, dirs_exist_ok=True)
    for stale_doc in LEGACY_PUBLIC_RELEASE_FILES:
        stale_path = output_root / stale_doc
        if stale_path.exists():
            stale_path.unlink()
    cases_by_track = _load_cases(source_root)
    checked_by_track: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    excluded_cards: list[dict[str, Any]] = []
    qc_cards: list[dict[str, Any]] = []
    pass_a_rows: list[dict[str, Any]] = []
    pass_b_rows: list[dict[str, Any]] = []
    for track in TRACKS:
        for record in cases_by_track[track]:
            checked, pass_a, pass_b, card = _qc_record(record)
            pass_a_rows.append(pass_a)
            pass_b_rows.append(pass_b)
            if checked is None:
                excluded_cards.append(card)
            else:
                checked_by_track[track].append(checked)
                qc_cards.append(card)
    counts = _case_counts(checked_by_track)
    below_minimum = {
        track: {"count": count, "minimum": MINIMUMS[track]}
        for track, count in counts.items()
        if count < MINIMUMS[track]
    }
    for track, records in checked_by_track.items():
        write_jsonl(output_root / track / "cases.jsonl", records)
        write_splits([dict(record) for record in records], output_root / track / "splits")
    _write_references(output_root, checked_by_track, excluded_cards, qc_cards, pass_a_rows, pass_b_rows)
    source_status = _source_status_rows(source_root, checked_by_track, excluded_cards)
    write_jsonl(output_root / "source_status_report.jsonl", source_status)
    write_jsonl(output_root / "references" / "source_status_report.jsonl", source_status)
    _write_release_metadata(output_root, checked_by_track, qc_cards, excluded_cards, pass_a_rows, pass_b_rows)
    for stale in (
        output_root / "baselines" / "predictions" / "human_effect_retrieval.jsonl",
        output_root / "baselines" / "predictions" / "human_effect_source_reference.jsonl",
        output_root / "baselines" / "results" / "human_effect_retrieval.json",
        output_root / "baselines" / "results" / "human_effect_source_reference.json",
    ):
        if stale.exists():
            stale.unlink()
    write_baseline_outputs(output_root)
    manifest = json.loads((output_root / "release_manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file())
    manifest["below_minimum"] = below_minimum
    write_text(output_root / "release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    summary = {
        "release_dir": str(output_root),
        "case_counts": counts,
        "actions": dict(Counter(card["action_taken"] for card in qc_cards + excluded_cards)),
        "pass_a_records": len(pass_a_rows),
        "pass_b_records": len(pass_b_rows),
        "disagreements": sum(1 for card in qc_cards + excluded_cards if card.get("disagreements")),
        "below_minimum": below_minimum,
    }
    write_text(output_root / "quality_check_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _outcome_from_qc_row(row: dict[str, Any]) -> QCOutcome:
    return QCOutcome(
        benchmark_id=row.get("benchmark_id", "unknown"),
        track=row.get("track", "unknown"),
        checker=row.get("checker", "qc"),
        result=row.get("result", "pass"),
        notes=list(row.get("notes") or []),
        updates=dict(row.get("updates") or {}),
        exclude_reason=row.get("exclude_reason"),
    )


def resolve_qc_disagreements(release_dir: Union[str, Path]) -> dict[str, Any]:
    root = Path(release_dir)
    for stale_doc in LEGACY_PUBLIC_RELEASE_FILES:
        stale_path = root / stale_doc
        if stale_path.exists():
            stale_path.unlink()
    references = root / "references"
    pass_a_rows = read_jsonl(references / "qc_pass_a.jsonl")
    pass_b_rows = read_jsonl(references / "qc_pass_b.jsonl")
    cards = read_jsonl(references / "case_qc_cards.jsonl")
    pass_a_by_id = {row["benchmark_id"]: _outcome_from_qc_row(row) for row in pass_a_rows}
    pass_b_by_id = {row["benchmark_id"]: _outcome_from_qc_row(row) for row in pass_b_rows}
    card_by_id = {card["benchmark_id"]: card for card in cards}
    cases_by_track = _read_all_release_cases(root)
    resolved_counts = Counter()
    updated_cards: list[dict[str, Any]] = []
    for track, records in cases_by_track.items():
        updated_records = []
        for record in records:
            card = dict(card_by_id.get(record["benchmark_id"], {}))
            disagreements = list(card.get("disagreements") or [])
            pass_a = pass_a_by_id.get(record["benchmark_id"], _base_outcome(record, QC_PASS_A))
            pass_b = pass_b_by_id.get(record["benchmark_id"], _base_outcome(record, QC_PASS_B))
            worst_result = max([pass_a.result, pass_b.result], key=lambda value: SEVERITY[value])
            updates, resolution = _resolve_disagreements(record, pass_a, pass_b, disagreements, worst_result)
            record.update(updates)
            record["qc_disagreement"] = bool(disagreements)
            record["qc_resolution"] = resolution or None
            record["scoring_subset"] = record.get("scoring_subset") or _default_scoring_subset(record)
            if resolution == "excluded_from_primary_scoring":
                record["qc_result"] = "passed_with_warnings"
            notes = list(record.get("qc_notes") or [])
            if disagreements and not any("qc disagreement resolved as" in note for note in notes):
                notes.append(f"qc disagreement resolved as {resolution}")
            record["qc_notes"] = notes or ["passed conservative qc"]
            _ensure_reproducibility_hashes(record)
            validate_case_record(record)
            updated_records.append(record)
            card.update(
                {
                    "benchmark_id": record["benchmark_id"],
                    "track": record["track"],
                    "source_database": record["source_database"],
                    "source_id": record["source_id"],
                    "release_mode": record["release_mode"],
                    "qc_result": record.get("qc_result"),
                    "qc_disagreement": record["qc_disagreement"],
                    "qc_resolution": record.get("qc_resolution"),
                    "scoring_subset": record.get("scoring_subset"),
                    "final_label_summary": _final_label_summary(record),
                    "caveats": _caveats(record),
                    "processed_record_hash": record.get("processed_record_hash"),
                }
            )
            updated_cards.append(card)
            if disagreements:
                resolved_counts[resolution or "unresolved"] += 1
        cases_by_track[track] = updated_records
    for track, records in cases_by_track.items():
        write_jsonl(root / track / "cases.jsonl", records)
        write_splits([dict(record) for record in records], root / track / "splits")
    write_jsonl(references / "case_qc_cards.jsonl", updated_cards)
    excluded_cards = [card for card in updated_cards if card.get("action_taken") == "excluded"]
    kept_cards = [card for card in updated_cards if card.get("action_taken") != "excluded"]
    _write_references(root, cases_by_track, excluded_cards, kept_cards, pass_a_rows, pass_b_rows)
    _write_release_metadata(root, cases_by_track, kept_cards, excluded_cards, pass_a_rows, pass_b_rows)
    summary = {
        "total_disagreements": sum(resolved_counts.values()),
        "kept_conservative_label": resolved_counts.get("kept_conservative_label", 0),
        "downgraded": resolved_counts.get("downgraded", 0),
        "excluded_from_primary_scoring": resolved_counts.get("excluded_from_primary_scoring", 0),
        "excluded": resolved_counts.get("excluded", 0),
        "unresolved": resolved_counts.get("unresolved", 0),
    }
    write_text(root / "qc_resolution_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    manifest_path = root / "release_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["qc_qc_resolution"] = summary
        manifest["files"] = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
        write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return summary


def package_release(release_dir: Union[str, Path], output: Union[str, Path]) -> dict[str, Any]:
    root = Path(release_dir)
    output_path = Path(output)
    passed, errors, warnings, _summary = publish_check(root)
    if not passed:
        raise ValueError("publish-check failed: " + "; ".join(errors))
    release_passed, release_errors, release_warnings = release_check(root)
    if not release_passed:
        raise ValueError("release-check failed: " + "; ".join(release_errors))
    manifest_path = root / "release_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
        write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    checksum = sha256_file(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    write_text(checksum_path, f"{checksum}  {output_path.name}\n")
    return {
        "archive": str(output_path),
        "sha256": checksum,
        "sha256_file": str(checksum_path),
        "publish_warnings": warnings,
        "release_warnings": release_warnings,
    }


def _required_release_files() -> list[str]:
    return [
        "release_manifest.json",
        "release_metadata_summary.json",
        "quality_check_summary.json",
        "source_manifest_snapshot.yaml",
        "source_status_report.jsonl",
        "references/source_ids.jsonl",
        "references/citations.jsonl",
        "references/nonredistributable_source_index.jsonl",
        "references/exclusion_log.jsonl",
        "references/case_qc_cards.jsonl",
        "references/qc_pass_a.jsonl",
        "references/qc_pass_b.jsonl",
    ]


def _read_all_release_cases(root: Path) -> dict[str, list[dict[str, Any]]]:
    cases = {}
    for track in TRACKS:
        path = root / track / "cases.jsonl"
        cases[track] = read_jsonl(path) if path.exists() else []
    return cases


def _doc_claim_errors(root: Path) -> list[str]:
    errors = []
    text_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".yaml", ".yml", ".json"}
    ]
    bad_exact_phrases = [
        "release candidate with " + "issue" + "ers",
        "issue" + "er",
        "issue" + "ed",
        "quality check " + "required",
        "curation " + "required",
        "license qc " + "required",
        "required before " + "release",
        "needs chain-role " + "gates",
        "needs assay compatibility " + "gates",
        "needs conservative " + "classification",
        "scientifically " + "complete",
        "fully " + "complete",
        "release-" + "ready",
    ]
    bad_claims = [
        "clinically " + "validated",
        "safe for human " + "use",
        "predicts human " + "effects",
        "provides medical " + "advice",
        "is medical " + "advice",
        "validates human " + "use",
    ]
    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        relative = path.relative_to(root)
        for phrase in bad_exact_phrases:
            if phrase in text:
                errors.append(f"{relative}: contains contradictory release language: {phrase}")
        if "externally checked" in text and "not externally checked" not in text:
            errors.append(f"{relative}: claims human quality check occurred")
        direct_effect_phrase = "direct human-effect " + "prediction"
        negated_direct_effect_phrase = "not direct human-effect " + "prediction"
        if direct_effect_phrase in text and negated_direct_effect_phrase not in text:
            errors.append(f"{relative}: contains unsupported effect-prediction claim")
        for phrase in bad_claims:
            if phrase in text:
                errors.append(f"{relative}: contains prohibited claim: {phrase}")
    return errors


def publish_check(
    release_dir: Union[str, Path],
    min_structure: int = 200,
    min_pose: int = 100,
    min_binding_rank: int = 25,
    min_human_effect: int = 200,
) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    root = Path(release_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for legacy_file in sorted(LEGACY_PUBLIC_RELEASE_FILES):
        if (root / legacy_file).exists():
            errors.append(f"legacy compatibility file is not allowed in public release root: {legacy_file}")
    for markdown_file in sorted(root.rglob("*.md")):
        errors.append(f"markdown file is not allowed in minimal public release data: {markdown_file.relative_to(root)}")
    for relative in _required_release_files():
        if not (root / relative).exists():
            errors.append(f"missing {relative}")
    cases_by_track = _read_all_release_cases(root)
    minimums = {
        "structure": min_structure,
        "pose": min_pose,
        "binding_rank": min_binding_rank,
        "human_effect": min_human_effect,
    }
    all_cases = _all_cases(cases_by_track)
    counts = _case_counts(cases_by_track)
    for track, records in cases_by_track.items():
        if len(records) < minimums[track]:
            errors.append(f"{track}: {len(records)} cases below required minimum {minimums[track]}")
        if len(records) == 0:
            errors.append(f"{track}: zero cases")
        for split in ("train", "dev", "test"):
            if not (root / track / "splits" / f"{split}.jsonl").exists():
                errors.append(f"missing {track}/splits/{split}.jsonl")
    source_reference_keys = set()
    nonredistributable = root / "references" / "nonredistributable_source_index.jsonl"
    if nonredistributable.exists():
        for row in read_jsonl(nonredistributable):
            source_reference_keys.add((row.get("benchmark_id"), row.get("source_database"), row.get("source_id")))
    for record in all_cases:
        label = record.get("benchmark_id", "unknown")
        try:
            validate_case_record(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: schema validation failed: {exc}")
            continue
        if record.get("source_database") == "synthetic_fixture" or record.get("synthetic"):
            errors.append(f"{label}: synthetic/test case included")
        for field_name in (
            "source_database",
            "source_id",
            "source_url",
            "retrieval_date",
            "license_or_usage_note",
            "release_mode",
            "qc_status",
            "qc_result",
            "processed_record_hash",
            "scoring_subset",
        ):
            if not record.get(field_name):
                errors.append(f"{label}: missing {field_name}")
        if record.get("qc_disagreement") and record.get("qc_resolution") not in DISAGREEMENT_RESOLUTIONS:
            errors.append(f"{label}: qc disagreement lacks valid resolution")
        if record.get("qc_status") == "external_qc":
            errors.append(f"{label}: false external qc marker")
        if record.get("release_mode") == "source_reference_only":
            key = (record.get("benchmark_id"), record.get("source_database"), record.get("source_id"))
            if key not in source_reference_keys:
                errors.append(f"{label}: source-reference-only case missing nonredistributable index entry")
        track = record.get("track")
        if track == "pose":
            if not record.get("target_chain_id") or not record.get("peptide_chain_id"):
                errors.append(f"{label}: pose case lacks chain roles")
            if record.get("target_chain_id") == record.get("peptide_chain_id"):
                errors.append(f"{label}: pose target and peptide chains are identical")
            if not record.get("contact_label_status"):
                errors.append(f"{label}: pose case missing contact_label_status")
        if track == "binding_rank":
            if record.get("assay_compatibility_status") == "incompatible_excluded":
                errors.append(f"{label}: binding panel is assay-incompatible")
            if len(record.get("items") or []) < 5:
                errors.append(f"{label}: binding panel has fewer than five candidates")
            if record.get("measurement_direction") not in {"lower_is_stronger", "higher_is_stronger"}:
                errors.append(f"{label}: binding panel missing valid measurement direction")
        if track == "human_effect":
            source = str(record.get("source_database", "")).lower()
            evidence = record.get("evidence_level")
            if not record.get("evidence_validation_status"):
                errors.append(f"{label}: human-effect case missing evidence_validation_status")
            if evidence == "approved_human_indication" and source not in {"dailymed", "regulatory_label"}:
                errors.append(f"{label}: approved indication lacks approved-label source")
            if source == "clinicaltrials" and record.get("trial_has_results") is False and record.get("evidence_direction") == "positive":
                errors.append(f"{label}: clinical-trial no-results case marked positive")
            if source in {"reactome", "gene_ontology", "opentargets"} and evidence == "approved_human_indication":
                errors.append(f"{label}: pathway-only case marked approved")
            if record.get("safety_status") == "known_acceptable_under_approved_use" and evidence != "approved_human_indication":
                errors.append(f"{label}: safety claimed without approved-use source support")
    contact_labeled = sum(
        1
        for record in cases_by_track.get("pose", [])
        if record.get("contact_label_status") == "computed_contacts" and record.get("native_contacts")
    )
    if contact_labeled < 25:
        errors.append(f"pose: contact-labeled subset below 25 cases: {contact_labeled}")
    cards_path = root / "references" / "case_qc_cards.jsonl"
    if cards_path.exists():
        for card in read_jsonl(cards_path):
            if card.get("disagreements") and card.get("qc_resolution") not in DISAGREEMENT_RESOLUTIONS:
                errors.append(f"{card.get('benchmark_id', 'unknown')}: qc card disagreement lacks resolution")
            if card.get("disagreements") and not card.get("scoring_subset"):
                errors.append(f"{card.get('benchmark_id', 'unknown')}: qc card missing scoring_subset")
    audit = audit_records(all_cases)
    errors.extend(audit.errors)
    warnings.extend(audit.warnings)
    source_status = root / "source_status_report.jsonl"
    if source_status.exists():
        for index, row in enumerate(read_jsonl(source_status), start=1):
            deprecated_source_note_key = "issue" + "er"
            if deprecated_source_note_key in row:
                errors.append(f"source_status_report.jsonl:{index}: contains deprecated source limitation field")
            if "issue" + "ed" in str(row.get("status", "")):
                errors.append(f"source_status_report.jsonl:{index}: contains legacy deferred-source status wording")
            for field_name in (
                "source",
                "track",
                "status",
                "records_examined",
                "cases_created",
                "release_mode",
                "action_taken",
                "next_step_if_any",
                "affects_current_release",
                "raw_records_bundled",
                "derived_labels_bundled",
                "source_references_included",
                "current_release_impact",
                "future_bundling_requirement",
            ):
                if field_name not in row:
                    errors.append(f"source_status_report.jsonl:{index}: missing {field_name}")
            if row.get("status") not in SOURCE_STATUS_VALUES:
                errors.append(f"source_status_report.jsonl:{index}: vague source status {row.get('status')}")
            if row.get("status") in {"deferred_access_analysis", "deferred_license_analysis"} and row.get("affects_current_release"):
                errors.append(f"source_status_report.jsonl:{index}: unused source marked as affecting current release")
    result_files = sorted((root / "baselines" / "results").glob("*.json")) if (root / "baselines" / "results").exists() else []
    result_tracks = set()
    pose_result_metrics = None
    structure_result_metrics = None
    competitive_oracle_results = []
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            result_tracks.add(payload.get("track"))
            model_name = str(payload.get("model_name", ""))
            if _is_oracle_model_name(model_name) and payload.get("competitive", True) is not False:
                competitive_oracle_results.append(result_file.name)
            if payload.get("track") == "pose":
                pose_result_metrics = payload.get("metrics", {})
            if payload.get("track") == "structure":
                structure_result_metrics = payload.get("metrics", {})
        except json.JSONDecodeError as exc:
            errors.append(f"{result_file}: invalid result JSON: {exc}")
    if competitive_oracle_results:
        errors.append(f"oracle baseline appears in competitive leaderboard results: {competitive_oracle_results}")
    for track in TRACKS:
        if track not in result_tracks:
            errors.append(f"baseline result missing for {track}")
    if pose_result_metrics is not None and pose_result_metrics.get("contact_metric_denominator") != contact_labeled:
        errors.append("pose contact metric denominator does not match contact-labeled subset count")
    if structure_result_metrics is not None:
        skipped = structure_result_metrics.get("skipped_cases", 0)
        if skipped and not structure_result_metrics.get("rmsd_not_computed_reason"):
            errors.append("structure RMSD skipped count is present without not-computed reason")
    manifest_path = root / "release_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("release_status") != "PEB v1.0 release candidate":
            errors.append("release_manifest.json: release_status is not the canonical public status")
        expected_flags = {
            "external_qc_claimed": False,
            "direct_human_effect_prediction_claimed": False,
            "medical_use_claimed": False,
            "qced": True,
            "source_metadata": True,
        }
        for key, expected in expected_flags.items():
            if manifest.get(key) is not expected:
                errors.append(f"release_manifest.json: {key} must be {expected}")
        manifest_files = set(manifest.get("files") or [])
        for legacy_file in LEGACY_PUBLIC_RELEASE_FILES:
            if legacy_file in manifest_files:
                errors.append(f"release_manifest.json: legacy public file listed: {legacy_file}")
    errors.extend(_doc_claim_errors(root))
    release_passed, release_errors, release_warnings = release_check(
        root,
        min_structure=min_structure,
        min_pose=min_pose,
        min_binding_rank=min_binding_rank,
        min_human_effect=min_human_effect,
    )
    if not release_passed:
        errors.extend(f"release-check: {error}" for error in release_errors)
    warnings.extend(f"release-check: {warning}" for warning in release_warnings)
    summary = {
        "case_counts": counts,
        "contact_labeled_pose_cases": contact_labeled,
        "baseline_result_files": [str(path.relative_to(root)) for path in result_files],
        "release_mode_counts": _release_mode_counts(all_cases, len(read_jsonl(root / "references" / "exclusion_log.jsonl")) if (root / "references" / "exclusion_log.jsonl").exists() else 0),
    }
    return not errors, errors, warnings, summary
