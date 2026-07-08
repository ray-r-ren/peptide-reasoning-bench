"""CSV curation templates and parsers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Union

from peb.io import sha256_text

COMMON_FIELDS = [
    "benchmark_id",
    "source_database",
    "source_id",
    "source_url",
    "source_version",
    "retrieval_date",
    "license_or_usage_note",
    "citation",
    "split",
    "qc_status",
    "curator_notes",
]

TEMPLATES = {
    "human_effect_cases_template.csv": COMMON_FIELDS
    + [
        "peptide_sequence",
        "claim_text",
        "category",
        "evidence_level",
        "evidence_direction",
        "claim_status",
        "safety_status",
        "source_evidence_type",
        "trial_status",
        "trial_phase",
        "trial_has_results",
    ],
    "pdb_structure_cases_template.csv": COMMON_FIELDS
    + [
        "peptide_sequence",
        "structure_id",
        "experimental_method",
        "resolution_angstrom",
    ],
    "pdb_pose_cases_template.csv": COMMON_FIELDS
    + [
        "peptide_sequence",
        "target_name",
        "target_chain_id",
        "peptide_chain_id",
        "pdb_id",
        "native_contacts_json",
        "binding_site_residues",
    ],
    "mhc_bindrank_panels_template.csv": COMMON_FIELDS
    + [
        "panel_id",
        "assay_type",
        "assay_unit",
        "assay_conditions",
        "measurement_direction",
        "normalization_method",
        "comparable_panel",
        "panel_exclusion_reason",
        "items_json",
    ],
    "source_license_qc_template.csv": [
        "name",
        "url",
        "source_bucket",
        "raw_data_redistribution",
        "processed_label_redistribution",
        "commercial_use",
        "attribution_required",
        "qc_notes",
    ],
    "exclusion_log_template.csv": [
        "source_database",
        "source_id",
        "track",
        "exclusion_reason",
        "qc_status",
    ],
}


def create_templates(output_dir: Union[str, Path]) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, fields in TEMPLATES.items():
        path = root / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
        paths.append(path)
    return paths


def read_csv_rows(path: Union[str, Path]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def base_case_fields(row: dict[str, str], track: str) -> dict[str, Any]:
    return {
        "benchmark_id": row["benchmark_id"],
        "track": track,
        "release_mode": row.get("release_mode") or "source_reference_only",
        "source_database": row["source_database"],
        "source_id": row["source_id"],
        "source_url": row.get("source_url") or None,
        "source_version": row["source_version"],
        "retrieval_date": row["retrieval_date"],
        "license_or_usage_note": row["license_or_usage_note"],
        "redistribution_policy_snapshot": {
            "raw_data_redistribution": "restricted",
            "processed_label_redistribution": "allowed",
            "commercial_use": "unknown",
            "attribution_required": True,
            "share_alike_obligation": False,
            "use_in_public_leaderboard": "caution",
        },
        "curator_notes": row.get("curator_notes", ""),
        "qc_status": row.get("qc_status") or "source_checked",
        "citation": row.get("citation") or None,
        "split": row.get("split") or "train",
        "leakage_group": {"peptide_cluster": row.get("peptide_sequence") or None},
        "processed_record_hash": sha256_text(str(sorted(row.items()))),
    }
