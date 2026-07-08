"""Release directory checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from peb.io import read_jsonl
from peb.processing.audit import audit_records

TRACKS = ("structure", "pose", "binding_rank", "human_effect")
SPLITS = ("train", "dev", "test")


def release_check(
    input_dir: Union[str, Path],
    min_structure: int = 200,
    min_pose: int = 100,
    min_binding_rank: int = 25,
    min_human_effect: int = 200,
) -> tuple[bool, list[str], list[str]]:
    root = Path(input_dir)
    errors: list[str] = []
    warnings: list[str] = []
    required_root = [
        "release_manifest.json",
        "release_metadata_summary.json",
        "quality_check_summary.json",
        "source_manifest_snapshot.yaml",
        "source_status_report.jsonl",
        "references/source_ids.jsonl",
        "references/citations.jsonl",
        "references/nonredistributable_source_index.jsonl",
        "references/exclusion_log.jsonl",
    ]
    for markdown_file in sorted(root.rglob("*.md")):
        errors.append(f"markdown file is not allowed in minimal release data: {markdown_file.relative_to(root)}")
    for relative in required_root:
        if not (root / relative).exists():
            errors.append(f"missing {relative}")
    minimums = {
        "structure": min_structure,
        "pose": min_pose,
        "binding_rank": min_binding_rank,
        "human_effect": min_human_effect,
    }
    for track in TRACKS:
        cases_path = root / track / "cases.jsonl"
        if not cases_path.exists():
            errors.append(f"missing {track}/cases.jsonl")
            continue
        records = read_jsonl(cases_path)
        if len(records) < minimums[track]:
            errors.append(f"{track}: {len(records)} cases below required minimum {minimums[track]}")
        if records and all(record.get("source_database") == "synthetic_fixture" for record in records):
            errors.append(f"{track}: all cases are synthetic fixtures")
        if track == "pose":
            contact_count = sum(1 for record in records if record.get("native_contacts"))
            if contact_count < 25:
                errors.append(f"pose: {contact_count} cases with contact maps below required minimum 25")
        for record in records:
            label = record.get("benchmark_id", "unknown")
            if not record.get("release_mode"):
                errors.append(f"{track}: {label}: missing release_mode")
            if not record.get("source_id"):
                errors.append(f"{track}: {label}: missing source_id")
            if not record.get("retrieval_date"):
                errors.append(f"{track}: {label}: missing retrieval_date")
            if not record.get("license_or_usage_note"):
                errors.append(f"{track}: {label}: missing license note")
        audit = audit_records(records)
        errors.extend(f"{track}: {error}" for error in audit.errors)
        warnings.extend(f"{track}: {warning}" for warning in audit.warnings)
        for split in SPLITS:
            if not (root / track / "splits" / f"{split}.jsonl").exists():
                errors.append(f"missing {track}/splits/{split}.jsonl")
    manifest_path = root / "release_manifest.json"
    if manifest_path.exists():
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"release_manifest.json invalid: {exc}")
    source_snapshot = root / "source_manifest_snapshot.yaml"
    if source_snapshot.exists():
        try:
            yaml.safe_load(source_snapshot.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            errors.append(f"source_manifest_snapshot.yaml invalid: {exc}")
    return not errors, errors, warnings
