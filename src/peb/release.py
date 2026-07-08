"""Release-directory generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Union

from pydantic import TypeAdapter

from peb.io import read_jsonl, write_jsonl, write_text
from peb.processing.release_check import TRACKS
from peb.processing.splits import write_splits
from peb.registry import SOURCE_MANIFEST, validate_source_manifest
from peb.reports.benchmark_card import render_benchmark_card
from peb.reports.datasheet import render_datasheet
from peb.reports.leaderboard import render_leaderboard
from peb.reports.release_report import render_release_report
from peb.schemas import Case, Prediction, ReleaseManifest

RELEASE_ID = "peb-v1.0-rc"
RELEASE_DATE = "2026-07-07"


def write_json_schemas(output_dir: Union[str, Path]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_schema = TypeAdapter(Case).json_schema()
    prediction_schema = TypeAdapter(Prediction).json_schema()
    write_text(root / "benchmark_case.schema.json", json.dumps(case_schema, indent=2, sort_keys=True))
    write_text(root / "prediction.schema.json", json.dumps(prediction_schema, indent=2, sort_keys=True))


def _empty_release_cases() -> dict[str, list[dict[str, Any]]]:
    return {track: [] for track in TRACKS}


def _all_release_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _reference_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = []
    citations = []
    restricted = []
    for entry in validate_source_manifest():
        source_ids.append(
            {
                "source_database": entry.name,
                "url": str(entry.url),
                "source_bucket": entry.source_bucket.value,
                "adapter_status": entry.adapter_status.value,
            }
        )
        citations.append({"source_database": entry.name, "citation": entry.citation})
        if entry.redistribution_policy.raw_data_redistribution.value != "allowed":
            restricted.append(
                {
                    "source_database": entry.name,
                    "source_bucket": entry.source_bucket.value,
                    "reason": "raw redistribution not cleared for this release candidate",
                    "safe_release_form": "source IDs, citations, and derived labels after source-specific qc",
                }
            )
    return source_ids, citations, restricted


def _release_readme(case_counts: dict[str, int]) -> str:
    return f"""# Peptide Engineering Benchmark {RELEASE_ID}

Status: release candidate with source-backed metadata.

This generated release directory contains the public-release structure, source manifest snapshot, governance reports, and audit outputs for PEB v1.0-rc.

PEB evaluates evidence-grounded peptide engineering judgment. It does not claim to predict human effects directly, and it does not validate any peptide for human use.

## Case Counts
- structure: {case_counts["structure"]}
- pose: {case_counts["pose"]}
- binding_rank: {case_counts["binding_rank"]}
- human_effect: {case_counts["human_effect"]}

The current release directory intentionally contains no synthetic cases as benchmark data. Synthetic fixtures remain under `data/fixtures` for tests and CLI smoke runs.

## Limitations
Full source-backed data requires explicit source fetching, source governance for cautious sources, and automated conservative qc before cases can be marked publishable.
"""


def _governance_report() -> str:
    return """# Data Governance Report

PEB keeps raw-source fetching explicit. Bucket A sources may be used when attribution, source identifiers, retrieval dates, and source versions are retained. Bucket B and C sources must not have raw records bundled unless a license qc clears redistribution.

Human-effect evidence cases must separate evidence direction, claim status, and safety status. Clinical trial records without results cannot be used as positive evidence. Pathway or similarity context cannot be promoted to approved human indication evidence.

The release candidate records remaining limitations instead of presenting synthetic fixtures as real source-backed benchmark data.
"""


def _limitations() -> list[str]:
    return [
        "Scaled source-backed structure and pose case construction requires approved RCSB fetches and chain-role validation.",
        "Binding-rank panels require IEDB or comparable assay exports plus license qc and assay-compatibility validation.",
        "Human-effect evidence cases require DailyMed and ClinicalTrials source validation plus conservative evidence classification.",
        "Geometry scoring beyond contact labels requires curated structures, atom mappings, and validated RMSD/interface hooks.",
    ]


def build_release(output_dir: Union[str, Path]) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json_schemas(Path("schemas"))
    existing_case_files = {
        track: root / track / "cases.jsonl"
        for track in TRACKS
    }
    if all(path.exists() for path in existing_case_files.values()) and any(
        read_jsonl(path) for path in existing_case_files.values()
    ):
        from peb.public_build import (
            _counts_by_source,
            source_status_rows,
            write_baseline_outputs,
            write_release_docs,
            write_release_references,
        )

        for track, path in existing_case_files.items():
            write_splits(read_jsonl(path), root / track / "splits")
        write_release_references(root, existing_case_files)
        source_status_rows(root, _counts_by_source(existing_case_files))
        write_release_docs(root, existing_case_files)
        write_baseline_outputs(root)
        write_text(root / "release_check_report.md", render_release_report(root))
        return root

    cases_by_track = _empty_release_cases()
    case_counts = {track: len(records) for track, records in cases_by_track.items()}

    for track, records in cases_by_track.items():
        track_dir = root / track
        write_jsonl(track_dir / "cases.jsonl", records)
        write_splits([dict(record) for record in records], track_dir / "splits")

    references = root / "references"
    references.mkdir(parents=True, exist_ok=True)
    source_ids, citations, restricted = _reference_records()
    write_jsonl(references / "source_ids.jsonl", source_ids)
    write_jsonl(references / "citations.jsonl", citations)
    write_jsonl(references / "nonredistributable_source_index.jsonl", restricted)

    (root / "baselines" / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "baselines" / "results").mkdir(parents=True, exist_ok=True)
    write_text(root / "baselines" / "predictions" / ".gitkeep", "")
    write_text(root / "baselines" / "results" / ".gitkeep", "")

    shutil.copyfile(SOURCE_MANIFEST, root / "source_manifest_snapshot.yaml")
    write_text(root / "README.md", _release_readme(case_counts))
    write_text(root / "data_governance_report.md", _governance_report())
    write_text(root / "benchmark_card.md", render_benchmark_card(case_counts, "release candidate with source-backed metadata"))
    all_cases: list[dict[str, Any]] = []
    for records in cases_by_track.values():
        all_cases.extend(records)
    write_text(root / "datasheet.md", render_datasheet(all_cases, "PEB v1.0-rc Datasheet"))
    write_text(root / "leaderboard_baselines.md", render_leaderboard(root / "baselines" / "results"))

    manifest = ReleaseManifest(
        benchmark_name="Peptide Engineering Benchmark",
        benchmark_abbreviation="PEB",
        release_id=RELEASE_ID,
        release_date=RELEASE_DATE,
        release_status="release_candidate_with_source_metadata",
        case_counts=case_counts,
        source_manifest_snapshot="source_manifest_snapshot.yaml",
        limitations=_limitations(),
        files=[],
    )
    write_text(
        root / "release_manifest.json",
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True),
    )
    manifest_record = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    manifest_record["files"] = _all_release_files(root)
    write_text(root / "release_manifest.json", json.dumps(manifest_record, indent=2, sort_keys=True))
    write_text(root / "release_check_report.md", render_release_report(root))
    return root


def fixture_counts() -> dict[str, int]:
    fixture_root = Path("data/fixtures")
    return {
        "structure": len(read_jsonl(fixture_root / "structure_fixture.jsonl")),
        "pose": len(read_jsonl(fixture_root / "pose_fixture.jsonl")),
        "binding_rank": len(read_jsonl(fixture_root / "binding_rank_fixture.jsonl")),
        "human_effect": len(read_jsonl(fixture_root / "human_effect_fixture.jsonl")),
    }


def write_root_limitations_docs() -> None:
    write_text(
        "REMAINING_LIMITATIONS.md",
        """# Remaining Limitations

The repository builds a source-backed release candidate with source-backed metadata.

## Source Governance
- Bucket B and C sources use source-reference or derived records unless future governance analysis supports raw-record bundling.
- IEDB-derived binding-rank panels retain source IDs and compact assay labels.

## Network and Source Access
- Real source-backed cases require explicit fetch commands against RCSB, DailyMed, ClinicalTrials.gov, and assay sources.
- The code provides fetch commands, but large source retrievals were not bundled into this candidate.

## Automated Scientific QC
- Future scientific hardening can further improve chain-role validation, leakage grouping, and quality filtering.
- Future source expansion can broaden assay comparability.
- Source-backed labels are conservative but not human-externally checked.

## Geometry and Scoring
- Interface RMSD, clash scoring, and confidence calibration are subset/future metrics until coordinate and confidence coverage expands.
""",
    )
    write_text(
        "FUTURE_SCIENTIFIC_HARDENING.md",
        """# Future Scientific Hardening

PEB source-backed cases pass automated conservative qc before public leaderboard use.

Future scientific hardening can add broader source imports, additional contact labels, more coordinate-evaluated structure subsets, and another qc layer.
""",
    )
    write_text(
        "SOURCE_GOVERNANCE_NOTES.md",
        """# Source Governance Notes

Bucket A sources can generally be used with attribution and source identifiers. Bucket B and C sources use source-reference or derived records unless future source-specific governance analysis supports raw-record bundling.

Restricted or unclear records should be represented through source identifiers, citations, and `nonredistributable_source_index.jsonl`.
""",
    )
