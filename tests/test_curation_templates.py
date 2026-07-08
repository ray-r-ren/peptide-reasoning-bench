import csv
import json
import subprocess
import sys

from peb.processing.curation_templates import create_templates


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "peb.cli", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_curation_templates_created(tmp_path):
    paths = create_templates(tmp_path)
    assert len(paths) == 6
    assert (tmp_path / "human_effect_cases_template.csv").exists()


def test_prepare_commands_on_synthetic_rows(tmp_path):
    human_csv = tmp_path / "human.csv"
    with human_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "benchmark_id": "CSV-HFX-1",
                "source_database": "synthetic_fixture",
                "source_id": "CSV-HFX-1",
                "source_url": "https://example.org/peb-fixture",
                "source_version": "fixture-v1",
                "retrieval_date": "2026-07-07",
                "license_or_usage_note": "Synthetic fixture row.",
                "citation": "PEB synthetic fixture protocol.",
                "split": "train",
                "qc_status": "source_checked",
                "curator_notes": "synthetic",
                "peptide_sequence": "AAAA",
                "claim_text": "Synthetic row for evidence classification.",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "unsupported_contradicted_or_unsafe_claim",
                "evidence_direction": "not_applicable",
                "claim_status": "insufficient_information",
                "safety_status": "insufficient_safety_data",
                "source_evidence_type": "synthetic_fixture",
            }
        )
    human_out = tmp_path / "human.jsonl"
    _run_cli("prepare-human-effect-cases", "--input", str(human_csv), "--output", str(human_out))
    assert human_out.exists()

    bind_csv = tmp_path / "bind.csv"
    with bind_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
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
            "panel_id",
            "assay_type",
            "assay_unit",
            "assay_conditions",
            "measurement_direction",
            "normalization_method",
            "comparable_panel",
            "panel_exclusion_reason",
            "items_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "benchmark_id": "CSV-BIND-1",
                "source_database": "synthetic_fixture",
                "source_id": "CSV-BIND-1",
                "source_url": "https://example.org/peb-fixture",
                "source_version": "fixture-v1",
                "retrieval_date": "2026-07-07",
                "license_or_usage_note": "Synthetic fixture row.",
                "citation": "PEB synthetic fixture protocol.",
                "split": "train",
                "qc_status": "source_checked",
                "curator_notes": "synthetic",
                "panel_id": "CSV-PANEL",
                "assay_type": "synthetic_binding_signal",
                "assay_unit": "relative_unit",
                "assay_conditions": "single synthetic condition",
                "measurement_direction": "higher_is_stronger",
                "normalization_method": "rank scaled within panel",
                "comparable_panel": "true",
                "items_json": json.dumps(
                    [
                        {
                            "item_id": "a",
                            "peptide": {"sequence": "AAAA", "modifications": []},
                            "measured_value": 1.0,
                            "normalized_rank": 1.0,
                        },
                        {
                            "item_id": "b",
                            "peptide": {"sequence": "CCCC", "modifications": []},
                            "measured_value": 0.5,
                            "normalized_rank": 0.0,
                        },
                    ]
                ),
            }
        )
    bind_out = tmp_path / "bind.jsonl"
    _run_cli("prepare-bindrank-panels", "--input", str(bind_csv), "--output", str(bind_out))
    assert bind_out.exists()

