"""Command-line interface for PEB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from peb.baselines import make_baseline_predictions
from peb.io import read_jsonl, write_jsonl, write_text
from peb.metrics import (
    evaluate_binding_rank,
    evaluate_human_effect,
    evaluate_pose,
    evaluate_structure,
)
from peb.processing.audit import audit_jsonl
from peb.processing.curation_templates import base_case_fields, create_templates, read_csv_rows
from peb.processing.pdb_contacts import compute_contacts
from peb.processing.release_check import release_check
from peb.processing.splits import write_splits
from peb.processing.validate import validate_records
from peb.public_build import (
    build_all_public_sources,
    source_status_rows,
)
from peb.public_build import (
    build_bindrank_panels as build_public_bindrank_panels,
)
from peb.public_build import (
    build_human_effect_cases as build_public_human_effect_cases,
)
from peb.public_build import (
    build_pose_cases as build_public_pose_cases,
)
from peb.public_build import (
    build_structure_cases as build_public_structure_cases,
)
from peb.quality_check import (
    package_release,
    publish_check,
    quality_check_release,
    resolve_qc_disagreements,
)
from peb.registry import source_by_name, validate_source_manifest
from peb.release import build_release
from peb.reports.datasheet import render_datasheet
from peb.reports.leaderboard import render_leaderboard
from peb.reports.release_report import render_release_report
from peb.reports.source_card import render_source_card
from peb.schemas import (
    BindingRankCase,
    BindingRankPrediction,
    HumanEffectCase,
    HumanEffectPrediction,
    PoseCase,
    PosePrediction,
    StructureCase,
    StructurePrediction,
    Track,
    validate_case_record,
    validate_prediction_record,
)
from peb.sources.clinicaltrials import ClinicalTrialsAdapter
from peb.sources.dailymed import DailyMedAdapter
from peb.sources.rcsb import RCSBAdapter

app = typer.Typer(help="Peptide Engineering Benchmark tooling.")


def _track(value: str) -> Track:
    try:
        return Track(value)
    except ValueError as exc:
        raise typer.BadParameter(f"unknown track: {value}") from exc


def _load_cases(path: Path, track: Track):
    cases = [validate_case_record(record) for record in read_jsonl(path)]
    wrong = [case.benchmark_id for case in cases if case.track != track]
    if wrong:
        raise typer.BadParameter(f"records do not match track {track.value}: {wrong}")
    return cases


def _load_predictions(path: Path, track: Track):
    predictions = [validate_prediction_record(record) for record in read_jsonl(path)]
    wrong = [prediction.benchmark_id for prediction in predictions if prediction.track != track]
    if wrong:
        raise typer.BadParameter(f"predictions do not match track {track.value}: {wrong}")
    return predictions


def _is_oracle_model_name(value: str) -> bool:
    return "oracle" in value and "non_oracle" not in value


@app.command()
def validate(jsonl: Path) -> None:
    """Validate case or prediction JSONL."""
    count, errors = validate_records(jsonl)
    if errors:
        for error in errors:
            typer.echo(error, err=True)
        raise typer.Exit(1)
    typer.echo(f"Validated {count} records")


@app.command("eval")
def eval_command(
    track: str = typer.Option(...),
    gold: Path = typer.Option(...),
    pred: Path = typer.Option(...),
) -> None:
    """Evaluate predictions against gold cases."""
    track_value = _track(track)
    cases = _load_cases(gold, track_value)
    predictions = _load_predictions(pred, track_value)
    if track_value == Track.structure:
        result = evaluate_structure(
            [case for case in cases if isinstance(case, StructureCase)],
            [item for item in predictions if isinstance(item, StructurePrediction)],
        )
    elif track_value == Track.pose:
        result = evaluate_pose(
            [case for case in cases if isinstance(case, PoseCase)],
            [item for item in predictions if isinstance(item, PosePrediction)],
        )
    elif track_value == Track.binding_rank:
        result = evaluate_binding_rank(
            [case for case in cases if isinstance(case, BindingRankCase)],
            [item for item in predictions if isinstance(item, BindingRankPrediction)],
        )
    else:
        result = evaluate_human_effect(
            [case for case in cases if isinstance(case, HumanEffectCase)],
            [item for item in predictions if isinstance(item, HumanEffectPrediction)],
        )
    payload = result.model_dump(mode="json")
    model_name = predictions[0].model_name if predictions else "unspecified"
    payload["model_name"] = model_name
    if _is_oracle_model_name(model_name):
        payload["leaderboard_group"] = "oracle_sanity_check"
        payload["competitive"] = False
        payload["oracle_baseline"] = True
    else:
        payload["leaderboard_group"] = "competitive"
        payload["competitive"] = True
        payload["oracle_baseline"] = False
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    pred_parts = pred.parts
    if "baselines" in pred_parts and "predictions" in pred_parts:
        release_root = pred.parent
        while release_root.name != "baselines" and release_root != release_root.parent:
            release_root = release_root.parent
        result_dir = release_root / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        output_name = pred.stem + ".json"
        write_text(result_dir / output_name, json.dumps(payload, indent=2, sort_keys=True))


@app.command("make-baseline")
def make_baseline(
    track: str = typer.Option(...),
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
    seed: int = typer.Option(13),
    model_name: Optional[str] = typer.Option(None),
) -> None:
    """Create deterministic weak baseline predictions."""
    track_value = _track(track)
    cases = _load_cases(input, track_value)
    selected_model = model_name or "weak_seeded_baseline"
    if track_value == Track.human_effect:
        selected_model = model_name or (
            "human_effect_oracle_source_reference_baseline"
            if _is_oracle_model_name(output.stem)
            else "human_effect_non_oracle_baseline"
        )
    write_jsonl(output, make_baseline_predictions(track_value, cases, seed=seed, model_name=selected_model))
    typer.echo(f"Wrote {output}")


@app.command("list-sources")
def list_sources() -> None:
    """List registered sources."""
    for entry in validate_source_manifest():
        typer.echo(f"{entry.name}\t{entry.source_bucket.value}\t{entry.adapter_status.value}")


@app.command("manifest-check")
def manifest_check() -> None:
    """Validate source manifest."""
    entries = validate_source_manifest()
    typer.echo(f"Source manifest valid: {len(entries)} sources")


@app.command("build-splits")
def build_splits(
    input: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
) -> None:
    """Write train/dev/test split files."""
    counts = write_splits(read_jsonl(input), output_dir)
    typer.echo(json.dumps(counts, sort_keys=True))


@app.command()
def datasheet(input: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Render a datasheet for a JSONL case file."""
    write_text(output, render_datasheet(read_jsonl(input)))
    typer.echo(f"Wrote {output}")


@app.command()
def leaderboard(results: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Render a leaderboard from result JSON files."""
    write_text(output, render_leaderboard(results))
    typer.echo(f"Wrote {output}")


@app.command("create-curation-templates")
def create_curation_templates(output_dir: Path = typer.Option(...)) -> None:
    """Create CSV templates for source-backed case curation."""
    paths = create_templates(output_dir)
    typer.echo(f"Wrote {len(paths)} templates")


@app.command("source-card")
def source_card(source: str = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Generate a source card from the manifest."""
    entry = source_by_name(source)
    write_text(output, render_source_card(entry))
    typer.echo(f"Wrote {output}")


def _write_prepared(records: list[dict], output: Path) -> None:
    for record in records:
        validate_case_record(record)
    write_jsonl(output, records)
    typer.echo(f"Wrote {len(records)} records to {output}")


@app.command("prepare-human-effect-cases")
def prepare_human_effect_cases(
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Convert human-effect curation CSV to JSONL cases."""
    records = []
    for row in read_csv_rows(input):
        record = base_case_fields(row, "human_effect")
        value = row.get("trial_has_results", "")
        trial_has_results: Optional[bool]
        if value == "":
            trial_has_results = None
        else:
            trial_has_results = value.lower() in {"true", "1", "yes"}
        record.update(
            {
                "peptide": {"sequence": row["peptide_sequence"], "modifications": []},
                "claim_text": row["claim_text"],
                "category": row["category"],
                "evidence_level": row["evidence_level"],
                "evidence_direction": row["evidence_direction"],
                "claim_status": row["claim_status"],
                "safety_status": row["safety_status"],
                "source_evidence_type": row["source_evidence_type"],
                "trial_status": row.get("trial_status") or None,
                "trial_phase": row.get("trial_phase") or None,
                "trial_has_results": trial_has_results,
            }
        )
        records.append(record)
    _write_prepared(records, output)


@app.command("prepare-pdb-structure-cases")
def prepare_pdb_structure_cases(
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Convert PDB structure curation CSV to JSONL cases."""
    records = []
    for row in read_csv_rows(input):
        record = base_case_fields(row, "structure")
        record.update(
            {
                "peptide": {"sequence": row["peptide_sequence"], "modifications": []},
                "structure_id": row["structure_id"],
                "experimental_method": row["experimental_method"],
                "resolution_angstrom": float(row["resolution_angstrom"])
                if row.get("resolution_angstrom")
                else None,
                "gold_structure_reference": {
                    "source_database": row["source_database"],
                    "source_id": row["source_id"],
                    "source_url": row.get("source_url") or None,
                    "source_version": row["source_version"],
                    "retrieval_date": row["retrieval_date"],
                    "citation": row.get("citation") or None,
                },
                "gold_coordinates": [],
            }
        )
        records.append(record)
    _write_prepared(records, output)


@app.command("prepare-pdb-pose-cases")
def prepare_pdb_pose_cases(
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Convert PDB pose curation CSV to JSONL cases."""
    records = []
    for row in read_csv_rows(input):
        record = base_case_fields(row, "pose")
        contacts = json.loads(row.get("native_contacts_json") or "[]")
        record.update(
            {
                "peptide": {"sequence": row["peptide_sequence"], "modifications": []},
                "target": {
                    "protein": {
                        "name": row["target_name"],
                        "chain_id": row["target_chain_id"],
                    },
                    "target_family": row.get("target_name") or None,
                },
                "pdb_id": row["pdb_id"],
                "target_chain_id": row["target_chain_id"],
                "peptide_chain_id": row["peptide_chain_id"],
                "native_contacts": contacts,
                "binding_site_residues": [
                    item for item in row.get("binding_site_residues", "").split(";") if item
                ],
            }
        )
        records.append(record)
    _write_prepared(records, output)


@app.command("prepare-bindrank-panels")
def prepare_bindrank_panels(
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Convert binding-rank curation CSV to JSONL cases."""
    records = []
    for row in read_csv_rows(input):
        record = base_case_fields(row, "binding_rank")
        record.update(
            {
                "panel_id": row["panel_id"],
                "assay_type": row["assay_type"],
                "assay_unit": row["assay_unit"],
                "assay_conditions": row["assay_conditions"],
                "measurement_direction": row["measurement_direction"],
                "normalization_method": row["normalization_method"],
                "comparable_panel": row["comparable_panel"].lower() in {"true", "1", "yes"},
                "panel_exclusion_reason": row.get("panel_exclusion_reason") or None,
                "items": json.loads(row["items_json"]),
            }
        )
        records.append(record)
    _write_prepared(records, output)


@app.command("audit-real-cases")
def audit_real_cases(input: Path = typer.Option(...)) -> None:
    """Audit curation and provenance safeguards for cases."""
    result = audit_jsonl(input)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    if not result.passed:
        raise typer.Exit(1)


@app.command("release-check")
def release_check_command(
    input_dir: Path = typer.Option(...),
    min_structure: int = typer.Option(200),
    min_pose: int = typer.Option(100),
    min_binding_rank: int = typer.Option(25),
    min_human_effect: int = typer.Option(200),
) -> None:
    """Run release-directory checks."""
    passed, errors, warnings = release_check(
        input_dir,
        min_structure=min_structure,
        min_pose=min_pose,
        min_binding_rank=min_binding_rank,
        min_human_effect=min_human_effect,
    )
    for warning in warnings:
        typer.echo(f"warning: {warning}")
    if errors:
        for error in errors:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Release check {'passed' if passed else 'failed'}")


@app.command("build-release")
def build_release_command(output_dir: Path = typer.Option(...)) -> None:
    """Build the generated v1.0 release-candidate directory."""
    root = build_release(output_dir)
    typer.echo(f"Wrote release candidate to {root}")


@app.command("build-all-public-sources")
def build_all_public_sources_command(
    output_dir: Path = typer.Option(...),
    max_records_per_source: int = typer.Option(5000),
) -> None:
    """Build all real public-source release cases and reports."""
    counts = build_all_public_sources(output_dir, max_records_per_source=max_records_per_source)
    typer.echo(json.dumps(counts, indent=2, sort_keys=True))


@app.command("build-structure-cases")
def build_structure_cases_command(
    output: Path = typer.Option(...),
    min_cases: int = typer.Option(200),
) -> None:
    """Build real source-backed PDB/RCSB structure cases."""
    cases = build_public_structure_cases(output, min_cases=min_cases)
    typer.echo(f"Wrote {len(cases)} structure cases")


@app.command("build-pose-cases")
def build_pose_cases_command(
    output: Path = typer.Option(...),
    min_cases: int = typer.Option(100),
) -> None:
    """Build real source-backed PDB/RCSB target-bound pose cases."""
    cases = build_public_pose_cases(output, min_cases=min_cases)
    typer.echo(f"Wrote {len(cases)} pose cases")


@app.command("build-bindrank-panels")
def build_bindrank_panels_command(
    output: Path = typer.Option(...),
    min_panels: int = typer.Option(25),
) -> None:
    """Build real assay-aware IEDB binding-rank panels."""
    cases = build_public_bindrank_panels(output, min_panels=min_panels)
    typer.echo(f"Wrote {len(cases)} binding-rank panels")


@app.command("build-human-effect-cases")
def build_human_effect_cases_command(
    output: Path = typer.Option(...),
    min_cases: int = typer.Option(200),
) -> None:
    """Build real source-backed human-effect evidence cases."""
    cases = build_public_human_effect_cases(output, min_cases=min_cases)
    typer.echo(f"Wrote {len(cases)} human-effect cases")


@app.command("source-status")
def source_status(output: Path = typer.Option(...)) -> None:
    """Write source status rows for the current release directory."""
    release_dir = output.parent if output.parent.name != "references" else output.parent.parent
    case_files = {
        "structure": release_dir / "structure" / "cases.jsonl",
        "pose": release_dir / "pose" / "cases.jsonl",
        "binding_rank": release_dir / "binding_rank" / "cases.jsonl",
        "human_effect": release_dir / "human_effect" / "cases.jsonl",
    }
    counts = {}
    for track, path in case_files.items():
        if not path.exists():
            continue
        for record in read_jsonl(path):
            counts.setdefault(record["source_database"], {}).setdefault(track, 0)
            counts[record["source_database"]][track] += 1
    rows = source_status_rows(release_dir, counts)
    write_jsonl(output, rows)
    typer.echo(f"Wrote {len(rows)} source status rows")


@app.command("import-source")
def import_source(
    source: str = typer.Option(...),
    input: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Normalize a public source export as source-reference JSONL."""
    rows = []
    if input.suffix.lower() == ".json":
        payload = json.loads(input.read_text(encoding="utf-8"))
        iterable = payload if isinstance(payload, list) else payload.get("data", [])
        for index, record in enumerate(iterable):
            rows.append(
                {
                    "source": source,
                    "source_index": index,
                    "record": record,
                    "release_mode": "source_reference_only",
                }
            )
    else:
        for index, line in enumerate(input.read_text(encoding="utf-8").splitlines()):
            rows.append(
                {
                    "source": source,
                    "source_index": index,
                    "record": line,
                    "release_mode": "source_reference_only",
                }
            )
    write_jsonl(output, rows)
    typer.echo(f"Wrote {len(rows)} normalized source-reference rows")


@app.command("release-report")
def release_report(
    release_dir: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Render release-check report."""
    write_text(output, render_release_report(release_dir))
    typer.echo(f"Wrote {output}")


@app.command("quality-check-release")
def quality_check_release_command(
    release_dir: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
) -> None:
    """Build a source-backed release from a source-backed release candidate."""
    summary = quality_check_release(release_dir, output_dir)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("below_minimum"):
        raise typer.Exit(1)


@app.command("publish-check")
def publish_check_command(
    release_dir: Path = typer.Option(...),
    min_structure: int = typer.Option(200),
    min_pose: int = typer.Option(100),
    min_binding_rank: int = typer.Option(25),
    min_human_effect: int = typer.Option(200),
) -> None:
    """Run strict publishability checks for a source-backed release."""
    passed, errors, warnings, summary = publish_check(
        release_dir,
        min_structure=min_structure,
        min_pose=min_pose,
        min_binding_rank=min_binding_rank,
        min_human_effect=min_human_effect,
    )
    for warning in warnings:
        typer.echo(f"warning: {warning}")
    if errors:
        for error in errors:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps({"passed": passed, **summary}, indent=2, sort_keys=True))


@app.command("resolve-qc-disagreements")
def resolve_qc_disagreements_command(release_dir: Path = typer.Option(...)) -> None:
    """Resolve quality check disagreements conservatively."""
    summary = resolve_qc_disagreements(release_dir)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("unresolved"):
        raise typer.Exit(1)


@app.command("package-release")
def package_release_command(
    release_dir: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Run final checks and package a release directory as tar.gz."""
    try:
        summary = package_release(release_dir, output)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("fetch-pdb")
def fetch_pdb(ids: Path = typer.Option(...), output_dir: Path = typer.Option(...)) -> None:
    """Fetch mmCIF files for explicit PDB IDs."""
    adapter = RCSBAdapter()
    count = 0
    for pdb_id in ids.read_text(encoding="utf-8").splitlines():
        pdb_id = pdb_id.strip()
        if not pdb_id:
            continue
        adapter.fetch_mmcif(pdb_id, output_dir)
        count += 1
    typer.echo(f"Fetched {count} mmCIF files")


@app.command("fetch-rcsb-query")
def fetch_rcsb_query(
    query_type: str = typer.Option(...),
    limit: int = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Fetch an explicit RCSB query result."""
    if query_type != "peptide_complexes":
        raise typer.BadParameter("only query-type 'peptide_complexes' is implemented")
    payload = RCSBAdapter().search_peptide_complexes(limit)
    write_text(output, json.dumps(payload, indent=2, sort_keys=True))
    typer.echo(f"Wrote {output}")


@app.command("fetch-dailymed")
def fetch_dailymed(
    input: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
) -> None:
    """Fetch DailyMed search results or labels from explicit names/set IDs."""
    adapter = DailyMedAdapter()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in read_csv_rows(input):
        if row.get("set_id"):
            adapter.fetch_label(row["set_id"], output_dir)
        elif row.get("name"):
            payload = adapter.search(row["name"])
            write_text(output_dir / f"{row['name']}.json", json.dumps(payload, indent=2))
        count += 1
    typer.echo(f"Fetched {count} DailyMed records")


@app.command("fetch-clinicaltrials")
def fetch_clinicaltrials(
    query: str = typer.Option(...),
    limit: int = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Fetch ClinicalTrials.gov search results."""
    ClinicalTrialsAdapter().fetch_search(query, limit, output)
    typer.echo(f"Wrote {output}")


@app.command("fetch-iedb")
def fetch_iedb(input: Path = typer.Option(...), output: Path = typer.Option(...)) -> None:
    """Record an explicit IEDB query or local export path for later import."""
    payload = {
        "status": "source_reference_only",
        "input": input.read_text(encoding="utf-8") if input.exists() else str(input),
        "note": "IEDB export parsing requires license qc for the exact export route.",
    }
    write_text(output, json.dumps(payload, indent=2, sort_keys=True))
    typer.echo(f"Wrote {output}")


@app.command("compute-pdb-contacts")
def compute_pdb_contacts(
    structure: Path = typer.Option(...),
    target_chain: str = typer.Option(...),
    peptide_chain: str = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Compute residue contacts from a local PDB-format file."""
    contacts = compute_contacts(structure, target_chain, peptide_chain)
    write_text(output, json.dumps(contacts, indent=2, sort_keys=True))
    typer.echo(f"Wrote {len(contacts)} contacts")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
