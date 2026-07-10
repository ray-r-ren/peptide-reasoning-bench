"""Check training data for leakage against held-out PEB benchmark cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training_common import DEFAULT_RELEASE_DIR, read_jsonl, write_json

EVAL_SPLITS = {"dev", "validation", "val", "test", "holdout", "leaderboard"}
ID_KEYS = {
    "benchmark_id",
    "case_id",
    "source_id",
    "source_ids",
    "panel_id",
    "target_id",
    "item_id",
}
HASH_KEYS = {
    "processed_record_hash",
    "source_record_hash",
    "record_hash",
    "case_hash",
    "hash",
}
ANSWER_LIKE_MARKERS = {
    "normalized_rank",
    "measured_value",
    "evidence_level",
    "claim_status",
    "safety_status",
    "native_contacts",
    "gold_coordinates",
}


def _collect_values(obj: Any, keys: set[str]) -> set[str]:
    values: set[str] = set()

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif key and key.lower() in keys and value is not None:
            text = str(value).strip()
            if text:
                values.add(text)

    walk(obj)
    return values


def _is_eval_row(row: dict[str, Any]) -> bool:
    split = str(row.get("split", "")).lower()
    if not split:
        return True
    return split in EVAL_SPLITS


def _benchmark_rows(release_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(release_dir.rglob("*.jsonl")):
        if "predictions" in path.parts or "leaderboard" in path.parts:
            continue
        for row in read_jsonl(path):
            if isinstance(row, dict) and _is_eval_row(row):
                rows.append(row)
    return rows


def _training_rows(training_path: Path) -> list[dict[str, Any]]:
    if training_path.is_dir():
        rows: list[dict[str, Any]] = []
        for filename in ("train.jsonl", "dev.jsonl", "holdout.jsonl"):
            path = training_path / filename
            if path.exists():
                rows.extend(read_jsonl(path))
        return rows
    return read_jsonl(training_path)


def _contains_answer_like_prompt_field(row: dict[str, Any]) -> bool:
    text = json.dumps(row.get("messages", row), sort_keys=True, ensure_ascii=False).lower()
    return any(marker in text for marker in ANSWER_LIKE_MARKERS)


def check_leakage(
    training_path: str | Path,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    training_path = Path(training_path)
    release_dir = Path(release_dir)

    train_rows = _training_rows(training_path)
    bench_rows = _benchmark_rows(release_dir)

    train_ids: set[str] = set()
    bench_ids: set[str] = set()
    train_hashes: set[str] = set()
    bench_hashes: set[str] = set()

    for row in train_rows:
        train_ids |= _collect_values(row, ID_KEYS)
        train_hashes |= _collect_values(row, HASH_KEYS)

    for row in bench_rows:
        bench_ids |= _collect_values(row, ID_KEYS)
        bench_hashes |= _collect_values(row, HASH_KEYS)

    id_overlaps = sorted(train_ids & bench_ids)
    hash_overlaps = sorted(train_hashes & bench_hashes)
    overlaps = sorted(set(id_overlaps) | set(hash_overlaps))
    issues: list[dict[str, Any]] = []
    if overlaps:
        issues.append(
            {
                "type": "test_case_label_in_training_data",
                "overlap_count": len(overlaps),
                "examples": overlaps[:20],
            }
        )
    answer_like_count = sum(1 for row in train_rows if _contains_answer_like_prompt_field(row))
    if answer_like_count:
        issues.append(
            {
                "type": "answer_like_field_in_prompt",
                "count": answer_like_count,
            }
        )

    report = {
        "status": "fail" if issues else "pass",
        "passed": not issues,
        "leakage_detected": bool(overlaps),
        "overlap_count": len(overlaps),
        "issues": issues,
        "identifier_overlap_count": len(id_overlaps),
        "exact_hash_overlap_count": len(hash_overlaps),
        "train_record_count": len(train_rows),
        "benchmark_eval_record_count": len(bench_rows),
        "training_identifier_count": len(train_ids),
        "benchmark_identifier_count": len(bench_ids),
        "training_hash_count": len(train_hashes),
        "benchmark_hash_count": len(bench_hashes),
        "overlaps": overlaps[:200],
        "identifier_overlaps": id_overlaps[:200],
        "exact_hash_overlaps": hash_overlaps[:200],
    }

    if output_path is not None:
        target = Path(output_path)
        if target.suffix != ".json":
            target = target / "leakage_report.json"
        write_json(target, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_path")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--output-path")
    args = parser.parse_args()

    report = check_leakage(
        training_path=args.training_path,
        release_dir=args.release_dir,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
