"""Split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from peb.io import write_jsonl


def assign_split(index: int) -> str:
    bucket = index % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


def write_splits(records: list[dict[str, Any]], output_dir: Union[str, Path]) -> dict[str, int]:
    split_records = {"train": [], "dev": [], "test": []}
    for index, record in enumerate(records):
        split = record.get("split") or assign_split(index)
        record["split"] = split
        split_records[split].append(record)
    for split, values in split_records.items():
        write_jsonl(Path(output_dir) / f"{split}.jsonl", values)
    return {split: len(values) for split, values in split_records.items()}
