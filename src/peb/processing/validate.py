"""Validation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from peb.io import read_jsonl
from peb.schemas import validate_case_record, validate_prediction_record


def validate_records(path: Union[str, Path]) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for index, record in enumerate(read_jsonl(path), start=1):
        try:
            if "prediction_id" in record:
                validate_prediction_record(record)
            else:
                validate_case_record(record)
            count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}:{index}: {exc}")
    return count, errors


def validate_record_list(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        try:
            validate_case_record(record)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"record {index}: {exc}")
    return errors
