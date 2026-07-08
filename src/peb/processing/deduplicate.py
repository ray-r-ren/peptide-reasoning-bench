"""Deduplication helpers."""

from __future__ import annotations

from typing import Any


def deduplicate_by_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        identifier = str(record.get("benchmark_id") or record.get("source_id"))
        if identifier in seen:
            continue
        seen.add(identifier)
        output.append(record)
    return output

