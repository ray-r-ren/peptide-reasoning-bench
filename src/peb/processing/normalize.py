"""Normalization helpers."""

from __future__ import annotations


def normalize_source_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    if value not in {"train", "dev", "test"}:
        raise ValueError(f"unknown split: {value}")
    return value

