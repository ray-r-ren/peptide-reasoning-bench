"""Leakage grouping helpers."""

from __future__ import annotations

from typing import Optional


def leakage_group_id(*parts: Optional[str]) -> str:
    cleaned = [part for part in parts if part]
    return "::".join(cleaned) if cleaned else "unknown"
