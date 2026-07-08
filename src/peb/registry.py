"""Source registry and manifest validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from peb.schemas import SourceManifestEntry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "sources.yaml"


def load_source_manifest(path: Union[str, Path] = SOURCE_MANIFEST) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a list under 'sources'")
    return sources


def validate_source_manifest(
    path: Union[str, Path] = SOURCE_MANIFEST,
) -> list[SourceManifestEntry]:
    return [SourceManifestEntry.model_validate(record) for record in load_source_manifest(path)]


def source_by_name(name: str, path: Union[str, Path] = SOURCE_MANIFEST) -> SourceManifestEntry:
    normalized = name.lower()
    for entry in validate_source_manifest(path):
        if entry.name.lower() == normalized:
            return entry
    raise KeyError(f"source not found: {name}")
