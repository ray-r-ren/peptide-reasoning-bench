"""Dataset datasheet generation."""

from __future__ import annotations

from collections import Counter
from typing import Any


def render_datasheet(records: list[dict[str, Any]], title: str = "PEB Datasheet") -> str:
    tracks = Counter(record.get("track", "unknown") for record in records)
    sources = Counter(record.get("source_database", "unknown") for record in records)
    lines = [
        f"# {title}",
        "",
        "This datasheet describes benchmark case metadata and provenance.",
        "",
        "PEB evaluates evidence-grounded peptide engineering judgment. The human-effect track is evidence classification and unsupported-claim detection.",
        "",
        "## Case Counts",
    ]
    lines.extend(f"- {track}: {count}" for track, count in sorted(tracks.items()))
    lines.extend(["", "## Sources"])
    lines.extend(f"- {source}: {count}" for source, count in sorted(sources.items()))
    lines.extend(
        [
            "",
            "## Redistribution",
            "Synthetic fixtures are included for tests only. Source-backed release cases must retain source identifiers, retrieval dates, license notes, and redistribution snapshots.",
            "",
            "## Safety Scope",
            "The benchmark does not validate peptide use in humans and does not score efficacy prediction.",
        ]
    )
    return "\n".join(lines) + "\n"
