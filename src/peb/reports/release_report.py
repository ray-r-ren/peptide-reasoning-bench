"""Release report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from peb.processing.release_check import release_check


def render_release_report(release_dir: Union[str, Path]) -> str:
    release_path = Path(release_dir)
    passed, errors, warnings = release_check(release_path)
    lines = [
        f"# {release_path.name} Release Check Report",
        "",
        f"Status: {'passed' if passed else 'failed'}",
        "",
        "## Errors",
    ]
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "A passing release check means the package structure, provenance fields, and safety gates are internally consistent. It does not certify scientific completeness.",
        ]
    )
    return "\n".join(lines) + "\n"
