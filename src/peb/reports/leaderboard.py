"""Leaderboard rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PRIMARY_METRICS = {
    "structure": "backbone_rmsd",
    "pose": "native_contact_recovery",
    "binding_rank": "pairwise_ranking_accuracy",
    "human_effect": "category_macro_f1",
}


def _load_results(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json"))
    results = []
    for file in files:
        results.append(json.loads(file.read_text(encoding="utf-8")))
    return results


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for result in results:
        key = (
            result.get("model_name", "baseline"),
            result.get("track", "unknown"),
            result.get("leaderboard_group", "competitive"),
            result.get("competitive", True) is not False,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _primary_metric(result: dict[str, Any]) -> tuple[str, Any]:
    track = result.get("track", "unknown")
    metrics = result.get("metrics", {})
    preferred = PRIMARY_METRICS.get(track)
    if preferred in metrics:
        return preferred, metrics[preferred]
    primary_key = next(iter(metrics), "n/a")
    return primary_key, metrics.get(primary_key, "n/a")


def render_leaderboard(results_path: Union[str, Path]) -> str:
    results = _dedupe_results(_load_results(Path(results_path)))
    lines = [
        "# PEB Baseline Leaderboard",
        "",
        "## Competitive Baselines",
        "",
        "| model | track | primary metric | value |",
        "| --- | --- | --- | ---: |",
    ]
    competitive = [
        result
        for result in results
        if result.get("competitive", True) is not False
        and result.get("leaderboard_group") != "oracle_sanity_check"
    ]
    sanity = [result for result in results if result not in competitive]
    for result in competitive:
        track = result.get("track", "unknown")
        primary_key, primary_value = _primary_metric(result)
        model = result.get("model_name", "baseline")
        lines.append(f"| {model} | {track} | {primary_key} | {primary_value} |")
    if not competitive:
        lines.append("| no_results | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Oracle And Sanity-Check Baselines",
            "",
            "| model | track | primary metric | value | leaderboard use |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for result in sanity:
        track = result.get("track", "unknown")
        primary_key, primary_value = _primary_metric(result)
        model = result.get("model_name", "baseline")
        lines.append(f"| {model} | {track} | {primary_key} | {primary_value} | excluded |")
    if not sanity:
        lines.append("| no_results | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "Oracle sanity checks may copy labels or use answer-equivalent fields and are excluded from competitive scoring.",
            "",
            "Baselines are weak reference systems for smoke testing and do not indicate scientific adequacy.",
        ]
    )
    return "\n".join(lines) + "\n"
