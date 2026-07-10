"""Evaluate the trained adapter in PEB format and update the local leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import infer_peb_engineer
from training_common import DEFAULT_RELEASE_DIR, DEFAULT_RUN_DIR, write_json

from peb.io import read_jsonl, write_jsonl, write_text
from peb.openrouter_leaderboard import (
    _baseline_rows,
    _cases_for_track,
    _copy_leaderboard_to_release,
    _evaluate_track,
    _leaderboard_row,
    _load_release_id,
    _utc_now,
    _write_leaderboard,
    safe_model_slug,
)

MODES = ("base", "tools_high_reasoning")
BASE_TRACKS = ("human_effect", "binding_rank")
SCOREABLE_MANIFEST_STATUSES = {"completed_clean", "completed_with_failures", "completed"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _mode_error_counts(manifest: dict[str, Any]) -> tuple[int, int, int]:
    summaries = manifest.get("track_summaries") or {}
    if not isinstance(summaries, dict):
        return 0, 0, 0
    api_errors = 0
    invalid_json = 0
    recovered = 0
    for summary in summaries.values():
        if not isinstance(summary, dict):
            continue
        api_errors += int(summary.get("unresolved_api_errors") or 0)
        invalid_json += int(summary.get("unresolved_invalid_json") or 0)
        recovered += int(summary.get("retry_recovered_count") or 0)
    return api_errors, invalid_json, recovered


def _preserved_rows(leaderboard_dir: Path, model_name: str) -> list[dict[str, Any]]:
    rows = _read_rows(leaderboard_dir / "leaderboard.json")
    return [row for row in rows if row.get("model_id") != model_name]


def _artifact_slugs(model_name: str) -> list[str]:
    legacy = model_name.replace("/", "_").replace("-", "_")
    strict = safe_model_slug(model_name)
    return list(dict.fromkeys([legacy, strict]))


def _log(message: str) -> None:
    print(f"[peb-eval] {message}", flush=True)


def _blocked_result(
    *,
    eval_dir: Path,
    leaderboard_result_dirs: list[Path],
    run_manifest: dict[str, Any],
    adapter_manifest: dict[str, Any],
    model_name: str,
    reason: str,
    details: dict[str, Any],
    manifests: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "status": "eval_blocked",
        "reason": reason,
        "details": details,
        "model_id": model_name,
        "base_model": run_manifest.get("model_name")
        or adapter_manifest.get("base_model")
        or infer_peb_engineer.DEFAULT_BASE_MODEL,
        "training": "LoRA SFT",
        "provider": "Tinker",
        "training_status": run_manifest.get("status"),
        "adapter_status": adapter_manifest.get("adapter_status") or adapter_manifest.get("status"),
        "competitive": False,
        "leaderboard_row_added": False,
        "manifests": manifests,
        "scores": {},
    }
    write_json(eval_dir / "peb_scores.json", result)
    for leaderboard_result_dir in leaderboard_result_dirs:
        write_json(leaderboard_result_dir / "eval_blocked.json", result)
    return result


def evaluate(
    run_dir: Path,
    release_dir: Path,
    leaderboard_dir: Path,
    model_name: str,
    *,
    timeout: int = 300,
    retries: int = 2,
    chunk_size: int = infer_peb_engineer.DEFAULT_CHUNK_SIZE,
    resume: bool = True,
    modes: tuple[str, ...] = ("base",),
) -> dict[str, Any]:
    del timeout, retries, chunk_size, resume
    unsupported_modes = [mode for mode in modes if mode != "base"]
    if unsupported_modes:
        raise ValueError("local PEB scoring currently supports base mode only")

    _log("local base scoring start")
    eval_dir = run_dir / "eval"
    predictions_dir = eval_dir / "peb_predictions"
    artifact_slugs = _artifact_slugs(model_name)
    leaderboard_result_dirs = [leaderboard_dir / "results" / slug for slug in artifact_slugs]
    leaderboard_prediction_dirs = [leaderboard_dir / "predictions" / slug for slug in artifact_slugs]
    for directory in [*leaderboard_result_dirs, *leaderboard_prediction_dirs]:
        directory.mkdir(parents=True, exist_ok=True)

    run_manifest = _read_json(run_dir / "run_manifest.json")
    adapter_manifest = _read_json(run_dir / "hf_upload" / "adapter_export_manifest.json") or _read_json(
        run_dir / "adapter_export_manifest.json"
    )
    _log("manifest load start")
    manifests: dict[str, dict[str, Any]] = {}
    for mode in modes:
        manifest_path = predictions_dir / f"{mode}_inference_manifest.json"
        manifest = _read_json(manifest_path)
        if not manifest:
            return _blocked_result(
                eval_dir=eval_dir,
                leaderboard_result_dirs=leaderboard_result_dirs,
                run_manifest=run_manifest,
                adapter_manifest=adapter_manifest,
                model_name=model_name,
                reason="missing_inference_manifest",
                details={"path": str(manifest_path)},
                manifests=manifests,
            )
        manifests[mode] = manifest
    _log("manifest load end")

    release_id = _load_release_id(release_dir)
    run_timestamp = _utc_now()
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    all_scores: dict[str, Any] = {}
    blocker = next(
        (
            item
            for item in manifests.values()
            if item.get("status") not in SCOREABLE_MANIFEST_STATUSES
        ),
        None,
    )
    if blocker:
        return _blocked_result(
            eval_dir=eval_dir,
            leaderboard_result_dirs=leaderboard_result_dirs,
            run_manifest=run_manifest,
            adapter_manifest=adapter_manifest,
            model_name=model_name,
            reason=str(blocker.get("reason") or "manifest_not_scoreable"),
            details={"manifest_status": blocker.get("status")},
            manifests=manifests,
        )

    for mode, manifest in manifests.items():
        mode_results: dict[str, dict[str, Any]] = {}
        attempted: dict[str, int] = {}
        completed: dict[str, int] = {}
        track_scores: dict[str, Any] = {}
        for track in BASE_TRACKS:
            prediction_path = predictions_dir / f"{mode}_{track}.jsonl"
            if not prediction_path.exists():
                return _blocked_result(
                    eval_dir=eval_dir,
                    leaderboard_result_dirs=leaderboard_result_dirs,
                    run_manifest=run_manifest,
                    adapter_manifest=adapter_manifest,
                    model_name=model_name,
                    reason="missing_prediction_file",
                    details={"path": str(prediction_path), "mode": mode, "track": track},
                    manifests=manifests,
                )
            if track == "human_effect":
                _log("human-effect prediction file load start")
            elif track == "binding_rank":
                _log("binding-rank prediction file load start")
            predictions = read_jsonl(prediction_path)
            if track == "human_effect":
                _log(f"human-effect prediction file load end rows={len(predictions)}")
                _log("human-effect scoring start")
            elif track == "binding_rank":
                _log(f"binding-rank prediction file load end rows={len(predictions)}")
                _log("binding-rank scoring start")
            cases = _cases_for_track(release_dir, track)
            for leaderboard_prediction_dir in leaderboard_prediction_dirs:
                write_jsonl(leaderboard_prediction_dir / f"{mode}_{track}.jsonl", predictions)
            attempted[track] = len(cases)
            completed[track] = len(predictions)
            result = _evaluate_track(track, cases, predictions if predictions else [])
            if track == "human_effect":
                _log("human-effect scoring end")
            elif track == "binding_rank":
                _log("binding-rank scoring end")
            mode_results[track] = result
            track_scores[track] = result
            for leaderboard_result_dir in leaderboard_result_dirs:
                write_text(
                    leaderboard_result_dir / f"{mode}_{track}.json",
                    json.dumps(result, indent=2, sort_keys=True),
                )

        api_errors, invalid_json, recovered = _mode_error_counts(manifest)
        row = _leaderboard_row(
            model={
                "model_id": model_name,
                "provider": "Tinker",
                "base_model": run_manifest.get("model_name")
                or adapter_manifest.get("base_model")
                or infer_peb_engineer.DEFAULT_BASE_MODEL,
            },
            mode=mode,
            results=mode_results,
            attempted=attempted,
            completed=completed,
            invalid_json_count=invalid_json,
            api_error_count=api_errors,
            cost=0.0,
            release_id=release_id,
            run_timestamp=run_timestamp,
            is_baseline=False,
            competitive=manifest.get("status") in SCOREABLE_MANIFEST_STATUSES,
        )
        row.update(
            {
                "artifact_required": True,
                "base_model": run_manifest.get("model_name")
                or adapter_manifest.get("base_model")
                or infer_peb_engineer.DEFAULT_BASE_MODEL,
                "training": "LoRA SFT",
                "retry_recovered_count": recovered,
                "retry_still_failed_count": api_errors + invalid_json,
                "fallback_prediction_count": 0,
                "unresolved_provider_error_count": api_errors,
                "unresolved_invalid_json_count": invalid_json,
            }
        )
        rows.append(row)
        details.append(
            {
                "model_id": model_name,
                "mode": mode,
                "attempted": attempted,
                "completed": completed,
                "invalid_json_count": invalid_json,
                "api_error_count": api_errors,
                "retry_recovered_count": recovered,
                "status": manifest.get("status"),
                "reason": manifest.get("reason"),
            }
        )
        all_scores[mode] = track_scores

    preserved = _preserved_rows(leaderboard_dir, model_name)
    if not preserved:
        preserved = _baseline_rows(release_dir, release_id, run_timestamp)
    _log("leaderboard write start")
    _write_leaderboard(leaderboard_dir, [*preserved, *rows])
    manifest_payload = {
        "run_timestamp": run_timestamp,
        "benchmark_release": release_id,
        "models": [model_name],
        "modes": list(modes),
        "details": details,
        "leaderboard_rows": len([*preserved, *rows]),
    }
    write_text(leaderboard_dir / "run_manifest.json", json.dumps(manifest_payload, indent=2, sort_keys=True))
    _copy_leaderboard_to_release(leaderboard_dir, release_dir)
    _log("leaderboard write end")

    complete_modes = [
        mode for mode, item in manifests.items() if item.get("status") in SCOREABLE_MANIFEST_STATUSES
    ]
    result = {
        "status": "eval_blocked" if blocker else "completed",
        "reason": blocker.get("reason") if blocker else None,
        "model_id": model_name,
        "base_model": run_manifest.get("model_name")
        or adapter_manifest.get("base_model")
        or infer_peb_engineer.DEFAULT_BASE_MODEL,
        "training": "LoRA SFT",
        "provider": "Tinker",
        "training_status": run_manifest.get("status"),
        "adapter_status": adapter_manifest.get("adapter_status") or adapter_manifest.get("status"),
        "competitive": not blocker and len(complete_modes) == len(modes),
        "leaderboard_row_added": True,
        "complete_modes": complete_modes,
        "manifests": manifests,
        "scores": all_scores,
    }
    write_json(eval_dir / "peb_scores.json", result)
    if blocker:
        for leaderboard_result_dir in leaderboard_result_dirs:
            write_json(leaderboard_result_dir / "eval_blocked.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--leaderboard-dir", type=Path, default=Path("leaderboard"))
    parser.add_argument("--model-name", default="the-spice")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=infer_peb_engineer.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mode", choices=("base",), default="base")
    args = parser.parse_args()
    _log("script start")
    modes = (args.mode,)
    result = evaluate(
        args.run_dir,
        args.release_dir,
        args.leaderboard_dir,
        args.model_name,
        timeout=args.timeout,
        retries=args.retries,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
        modes=modes,
    )
    print(f"PEB eval status={result['status']} reason={result.get('reason')}")


if __name__ == "__main__":
    main()
