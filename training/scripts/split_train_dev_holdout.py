"""Build train/dev/holdout splits and estimate lightweight training cost."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from training_common import (
    DEFAULT_RELEASE_DIR,
    DEFAULT_RUN_DIR,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)


def _token_estimate(value: Any) -> int:
    """Very small deterministic token estimate for budgeting/tests."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False) if not isinstance(value, str) else value
    return max(1, len(text) // 4)


def _cost_estimate(
    config_or_rows: dict[str, Any] | Iterable[dict[str, Any]] | int,
    train_rows: Iterable[dict[str, Any]] | None = None,
    dev_rows: Iterable[dict[str, Any]] | None = None,
    *,
    epochs: int = 1,
    price_per_million_tokens: float = 8.0,
) -> dict[str, Any]:
    """Estimate training tokens and cost.

    Accepts the historical ``_cost_estimate(rows)`` form and the richer
    ``_cost_estimate(config, train_rows, dev_rows)`` form used by the training
    pipeline tests.
    """
    if train_rows is not None or isinstance(config_or_rows, dict):
        config = config_or_rows if isinstance(config_or_rows, dict) else {}
        train_materialized = list(train_rows or [])
        dev_materialized = list(dev_rows or [])
        base_train_tokens = sum(int(row.get("token_estimate") or _token_estimate(row)) for row in train_materialized)
        dev_token_estimate = sum(int(row.get("token_estimate") or _token_estimate(row)) for row in dev_materialized)
        target_min = int(config.get("target_train_tokens_min") or base_train_tokens or 0)
        target_max = int(config.get("target_train_tokens_max") or max(target_min, base_train_tokens))
        expansion_factor = 1
        if base_train_tokens and base_train_tokens < target_min:
            expansion_factor = max(1, (target_min + base_train_tokens - 1) // base_train_tokens)
        train_token_estimate = base_train_tokens * expansion_factor
        if target_max:
            train_token_estimate = min(train_token_estimate, target_max)
        pricing = config.get("pricing") if isinstance(config.get("pricing"), dict) else {}
        sft_price = float(pricing.get("estimated_sft_usd_per_million_tokens", price_per_million_tokens))
        sampling_price = float(pricing.get("estimated_sampling_usd_per_million_tokens", 0.0))
        storage = float(pricing.get("estimated_storage_usd", 0.0))
        training_cost = train_token_estimate / 1_000_000 * sft_price
        sampling_cost = dev_token_estimate / 1_000_000 * sampling_price
        total_cost = training_cost + sampling_cost + storage
        hard_cap = float(config.get("hard_cost_cap_usd", total_cost))
        return {
            "record_count": len(train_materialized),
            "dev_record_count": len(dev_materialized),
            "base_train_token_estimate": base_train_tokens,
            "dev_token_estimate": dev_token_estimate,
            "training_expansion_factor": expansion_factor,
            "train_token_estimate": train_token_estimate,
            "estimated_training_tokens": train_token_estimate,
            "estimated_total_tokens": train_token_estimate + dev_token_estimate,
            "estimated_training_cost_usd": round(training_cost, 6),
            "estimated_sampling_cost_usd": round(sampling_cost, 6),
            "estimated_storage_usd": storage,
            "total_estimated_cost_usd": round(total_cost, 6),
            "hard_cost_cap_usd": hard_cap,
            "under_hard_cap": total_cost <= hard_cap,
        }

    rows = config_or_rows
    if isinstance(rows, int):
        record_count = 0
        input_tokens = int(rows)
    else:
        materialized = list(rows)
        record_count = len(materialized)
        input_tokens = sum(_token_estimate(row) for row in materialized)

    training_tokens = input_tokens * max(1, int(epochs))
    cost_usd = training_tokens / 1_000_000 * price_per_million_tokens

    return {
        "record_count": record_count,
        "epochs": max(1, int(epochs)),
        "estimated_input_tokens": input_tokens,
        "estimated_training_tokens": training_tokens,
        "estimated_total_tokens": training_tokens,
        "price_per_million_tokens_usd": price_per_million_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        "estimated_training_cost_usd": round(cost_usd, 6),
    }


def _load_release_rows(release_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(release_dir.rglob("cases.jsonl")):
        for row in read_jsonl(path):
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_splits(
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    output_dir: str | Path = DEFAULT_RUN_DIR / "splits",
    *,
    seed: int = 7,
    dev_fraction: float = 0.1,
    holdout_fraction: float = 0.1,
) -> dict[str, Any]:
    release_dir = Path(release_dir)
    output_dir = Path(output_dir)
    rows = _load_release_rows(release_dir)

    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: stable_hash(row.get("benchmark_id", row)))
    rng.shuffle(rows)

    n = len(rows)
    dev_n = int(n * dev_fraction)
    holdout_n = int(n * holdout_fraction)

    dev = rows[:dev_n]
    holdout = rows[dev_n : dev_n + holdout_n]
    train = rows[dev_n + holdout_n :]

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_jsonl(output_dir / "holdout.jsonl", holdout)

    manifest = {
        "status": "ok",
        "release_dir": str(release_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "train_count": len(train),
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "cost_estimate": _cost_estimate(train),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_DIR / "splits"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    manifest = build_splits(
        release_dir=args.release_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
