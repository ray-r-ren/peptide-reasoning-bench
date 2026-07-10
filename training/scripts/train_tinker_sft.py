"""Launch or dry-run Tinker SFT training for the PEB reference model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from training_common import DEFAULT_RUN_DIR, TINKER_ENV_NAME, read_jsonl, write_json

DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_MODEL_NAME = "the-spice"
DEFAULT_TRAINING_DATA = Path("training/data/train.jsonl")


def _estimate_tokens(rows: list[dict[str, Any]]) -> int:
    return sum(max(1, len(json.dumps(row, sort_keys=True, ensure_ascii=False)) // 4) for row in rows)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        text = value.strip()
        if text.isdigit():
            parsed: Any = int(text)
        else:
            try:
                parsed = float(text)
            except ValueError:
                parsed = text
        data[key.strip()] = parsed
    return data


def _base_model_from_config(config: dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    model_name = str(config.get("base_model") or config.get("model_name") or DEFAULT_BASE_MODEL)
    if model_name == "GPT-OSS-20B":
        return DEFAULT_BASE_MODEL
    return model_name


def _valid_training_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid = []
    skipped = 0
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            valid.append(row)
        else:
            skipped += 1
    return valid, skipped


def _write_blocked_manifest(run_dir: Path, hard_blocker: str, details: dict[str, Any]) -> dict[str, Any]:
    manifest = {"status": "blocked", "hard_blocker": hard_blocker, **details}
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def launch_training(
    config_path: str | Path = DEFAULT_TRAINING_DATA,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    base_model: str | None = None,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    dry_run: bool = True,
    require_tinker: bool = False,
    run_to_completion: bool = False,
    max_cost_usd: float | None = None,
    export_adapter: bool = False,
    batch_size: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    """Prepare a Tinker SFT run.

    This function is CI-safe: it does not import Tinker or require secrets at module import time.
    By default it performs a dry run and writes reproducible local metadata.
    """
    config_path = Path(config_path)
    run_dir = Path(run_dir)
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_TRAINING_DATA.parent
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = _read_simple_yaml(config_path)
    resolved_base_model = _base_model_from_config(config, base_model)
    resolved_learning_rate = float(config.get("learning_rate", learning_rate))
    lora_rank = int(config.get("lora_rank", 32))
    seed = int(config.get("seed", 42))
    train_rows = _load_rows(data_dir / "train.jsonl")
    valid_rows, skipped_examples = _valid_training_rows(train_rows)
    token_estimate = _estimate_tokens(valid_rows)
    has_tinker_key = bool(os.environ.get(TINKER_ENV_NAME))
    cost_path = artifacts_dir / "cost_estimate.json"
    cost_report = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
    total_cost = float(cost_report.get("total_estimated_cost_usd") or 0.0)

    training_config = {
        "model_name": model_name,
        "base_model": resolved_base_model,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "epochs": int(epochs),
        "learning_rate": resolved_learning_rate,
        "lora_rank": lora_rank,
        "seed": seed,
        "record_count": len(valid_rows),
        "skipped_examples": skipped_examples,
        "estimated_training_tokens": token_estimate * max(1, int(epochs)),
        "dry_run": bool(dry_run),
        "require_tinker": bool(require_tinker),
        "extra": extra,
    }
    write_json(run_dir / "training_config.json", training_config)

    if max_cost_usd is not None and total_cost > max_cost_usd:
        manifest = _write_blocked_manifest(
            run_dir,
            "cost_cap_exceeded",
            {
                "estimated_cost_usd": total_cost,
                "max_cost_usd": max_cost_usd,
                "training_config": str(run_dir / "training_config.json"),
            },
        )
        raise SystemExit(manifest["hard_blocker"])

    needs_tinker = run_to_completion or require_tinker or not dry_run
    if needs_tinker and not has_tinker_key:
        manifest = _write_blocked_manifest(
            run_dir,
            "missing_tinker_api_key",
            {"training_config": str(run_dir / "training_config.json")},
        )
        raise SystemExit(manifest["hard_blocker"])

    if not run_to_completion:
        manifest = {
            "status": "dry_run",
            "reason": "dry_run_requested",
            "training_config": str(run_dir / "training_config.json"),
            "record_count": len(valid_rows),
            "valid_train_examples": len(valid_rows),
            "skipped_examples": skipped_examples,
            "estimated_training_tokens": training_config["estimated_training_tokens"],
        }
        write_json(run_dir / "run_manifest.json", manifest)
        return manifest

    import tinker  # type: ignore[import-not-found]

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(resolved_base_model, rank=lora_rank, seed=seed)
    tokenizer = training_client.get_tokenizer()
    for index in range(0, len(valid_rows), max(1, int(batch_size))):
        rows_batch = valid_rows[index : index + max(1, int(batch_size))]
        batch = []
        for row in rows_batch:
            tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
            model_input = tinker.ModelInput.from_ints(tokens)
            batch.append(tinker.Datum(model_input=model_input, loss_fn_inputs={"labels": tokens}))
        training_client.forward_backward(batch, "cross_entropy").result()
        training_client.optim_step(tinker.AdamParams(learning_rate=resolved_learning_rate)).result()
    state = training_client.save_state("peb-engineer-v0/final").result()
    sampler_weights = None
    if export_adapter and hasattr(training_client, "save_weights_for_sampler"):
        sampler_weights = training_client.save_weights_for_sampler("peb-engineer-v0/sampler_weights/final").result()

    hf_upload = run_dir / "hf_upload"
    hf_upload.mkdir(parents=True, exist_ok=True)
    model_card = {
        "model_name": model_name,
        "base_model": resolved_base_model,
        "adapter_type": "LoRA",
        "status": "training_completed",
    }
    write_json(hf_upload / "model_card.json", model_card)
    adapter_manifest = {
        "status": "remote_export_available" if sampler_weights else "remote_state_available",
        "adapter_type": "LoRA",
        "remote_state_path": getattr(state, "path", None),
        "remote_sampler_path": getattr(sampler_weights, "path", None) if sampler_weights else None,
        "adapter_model_exists": False,
    }
    write_json(run_dir / "adapter_export_manifest.json", adapter_manifest)
    write_json(hf_upload / "adapter_export_manifest.json", adapter_manifest)
    run_manifest = {
        "status": "training_completed",
        "model_name": resolved_base_model,
        "display_model_name": model_name,
        "training_config": str(run_dir / "training_config.json"),
        "valid_train_examples": len(valid_rows),
        "skipped_examples": skipped_examples,
        "estimated_training_tokens": training_config["estimated_training_tokens"],
        "adapter_status": adapter_manifest["status"],
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    return {**run_manifest, "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-path", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--require-tinker", action="store_true")
    args = parser.parse_args()

    manifest = launch_training(
        config_path=args.training_path,
        run_dir=args.run_dir,
        model_name=args.model_name,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dry_run=not args.launch,
        require_tinker=args.require_tinker,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
