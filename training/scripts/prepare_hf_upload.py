"""Prepare a safe Hugging Face upload folder for the PEB LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from training_common import DEFAULT_RUN_DIR, SECRET_PATTERNS, TINKER_ENV_NAME, write_json

DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"
DEFAULT_MODEL_NAME = "the-spice"
BINARY_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def _redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _copy_or_redact(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix in BINARY_SUFFIXES:
        shutil.copy2(src, dst)
        return "copied_binary"

    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return "copied_binary"

    redacted = _redact_secrets(text)
    dst.write_text(redacted, encoding="utf-8")
    return "redacted_text" if redacted != text else "copied_text"


def _find_first(run_dir: Path, filename: str) -> Path | None:
    candidates = [
        run_dir / filename,
        run_dir / "adapter" / filename,
        run_dir / "adapter_raw" / filename,
        run_dir / "peft_adapter" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    for path in sorted(run_dir.rglob(filename)):
        if "hf_upload" not in path.parts and path.is_file():
            return path

    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _base_model(run_dir: Path, config_path: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("model_name"):
        return str(run_manifest["model_name"])
    config = _read_simple_yaml(config_path) if config_path is not None else {}
    model_name = str(config.get("base_model") or config.get("model_name") or DEFAULT_BASE_MODEL)
    return DEFAULT_BASE_MODEL if model_name == "GPT-OSS-20B" else model_name


def prepare_upload(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    config_or_output_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    base_model: str | None = None,
    export_adapter: bool = False,
) -> dict[str, Any]:
    """Prepare a Hugging Face upload directory and return a manifest."""
    run_dir = Path(run_dir)
    config_path: Path | None = None
    output_dir: Path
    if config_or_output_dir is not None and Path(config_or_output_dir).suffix in {".yaml", ".yml"}:
        config_path = Path(config_or_output_dir)
        output_dir = run_dir / "hf_upload"
    else:
        output_dir = Path(config_or_output_dir) if config_or_output_dir is not None else run_dir / "hf_upload"
    resolved_base_model = _base_model(run_dir, config_path, base_model)
    root_adapter_manifest = _read_json(run_dir / "adapter_export_manifest.json")
    remote_sampler_path = root_adapter_manifest.get("remote_sampler_path")
    local_download_status = "not_requested"
    adapter_status = root_adapter_manifest.get("status", "unknown")
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None

    if export_adapter and remote_sampler_path:
        if not os.environ.get(TINKER_ENV_NAME):
            local_download_status = "blocked_missing_auth_env"
        else:
            import tinker_cookbook  # type: ignore[import-not-found]

            raw_dir = run_dir / "adapter_raw"
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            downloaded = tinker_cookbook.weights.download(
                tinker_path=str(remote_sampler_path),
                output_dir=str(raw_dir),
            )
            tinker_cookbook.weights.build_lora_adapter(
                base_model=resolved_base_model,
                adapter_path=str(downloaded),
                output_path=str(output_dir),
            )
            local_download_status = "completed"
            adapter_status = "local_peft_adapter_available"

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_readme = output_dir / "README.md"
    if generated_readme.exists():
        generated_readme.unlink()

    copied: list[dict[str, str]] = []
    required_files = ["adapter_config.json", "adapter_model.safetensors"]

    for filename in required_files:
        if (output_dir / filename).exists():
            continue
        src = _find_first(run_dir, filename)
        if src is None:
            continue
        status = _copy_or_redact(src, output_dir / filename)
        copied.append({"source": str(src), "target": filename, "status": status})

    optional_files = [
        "training_config.json",
        "dataset_summary.json",
        "source_coverage.json",
        "leakage_report.json",
        "sanity_eval_results.json",
        "cost_estimate.json",
        "run_config.yaml",
        "model_card.json",
    ]

    for filename in optional_files:
        src = _find_first(run_dir, filename)
        if src is None and artifacts_dir is not None:
            candidate = artifacts_dir / filename
            src = candidate if candidate.exists() else None
        if src is None:
            continue
        status = _copy_or_redact(src, output_dir / filename)
        copied.append({"source": str(src), "target": filename, "status": status})

    model_card = {
        "model_name": model_name,
        "base_model": resolved_base_model,
        "benchmark": "PEB v1.0-RC",
        "task": "peptide-reasoning model evaluation",
        "adapter_format": "peft_lora",
    }
    write_json(output_dir / "model_card.json", model_card)
    write_json(
        output_dir / "training_adapter_config.json",
        {
            "model_name": model_name,
            "base_model": resolved_base_model,
            "adapter_type": "LoRA",
            "export_adapter": bool(export_adapter),
        },
    )

    adapter_manifest = {
        **root_adapter_manifest,
        "adapter_type": "LoRA",
        "adapter_status": adapter_status,
        "status": adapter_status,
        "local_download_status": local_download_status,
        "adapter_model_exists": (output_dir / "adapter_model.safetensors").exists(),
    }
    write_json(output_dir / "adapter_export_manifest.json", adapter_manifest)

    manifest = {
        "status": "ready" if adapter_manifest["adapter_model_exists"] else "metadata_only",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "base_model": resolved_base_model,
        "adapter_status": adapter_status,
        "local_download_status": local_download_status,
        "copied_files": copied,
        "required_files_present": {
            filename: (output_dir / filename).exists() for filename in required_files
        },
    }
    write_json(output_dir / "upload_manifest.json", manifest)

    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix not in BINARY_SUFFIXES:
            try:
                if _has_secret(path.read_text(encoding="utf-8")):
                    raise RuntimeError(f"Secret-like value remained in upload file: {path}")
            except UnicodeDecodeError:
                continue

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    args = parser.parse_args()

    manifest = prepare_upload(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        base_model=args.base_model,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
