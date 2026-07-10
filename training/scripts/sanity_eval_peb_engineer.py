"""Run a lightweight heldout sanity evaluation for the trained sampler."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import infer_peb_engineer
from training_common import (
    DEFAULT_RUN_DIR,
    read_jsonl,
    stable_hash,
    strict_response_template,
    write_json,
    write_jsonl,
)

RAW_OUTPUT_CHAR_LIMIT = 4000
KNOWN_REPORT_RETRY_MAX_TOKENS = 512


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_status(value: Any) -> bool:
    return str(value) in {"completed", "training_completed"}


def _prompt_messages(row: dict[str, Any]) -> Optional[list[dict[str, str]]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
            if role == "assistant":
                break
            if role and content:
                prompt.append({"role": role, "content": content})
        if prompt:
            return prompt
    prompt = row.get("prompt") or row.get("instruction")
    if prompt:
        return [
            {
                "role": "system",
                "content": "You are a peptide engineering evaluator that returns strict JSON.",
            },
            {"role": "user", "content": str(prompt)},
        ]
    return None


def _sanity_max_tokens(task_type: str) -> int:
    if "pose" in task_type:
        return 1536
    if "integrated" in task_type:
        return 1024
    if "structure" in task_type:
        return 512
    return 768


def _validate_engineer_report(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    template = strict_response_template()
    errors: list[str] = []
    for key, expected in template.items():
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        value = payload[key]
        if isinstance(expected, dict) and not isinstance(value, dict):
            errors.append(f"type:{key}:dict")
        elif isinstance(expected, list) and not isinstance(value, list):
            errors.append(f"type:{key}:list")
        elif isinstance(expected, str) and not isinstance(value, str):
            errors.append(f"type:{key}:str")
    pose = payload.get("pose_contact_assessment")
    if isinstance(pose, dict):
        for nested_key, expected_type in (
            ("status", str),
            ("confidence", str),
            ("interface_residues", list),
            ("predicted_contacts", list),
            ("notes", str),
        ):
            if nested_key not in pose:
                errors.append(f"missing:pose_contact_assessment.{nested_key}")
            elif not isinstance(pose[nested_key], expected_type):
                errors.append(f"type:pose_contact_assessment.{nested_key}:{expected_type.__name__}")
        contacts = pose.get("predicted_contacts")
        if isinstance(contacts, list) and len(contacts) > infer_peb_engineer.MAX_PREDICTED_CONTACTS:
            errors.append("pose_contact_assessment.predicted_contacts:too_many")
        if isinstance(contacts, list):
            allowed = {"peptide_residue", "target_residue", "distance_angstrom"}
            for index, contact in enumerate(contacts):
                if not isinstance(contact, dict):
                    errors.append(f"type:pose_contact_assessment.predicted_contacts.{index}:dict")
                    continue
                extra = set(contact) - allowed
                if extra:
                    errors.append(
                        f"pose_contact_assessment.predicted_contacts.{index}:extra_fields:{','.join(sorted(extra))}"
                    )
                if "peptide_residue" not in contact:
                    errors.append(f"missing:pose_contact_assessment.predicted_contacts.{index}.peptide_residue")
                if "target_residue" not in contact:
                    errors.append(f"missing:pose_contact_assessment.predicted_contacts.{index}.target_residue")
    return not errors, errors


def _stats_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "attempted": 0,
            "json_valid_count": 0,
            "parse_valid_count": 0,
            "schema_valid_count": 0,
            "parse_invalid_count": 0,
            "schema_invalid_count": 0,
        }
    )
    for row in rows:
        task_type = str(row.get("task_type") or "unknown")
        bucket = stats[task_type]
        bucket["attempted"] += 1
        if row.get("parse_valid"):
            bucket["parse_valid_count"] += 1
        else:
            bucket["parse_invalid_count"] += 1
        if row.get("schema_valid"):
            bucket["schema_valid_count"] += 1
        else:
            bucket["schema_invalid_count"] += 1
        if row.get("json_valid"):
            bucket["json_valid_count"] += 1
    return {
        task_type: {
            **bucket,
            "json_valid_rate": bucket["json_valid_count"] / bucket["attempted"]
            if bucket["attempted"]
            else 0.0,
        }
        for task_type, bucket in sorted(stats.items())
    }


def _prediction_record(
    *,
    index: int,
    source_row: dict[str, Any],
    text: str,
    prompt_hash: str,
    status_if_valid: str = "completed",
    error: Optional[str] = None,
) -> dict[str, Any]:
    task_type = str(source_row.get("task_type") or "unknown")
    parsed_payload: Optional[dict[str, Any]] = None
    extract_method = None
    schema_errors: list[str] = []
    parse_valid = False
    schema_valid = False
    status = "failed"
    normalization_notes: list[str] = []
    extracted_payload: Optional[dict[str, Any]] = None
    degenerate_reason = None
    if text:
        degenerate_reason = infer_peb_engineer._degenerate_output_reason(text)
        if degenerate_reason:
            error = error or f"degenerate_non_json_output:{degenerate_reason}"
        else:
            try:
                extracted = infer_peb_engineer._extract_json_payload_with_metadata(text)
                extracted_payload = extracted["payload"]
                parsed_payload, normalization_notes = infer_peb_engineer._normalise_report_payload(
                    extracted_payload
                )
                extract_method = extracted["method"]
                parse_valid = True
                schema_valid, schema_errors = _validate_engineer_report(parsed_payload)
                status = status_if_valid if schema_valid else "schema_invalid"
            except Exception as exc:
                error = error or infer_peb_engineer._sanitize_error(exc)
    elif error is None:
        error = "empty_prediction_text"
    raw_output = text[:RAW_OUTPUT_CHAR_LIMIT]
    return {
        "index": index,
        "example_id": source_row.get("example_id"),
        "benchmark_id": source_row.get("benchmark_id"),
        "task_type": task_type,
        "prompt_hash": prompt_hash,
        "status": status,
        "json_valid": parse_valid and schema_valid,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "degenerate_output_reason": degenerate_reason,
        "json_extract_method": extract_method,
        "schema_errors": schema_errors,
        "normalization_applied": bool(normalization_notes),
        "normalization_notes": normalization_notes,
        "extracted_json": extracted_payload,
        "parsed_json": parsed_payload,
        "raw_prediction_text": raw_output,
        "raw_prediction_truncated": len(text) > RAW_OUTPUT_CHAR_LIMIT,
        "prediction_text": raw_output,
        "error": error,
    }


def _sample_sanity_text(
    *,
    module: Any,
    sampler: Any,
    tokenizer: Any,
    prompt: list[dict[str, str]],
    source_row: dict[str, Any],
    task_type: str,
    timeout: int,
) -> tuple[str, dict[str, Any], Optional[str]]:
    metadata: dict[str, Any] = {
        "known_report_retry_attempted": False,
        "known_report_retry_status": "not_applicable",
        "initial_degenerate_reason": None,
        "retry_degenerate_reason": None,
    }
    text = infer_peb_engineer._sample_text(
        module=module,
        sampler=sampler,
        tokenizer=tokenizer,
        messages=prompt,
        timeout=timeout,
        max_tokens=_sanity_max_tokens(task_type),
        use_json_wrapper=task_type != "known_peptide_report",
    )
    initial_reason = infer_peb_engineer._degenerate_output_reason(text)
    if task_type != "known_peptide_report" or not initial_reason:
        return text, metadata, None

    metadata["known_report_retry_attempted"] = True
    metadata["known_report_retry_status"] = "attempted"
    metadata["initial_degenerate_reason"] = initial_reason
    retry_text = infer_peb_engineer._sample_text(
        module=module,
        sampler=sampler,
        tokenizer=tokenizer,
        messages=infer_peb_engineer._known_report_retry_messages(source_row),
        timeout=timeout,
        max_tokens=KNOWN_REPORT_RETRY_MAX_TOKENS,
        use_json_wrapper=False,
    )
    retry_reason = infer_peb_engineer._degenerate_output_reason(retry_text)
    if retry_reason:
        metadata["known_report_retry_status"] = "still_degenerate"
        metadata["retry_degenerate_reason"] = retry_reason
        return retry_text, metadata, f"degenerate_non_json_output:{retry_reason}"
    metadata["known_report_retry_status"] = "recovered"
    return retry_text, metadata, None


def _summary_result(
    *,
    rows: list[dict[str, Any]],
    holdout_count: int,
    model_name: str,
    source: str,
) -> dict[str, Any]:
    sampled = len(rows)
    valid_json = sum(1 for row in rows if row.get("json_valid"))
    parse_valid_count = sum(1 for row in rows if row.get("parse_valid"))
    schema_valid_count = sum(1 for row in rows if row.get("schema_valid"))
    json_validity = valid_json / sampled if sampled else 0.0
    task_type_counts = Counter(str(row.get("task_type") or "unknown") for row in rows)
    return {
        "status": "completed" if sampled and valid_json == sampled else "completed_with_warnings",
        "attempted": sampled,
        "model_name": model_name,
        "holdout_examples_available": holdout_count,
        "prediction_count": len(rows),
        "sampled_examples": sampled,
        "json_valid_count": valid_json,
        "json_valid_rate": json_validity,
        "json_validity": json_validity,
        "parse_valid_count": parse_valid_count,
        "schema_valid_count": schema_valid_count,
        "schema_adherence": json_validity,
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "validity_by_task_type": _stats_by_task(rows),
        "source": source,
    }


def _write_sanity_outputs(run_dir: Path, rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    eval_dir = run_dir / "eval"
    write_jsonl(eval_dir / "sanity_predictions.jsonl", rows)
    write_json(eval_dir / "sanity_eval_results.json", result)
    upload_dir = run_dir / "hf_upload"
    if upload_dir.exists():
        write_json(upload_dir / "sanity_eval_results.json", result)


def reparse_existing_sanity(
    run_dir: Path,
    model_name: str,
    *,
    task_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    eval_dir = run_dir / "eval"
    existing_rows = read_jsonl(eval_dir / "sanity_predictions.jsonl")
    if task_type:
        existing_rows = [row for row in existing_rows if str(row.get("task_type")) == task_type]
    if limit is not None:
        existing_rows = existing_rows[:limit]
    rows = []
    for index, row in enumerate(existing_rows):
        text = str(row.get("raw_prediction_text") or row.get("prediction_text") or "")
        rows.append(
            _prediction_record(
                index=index,
                source_row=row,
                text=text,
                prompt_hash=str(row.get("prompt_hash") or stable_hash(row)),
                status_if_valid="reparsed",
                error=None,
            )
        )
    result = _summary_result(
        rows=rows,
        holdout_count=len(existing_rows),
        model_name=model_name,
        source="reparsed_existing_predictions",
    )
    _write_sanity_outputs(run_dir, rows, result)
    return result


def sanity_eval(
    run_dir: Path,
    data_dir: Path,
    max_examples: int,
    model_name: str,
    *,
    task_type: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    holdout = read_jsonl(data_dir / "holdout.jsonl")
    if task_type:
        holdout = [row for row in holdout if str(row.get("task_type")) == task_type]
    holdout = holdout[: limit if limit is not None else max_examples]
    existing = _read_json(eval_dir / "sanity_eval_results.json")
    if (
        not force
        and existing.get("status") == "completed"
        and (eval_dir / "sanity_predictions.jsonl").exists()
    ):
        existing["model_name"] = model_name
        write_json(eval_dir / "sanity_eval_results.json", existing)
        return existing

    if not _completed_status(run_manifest.get("status")):
        write_jsonl(eval_dir / "sanity_predictions.jsonl", [])
        result = {
            "status": "not_run",
            "reason": "base_sft_not_completed",
            "model_name": model_name,
            "holdout_examples_available": len(holdout),
            "json_validity": None,
            "schema_adherence": None,
        }
        write_json(eval_dir / "sanity_eval_results.json", result)
        return result

    base_model = str(run_manifest.get("model_name") or infer_peb_engineer.DEFAULT_BASE_MODEL)
    try:
        module, sampler, _sampler_path, tokenizer = infer_peb_engineer._create_sampler(run_dir, base_model)
    except infer_peb_engineer.InferenceBlocked as exc:
        write_jsonl(eval_dir / "sanity_predictions.jsonl", [])
        result = {
            "status": "blocked",
            "reason": exc.reason,
            "details": exc.details,
            "model_name": model_name,
            "holdout_examples_available": len(holdout),
        }
        write_json(eval_dir / "sanity_eval_results.json", result)
        return result

    rows = []
    sampled = 0
    for index, row in enumerate(holdout):
        prompt = _prompt_messages(row)
        task_type = str(row.get("task_type") or "unknown")
        if prompt:
            prompt = infer_peb_engineer._apply_task_output_contract(prompt, task_type=task_type)
        if not prompt:
            rows.append(
                {
                    "index": index,
                    "example_id": row.get("example_id"),
                    "task_type": task_type,
                    "prompt_hash": stable_hash(row),
                    "status": "skipped_malformed_prompt",
                    "json_valid": False,
                    "parse_valid": False,
                    "schema_valid": False,
                }
            )
            continue
        text = ""
        error = None
        sample_metadata: dict[str, Any] = {}
        sampled += 1
        try:
            text, sample_metadata, error = _sample_sanity_text(
                module=module,
                sampler=sampler,
                tokenizer=tokenizer,
                prompt=prompt,
                source_row=row,
                task_type=task_type,
                timeout=timeout,
            )
        except Exception as exc:
            error = infer_peb_engineer._sanitize_error(exc)
        record = _prediction_record(
            index=index,
            source_row=row,
            text=text,
            prompt_hash=stable_hash(prompt),
            error=error,
        )
        record.update(sample_metadata)
        rows.append(record)
    result = _summary_result(
        rows=rows,
        holdout_count=len(holdout),
        model_name=model_name,
        source="remote_sampler",
    )
    _write_sanity_outputs(run_dir, rows, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("training/data/sft"))
    parser.add_argument("--max-examples", type=int, default=80)
    parser.add_argument("--model-name", default="the-spice")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reparse-existing", action="store_true")
    parser.add_argument("--from-existing-raw", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--task-type")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.reparse_existing or args.from_existing_raw:
        result = reparse_existing_sanity(
            args.run_dir,
            args.model_name,
            task_type=args.task_type,
            limit=args.limit,
        )
    else:
        result = sanity_eval(
            args.run_dir,
            args.data_dir,
            args.max_examples,
            args.model_name,
            task_type=args.task_type,
            limit=args.limit,
            force=args.force_refresh or not args.resume,
            timeout=args.timeout,
        )
    print(f"Sanity eval status={result['status']}")


if __name__ == "__main__":
    main()
