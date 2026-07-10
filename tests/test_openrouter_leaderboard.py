import json
import shutil
from pathlib import Path

from peb.io import read_jsonl, write_jsonl
from peb.openrouter_leaderboard import (
    OpenRouterError,
    OpenRouterInsufficientCredits,
    _compact_batch_payload_to_predictions,
    _compact_payload_to_prediction,
    _prediction_records_from_response,
    _replace_prediction_record,
    _retry_single_with_attempts,
    build_openrouter_retry_queue,
    finalize_strict_leaderboard,
    inspect_leaderboard_input,
    leaderboard_check,
    normalize_leaderboard_rows,
    openrouter_retry_failures,
    public_leaderboard_rows,
    recompute_openrouter_leaderboard_from_predictions,
    safe_model_slug,
    sanitize_case_for_leaderboard,
    select_models,
    write_static_leaderboard_page,
)

RELEASE_DIR = Path("data/releases/peb-v1.0-rc")
OPENROUTER_ENV_NAME = "_".join(("OPENROUTER", "API", "KEY"))


def _first_case(track):
    return read_jsonl(RELEASE_DIR / track / "cases.jsonl")[0]


def _leaderboard_with_missing_retry_target(tmp_path):
    board = tmp_path / "leaderboard_with_retry_target"
    shutil.copytree("leaderboard", board)
    rows = json.loads((board / "leaderboard.json").read_text(encoding="utf-8"))
    for item in rows:
        item["api_error_count"] = 0
        item["invalid_json_count"] = 0
        coverage = item.get("coverage")
        if isinstance(coverage, dict):
            for value in coverage.values():
                if isinstance(value, dict):
                    value["completed"] = value.get("attempted", value.get("completed", 0))
    row = next(
        item
        for item in rows
        if not item.get("is_baseline")
        and item.get("mode") == "base"
        and "binding_rank" in (item.get("coverage") or {})
    )
    row["api_error_count"] = 1
    row["invalid_json_count"] = 0
    cases = read_jsonl(RELEASE_DIR / "binding_rank" / "cases.jsonl")
    row["coverage"] = {"binding_rank": {"attempted": len(cases), "completed": len(cases)}}
    slug = safe_model_slug(row["model_id"])
    pred_path = board / "predictions" / slug / "base_binding_rank.jsonl"
    predictions = [
        {
            "prediction_id": f"test:{case['benchmark_id']}",
            "benchmark_id": case["benchmark_id"],
            "track": "binding_rank",
            "model_name": row["model_id"],
            "scores": [
                {"item_id": item["item_id"], "score": float(index + 1), "rank": index + 1}
                for index, item in enumerate(case["items"])
            ],
        }
        for case in cases
    ]
    target_id = predictions[0]["benchmark_id"]
    write_jsonl(pred_path, predictions[1:])
    (board / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return board, row["model_id"], target_id


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_human_effect_sanitized_input_excludes_answer_fields():
    sanitized = sanitize_case_for_leaderboard(_first_case("human_effect"), "base")

    keys = set(_walk_keys(sanitized))

    assert "category" not in keys
    assert "evidence_level" not in keys
    assert "claim_status" not in keys
    assert "safety_status" not in keys
    assert "claim_text" in keys


def test_binding_rank_sanitized_input_keeps_candidates_without_measurements():
    sanitized = sanitize_case_for_leaderboard(_first_case("binding_rank"), "base")

    keys = set(_walk_keys(sanitized))

    assert "candidate_peptides" in keys
    assert "measured_value" not in keys
    assert "normalized_rank" not in keys
    assert "measurement_direction" in keys


def test_pose_sanitized_input_excludes_contact_answers():
    sanitized = sanitize_case_for_leaderboard(_first_case("pose"), "tools_high_reasoning")

    keys = set(_walk_keys(sanitized))

    assert "native_contacts" not in keys
    assert "binding_site_residues" not in keys
    assert sanitized["contact_label_status"] == "computed_contacts"


def test_inspect_leaderboard_input_writes_jsonl(tmp_path):
    output = tmp_path / "human_effect_base.jsonl"

    count = inspect_leaderboard_input(RELEASE_DIR, "human_effect", "base", output)
    rows = read_jsonl(output)

    assert count == 200
    assert len(rows) == 200
    assert "category" not in set(_walk_keys(rows[0]))


def test_select_models_uses_live_metadata_shape_without_aliases():
    models = [
        {
            "id": "openrouter/auto",
            "name": "Auto",
            "created": 9,
            "context_length": 100000,
            "supported_parameters": ["response_format"],
        },
        {
            "id": "openai/gpt-current",
            "name": "GPT Current",
            "created": 10,
            "context_length": 200000,
            "supported_parameters": ["response_format"],
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "anthropic/claude-sonnet-current",
            "name": "Claude Sonnet Current",
            "created": 11,
            "context_length": 200000,
            "supported_parameters": ["response_format"],
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
    ]

    selected = select_models(models)

    assert [item["model_id"] for item in selected][:2] == [
        "openai/gpt-current",
        "anthropic/claude-sonnet-current",
    ]
    assert all(item["model_id"] != "openrouter/auto" for item in selected)


def test_safe_model_slug_is_filesystem_safe():
    assert safe_model_slug("openai/gpt-5:stable") == "openai_gpt-5_stable"


def test_public_leaderboard_rows_surface_scores_and_status():
    rows = json.loads(Path("leaderboard/leaderboard.json").read_text(encoding="utf-8"))
    dirty_row = dict(rows[0])
    dirty_row.update(
        {
            "model_id": "synthetic/error-row",
            "api_error_count": 1,
            "invalid_json_count": 0,
            "competitive": True,
            "is_baseline": False,
        }
    )

    public_rows = public_leaderboard_rows([*rows, dirty_row])

    assert public_rows
    assert all("coverage_adjusted_score" in row for row in public_rows)
    assert any("errors" in row["status"] for row in public_rows)
    assert all(
        row["row_status"] in {"clean_completed", "completed_with_failures"}
        for row in public_rows
        if row["rank"] is not None
    )
    sanity = next(row for row in public_rows if row["model_id"] == "human_effect_oracle_sanity_check")
    assert sanity["competitive"] is False
    assert "non-competitive" in sanity["status"]
    assert sanity["rank"] is None


def test_accounted_failed_predictions_remain_scored_with_failure_status():
    row = {
        "model_id": "model/with-explicit-failures",
        "mode": "base",
        "completed_tracks": ["human_effect"],
        "coverage": {"human_effect": {"attempted": 10, "completed": 10}},
        "mean_score": 0.5,
        "invalid_json_count": 1,
        "unresolved_invalid_json_count": 1,
        "failed_prediction_count": 1,
        "valid_prediction_rate": 0.9,
        "api_error_count": 0,
        "fallback_prediction_count": 0,
        "unresolved_provider_error_count": 0,
        "competitive": True,
        "is_baseline": False,
    }

    [normalized] = normalize_leaderboard_rows([row])

    assert normalized["row_status"] == "completed_with_failures"
    assert normalized["scored"] is True
    assert normalized["competitive"] is True
    assert normalized["coverage_adjusted_score"] == 0.45
    assert "completed_with_failures" in normalized["status"]
    assert "errors" in normalized["status"]


def test_static_leaderboard_page_writes_clean_chart_data(tmp_path):
    board = tmp_path / "leaderboard"
    board.mkdir()
    shutil.copy2("leaderboard/leaderboard.json", board / "leaderboard.json")

    rows = write_static_leaderboard_page(board)
    payload = (board / "leaderboard_public.json").read_text(encoding="utf-8")

    source_rows = json.loads((board / "leaderboard.json").read_text(encoding="utf-8"))
    assert len(rows) == len(source_rows)
    assert (board / "index.html").exists()
    assert (board / "assets" / "leaderboard.js").exists()
    assert "raw_prompt" not in payload
    assert "prediction_text" not in payload
    assert "/" + "Users/" not in payload
    assert "sk-or-" + "v1" not in payload


def test_retry_queue_only_targets_fallback_or_missing_predictions(tmp_path):
    board, model_id, target_id = _leaderboard_with_missing_retry_target(tmp_path)
    queue = build_openrouter_retry_queue(
        leaderboard_dir=board,
        release_dir=RELEASE_DIR,
        models_config=board / "openrouter_models.json",
    )

    assert len(queue) == 1
    assert queue[0]["model_id"] == model_id
    assert queue[0]["benchmark_id"] == target_id
    assert all("retry_status" in item for item in queue)
    assert all("benchmark_id" in item for item in queue)
    assert not any("gold" in key for item in queue for key in item)
    assert "sk-or-" + "v1" not in json.dumps(queue)


def test_replace_prediction_record_updates_matching_case(tmp_path):
    path = tmp_path / "predictions.jsonl"
    original = [
        {"benchmark_id": "a", "track": "structure", "model_name": "m", "coordinates": []},
        {"benchmark_id": "b", "track": "structure", "model_name": "m", "coordinates": []},
    ]
    from peb.io import write_jsonl

    write_jsonl(path, original)
    replacement = {
        "benchmark_id": "b",
        "track": "structure",
        "model_name": "m",
        "coordinates": [{"atom_name": "CA", "residue_index": 1, "x": 1.0, "y": 2.0, "z": 3.0}],
    }

    assert _replace_prediction_record(path, replacement) is True
    rows = read_jsonl(path)

    assert len(rows) == 2
    assert rows[1] == replacement


def test_pose_prediction_accepts_contacts_alias():
    records = _prediction_records_from_response(
        {
            "predictions": [
                {
                    "benchmark_id": "PEB-POSE-00001",
                    "track": "pose",
                    "prediction_id": "p1",
                    "contacts": [],
                    "binding_site_residues": [],
                }
            ]
        },
        track="pose",
        model_id="model",
        expected_ids={"PEB-POSE-00001"},
    )

    assert records[0]["predicted_contacts"] == []
    assert "contacts" not in records[0]


def test_binding_rank_prediction_accepts_flat_candidate_rows():
    records = _prediction_records_from_response(
        {
            "predictions": [
                {
                    "benchmark_id": "PEB-BIND-00001",
                    "track": "binding_rank",
                    "item_id": "cand-1",
                    "score": 0.8,
                    "rank": 1,
                },
                {
                    "benchmark_id": "PEB-BIND-00001",
                    "track": "binding_rank",
                    "item_id": "cand-2",
                    "score": 0.2,
                    "rank": 2,
                },
            ]
        },
        track="binding_rank",
        model_id="model",
        expected_ids={"PEB-BIND-00001"},
    )

    assert len(records) == 1
    assert records[0]["benchmark_id"] == "PEB-BIND-00001"
    assert len(records[0]["scores"]) == 2


def test_binding_rank_prediction_repairs_numeric_score_list():
    case = _first_case("binding_rank")
    expected_ids = {case["benchmark_id"]}
    records = _prediction_records_from_response(
        {
            "predictions": [
                {
                    "benchmark_id": case["benchmark_id"],
                    "track": "binding_rank",
                    "prediction_id": "p1",
                    "scores": [1.0 for _item in case["items"]] + [0.0],
                    "ranks": list(range(1, len(case["items"]) + 1)),
                }
            ]
        },
        track="binding_rank",
        model_id="model",
        expected_ids=expected_ids,
        cases_by_id={case["benchmark_id"]: case},
    )

    assert len(records) == 1
    assert [score["item_id"] for score in records[0]["scores"]] == [
        item["item_id"] for item in case["items"]
    ]


def test_compact_binding_rank_payload_converts_to_prediction():
    case = _first_case("binding_rank")
    record = _compact_payload_to_prediction(
        {"scores": [0.5 for _item in case["items"]]},
        track="binding_rank",
        model_id="model",
        case=case,
    )

    assert record["benchmark_id"] == case["benchmark_id"]
    assert [score["item_id"] for score in record["scores"]] == [
        item["item_id"] for item in case["items"]
    ]


def test_compact_pose_payload_clamps_negative_clash_score():
    case = _first_case("pose")
    record = _compact_payload_to_prediction(
        {
            "predicted_contacts": [],
            "binding_site_residues": [],
            "orientation_label": "unknown",
            "clash_score": -1,
        },
        track="pose",
        model_id="model",
        case=case,
    )

    assert record["clash_score"] == 0.0


def test_google_compact_retry_retries_malformed_json(monkeypatch):
    case = _first_case("pose")
    calls = {"count": 0}

    def fake_chat_completion(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            content = "{\"predicted_contacts\":"
        else:
            content = json.dumps(
                {
                    "predicted_contacts": [],
                    "binding_site_residues": [],
                    "orientation_label": "unknown",
                    "clash_score": 0,
                }
            )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("peb.openrouter_leaderboard._chat_completion", fake_chat_completion)
    monkeypatch.setattr("peb.openrouter_leaderboard.time.sleep", lambda _seconds: None)

    prediction, category, message, attempts = _retry_single_with_attempts(
        model={"model_id": "google/gemini-2.5-pro"},
        mode="tools_high_reasoning",
        track="pose",
        case=case,
        timeout=300,
        retries=1,
        use_response_format=False,
    )

    assert prediction is not None
    assert category == ""
    assert message == ""
    assert attempts == 2


def test_compact_structure_empty_coordinates_are_not_retry_fallback_shape():
    case = _first_case("structure")
    record = _compact_payload_to_prediction(
        {"coordinates": [], "confidence": 0},
        track="structure",
        model_id="model",
        case=case,
    )

    assert record["coordinates"] == []
    assert record["confidence"] == 0.01


def test_compact_human_effect_batch_payload_converts_to_predictions():
    cases = [read_jsonl(RELEASE_DIR / "human_effect" / "cases.jsonl")[0]]
    record = _compact_batch_payload_to_predictions(
        {
            "predictions": [
                {
                    "benchmark_id": cases[0]["benchmark_id"],
                    "category": "no_known_human_effect_evidence",
                    "evidence_level": "unsupported_contradicted_or_unsafe_claim",
                    "evidence_direction": "not_applicable",
                    "claim_status": "insufficient_information",
                    "safety_status": "insufficient_safety_data",
                }
            ]
        },
        track="human_effect",
        model_id="model",
        cases=cases,
    )

    assert len(record) == 1
    assert record[0]["benchmark_id"] == cases[0]["benchmark_id"]


def test_recompute_leaderboard_records_post_retry_counts(tmp_path):
    board = tmp_path / "leaderboard"
    shutil.copytree("leaderboard", board)
    retry_results = [
        {
            "model_id": "anthropic/claude-sonnet-5",
            "mode": "base",
            "track": "binding_rank",
            "benchmark_id": "PEB-BIND-00001",
            "retry_status": "still_failed",
            "error_category": "provider_error_unresolved",
        },
        {
            "model_id": "anthropic/claude-sonnet-5",
            "mode": "base",
            "track": "binding_rank",
            "benchmark_id": "PEB-BIND-00002",
            "retry_status": "recovered",
            "error_category": None,
        },
    ]

    manifest = recompute_openrouter_leaderboard_from_predictions(
        leaderboard_dir=board,
        release_dir=RELEASE_DIR,
        models_config=board / "openrouter_models.json",
        retry_results=retry_results,
    )

    assert manifest["retry_queue_size"] == 2
    assert manifest["retry_recovered_count"] == 1
    assert manifest["retry_still_failed_count"] == 1
    assert manifest["post_retry_api_error_count"] == 1


def test_retry_without_api_key_writes_queue_but_does_not_mark_failures(tmp_path, monkeypatch):
    board, _model_id, _target_id = _leaderboard_with_missing_retry_target(tmp_path)
    output = tmp_path / "leaderboard"
    monkeypatch.delenv(OPENROUTER_ENV_NAME, raising=False)

    try:
        openrouter_retry_failures(
            leaderboard_dir=board,
            release_dir=RELEASE_DIR,
            models_config=board / "openrouter_models.json",
            output_dir=output,
            timeout=300,
            retries=5,
            max_concurrency=1,
            resume=True,
        )
    except OpenRouterError:
        pass
    else:
        raise AssertionError("missing API key should fail before live retry")

    retry_dir = output / "retry"
    summary = json.loads((retry_dir / "retry_summary.json").read_text(encoding="utf-8"))
    assert summary["retry_queue_size"] > 0
    assert summary["live_retry_attempted"] is False
    assert summary["retry_still_failed_count"] == 0
    assert read_jsonl(retry_dir / "retry_results.jsonl") == []


def test_retry_insufficient_credits_does_not_mark_case_failures(tmp_path, monkeypatch):
    board, _model_id, _target_id = _leaderboard_with_missing_retry_target(tmp_path)
    output = tmp_path / "leaderboard"
    monkeypatch.setenv(OPENROUTER_ENV_NAME, "test-key")

    def fail_for_credit(*args, **kwargs):
        del args, kwargs
        raise OpenRouterInsufficientCredits("OpenRouter HTTP 402: insufficient credits")

    monkeypatch.setattr("peb.openrouter_leaderboard._chat_completion", fail_for_credit)

    try:
        openrouter_retry_failures(
            leaderboard_dir=board,
            release_dir=RELEASE_DIR,
            models_config=board / "openrouter_models.json",
            output_dir=output,
            timeout=300,
            retries=5,
            max_concurrency=1,
            resume=True,
        )
    except OpenRouterInsufficientCredits:
        pass
    else:
        raise AssertionError("insufficient credits should abort live retry")

    assert read_jsonl(output / "retry" / "retry_results.jsonl") == []


def test_finalize_strict_leaderboard_excludes_dirty_rows(tmp_path):
    board = tmp_path / "leaderboard"
    shutil.copytree("leaderboard", board)

    summary = finalize_strict_leaderboard(board)
    rows = json.loads((board / "leaderboard.json").read_text(encoding="utf-8"))
    public_rows = json.loads((board / "leaderboard_public.json").read_text(encoding="utf-8"))

    assert summary["competitive_rows"] >= 1
    assert (board / "excluded_models.json").exists()
    assert (board / "diagnostics" / "error_diagnosis.json").exists()
    assert all(
        row["api_error_count"] == 0
        and row["fallback_prediction_count"] == 0
        and row["unresolved_provider_error_count"] == 0
        and row["row_status"] in {"clean_completed", "completed_with_failures"}
        and row["scored"] is True
        for row in rows
        if row.get("competitive") is True
    )
    assert all(row["competitive"] is True for row in public_rows if row["rank"] is not None)


def test_leaderboard_check_fails_on_competitive_error_row(tmp_path):
    board = tmp_path / "leaderboard"
    (board / "predictions" / "bad" ).mkdir(parents=True)
    (board / "results" / "bad").mkdir(parents=True)
    row = {
        "model_id": "bad",
        "mode": "base",
        "completed_tracks": ["human_effect"],
        "coverage": {"human_effect": {"attempted": 1, "completed": 1}},
        "competitive": True,
        "scored": True,
        "row_status": "clean_completed",
        "api_error_count": 1,
        "invalid_json_count": 0,
    }
    (board / "leaderboard.json").write_text(json.dumps([row]), encoding="utf-8")

    passed, errors, _summary = leaderboard_check(board)

    assert passed is False
    assert any("api_error_count" in error for error in errors)


def test_leaderboard_check_does_not_require_artifacts_for_excluded_rows(tmp_path):
    board = tmp_path / "leaderboard"
    row = {
        "model_id": "excluded",
        "mode": "base",
        "completed_tracks": ["human_effect"],
        "coverage": {"human_effect": {"attempted": 1, "completed": 0}},
        "competitive": True,
        "scored": False,
        "row_status": "excluded_incomplete",
        "api_error_count": 0,
        "invalid_json_count": 0,
        "fallback_prediction_count": 0,
        "unresolved_provider_error_count": 0,
        "unresolved_invalid_json_count": 0,
    }
    (board / "leaderboard.json").parent.mkdir(parents=True)
    (board / "leaderboard.json").write_text(json.dumps([row]), encoding="utf-8")

    passed, errors, summary = leaderboard_check(board)

    assert passed is False
    assert summary["competitive_rows"] == 0
    assert errors == ["no rankable competitive rows"]


def test_leaderboard_check_passes_on_clean_mock(tmp_path):
    board = tmp_path / "leaderboard"
    (board / "predictions" / "clean").mkdir(parents=True)
    (board / "results" / "clean").mkdir(parents=True)
    from peb.io import write_jsonl

    write_jsonl(
        board / "predictions" / "clean" / "base_human_effect.jsonl",
        [
            {
                "prediction_id": "p1",
                "benchmark_id": "case-1",
                "track": "human_effect",
                "model_name": "clean",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "unsupported_contradicted_or_unsafe_claim",
                "evidence_direction": "not_applicable",
                "claim_status": "insufficient_information",
                "safety_status": "insufficient_safety_data",
                "abstained": False,
            }
        ],
    )
    (board / "results" / "clean" / "base_human_effect.json").write_text("{}", encoding="utf-8")
    row = {
        "model_id": "clean",
        "mode": "base",
        "completed_tracks": ["human_effect"],
        "coverage": {"human_effect": {"attempted": 1, "completed": 1}},
        "competitive": True,
        "scored": True,
        "row_status": "clean_completed",
        "api_error_count": 0,
        "invalid_json_count": 0,
        "fallback_prediction_count": 0,
        "unresolved_provider_error_count": 0,
        "unresolved_invalid_json_count": 0,
    }
    (board / "leaderboard.json").write_text(json.dumps([row]), encoding="utf-8")

    passed, errors, summary = leaderboard_check(board)

    assert passed is True
    assert errors == []
    assert summary["competitive_rows"] == 1


def test_leaderboard_check_passes_on_accounted_failed_predictions(tmp_path):
    board = tmp_path / "leaderboard"
    (board / "predictions" / "accounted").mkdir(parents=True)
    (board / "results" / "accounted").mkdir(parents=True)
    from peb.io import write_jsonl

    write_jsonl(
        board / "predictions" / "accounted" / "base_human_effect.jsonl",
        [
            {
                "prediction_id": "p1",
                "benchmark_id": "case-1",
                "track": "human_effect",
                "model_name": "accounted",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "unsupported_contradicted_or_unsafe_claim",
                "evidence_direction": "not_applicable",
                "claim_status": "insufficient_information",
                "safety_status": "insufficient_safety_data",
                "abstained": True,
                "status": "failed",
                "json_valid": False,
                "schema_valid": False,
                "error_type": "unresolved_invalid_json",
            }
        ],
    )
    (board / "results" / "accounted" / "base_human_effect.json").write_text("{}", encoding="utf-8")
    row = {
        "model_id": "accounted",
        "mode": "base",
        "completed_tracks": ["human_effect"],
        "coverage": {"human_effect": {"attempted": 1, "completed": 1}},
        "competitive": True,
        "scored": True,
        "row_status": "completed_with_failures",
        "api_error_count": 0,
        "invalid_json_count": 1,
        "fallback_prediction_count": 0,
        "unresolved_provider_error_count": 0,
        "unresolved_invalid_json_count": 1,
        "failed_prediction_count": 1,
        "valid_prediction_rate": 0.0,
        "mean_score": 0.0,
    }
    (board / "leaderboard.json").write_text(json.dumps([row]), encoding="utf-8")

    passed, errors, summary = leaderboard_check(board)

    assert passed is True
    assert errors == []
    assert summary["competitive_rows"] == 1
