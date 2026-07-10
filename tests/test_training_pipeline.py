from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "scripts"))

import eval_peb_engineer
import infer_peb_engineer
import sanity_eval_peb_engineer
from check_training_leakage import check_leakage
from eval_peb_engineer import SCOREABLE_MANIFEST_STATUSES, evaluate
from prepare_hf_upload import prepare_upload
from sanity_eval_peb_engineer import sanity_eval
from split_train_dev_holdout import _cost_estimate
from train_tinker_sft import launch_training
from training_common import TINKER_ENV_NAME, strict_response_template

from peb.metrics.evidence_metrics import evaluate_human_effect
from peb.schemas import HumanEffectCase, HumanEffectPrediction


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _example(benchmark_id: str, source_split: str, user_text: str = "safe prompt") -> dict:
    return {
        "example_id": f"ex-{benchmark_id}",
        "benchmark_id": benchmark_id,
        "source_split": source_split,
        "task_type": "known_peptide_report",
        "messages": [
            {"role": "system", "content": "return json"},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "{}"},
        ],
        "token_estimate": 100,
    }


class _FakeFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _FakeOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        del tokenize
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self.encode(text)

    def encode(self, text):
        return [ord(char) % 251 + 1 for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del tokens, skip_special_tokens
        return "{}"


class _FakeModelInput:
    def __init__(self, tokens):
        self.tokens = tokens
        self.length = len(tokens)

    @classmethod
    def from_ints(cls, tokens):
        return cls(tokens)


class _FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class _FakeAdamParams:
    def __init__(self, learning_rate=0.0001, beta1=0.9, beta2=0.95, eps=1e-12):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTrainingClient:
    calls = []

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        self.calls.append("get_tokenizer")
        return self.tokenizer

    def forward_backward(self, batch, loss_fn):
        self.calls.append(("forward_backward", len(batch), loss_fn))
        return _FakeFuture(_FakeOutput(metrics={"loss": 0.25}))

    def optim_step(self, adam_params):
        self.calls.append(("optim_step", adam_params.learning_rate))
        return _FakeFuture(_FakeOutput(metrics={"optimizer": 1.0}))

    def save_state(self, name):
        self.calls.append(("save_state", name))
        return _FakeFuture(_FakeOutput(path=f"remote://{name}"))

    def save_weights_for_sampler(self, name):
        self.calls.append(("save_weights_for_sampler", name))
        return _FakeFuture(_FakeOutput(path=f"remote://{name}"))

    def create_sampling_client(self, model_path):
        self.calls.append(("create_sampling_client", model_path))
        return _FakeSamplingClient()


class _FakeSamplingClient:
    def sample(self, prompt, num_samples, sampling_params):
        del prompt, num_samples, sampling_params
        return _FakeFuture(_FakeOutput(sequences=[_FakeOutput(tokens=[123, 125])]))


class _FakeServiceClient:
    calls = []

    def get_server_capabilities(self):
        return _FakeOutput(supported_models=[_FakeOutput(model_name="openai/gpt-oss-20b")])

    def create_lora_training_client(self, base_model, rank=32, seed=None):
        self.calls.append(("create_lora_training_client", base_model, rank, seed))
        return _FakeTrainingClient()

    def create_training_client_from_state(self, path):
        self.calls.append(("create_training_client_from_state", path))
        return _FakeTrainingClient()


def _fake_tinker_module():
    return types.SimpleNamespace(
        ServiceClient=_FakeServiceClient,
        Datum=_FakeDatum,
        ModelInput=_FakeModelInput,
        AdamParams=_FakeAdamParams,
        SamplingParams=_FakeSamplingParams,
    )


def _write_training_fixture(root: Path, rows: Optional[list[dict]] = None) -> tuple[Path, Path, Path, Path]:
    config = root / "config.yaml"
    data = root / "data"
    artifacts = root / "artifacts"
    run = root / "run"
    config.write_text(
        "\n".join(
            [
                "model_name: GPT-OSS-20B",
                "run_name: test-peb-engineer",
                "lora_rank: 32",
                "learning_rate: 0.0001",
                "max_seq_len: 256",
                "seed: 42",
                "sanity_eval_examples: 2",
            ]
        ),
        encoding="utf-8",
    )
    train_rows = rows if rows is not None else [_example("PEB-1", "train"), _example("PEB-2", "train")]
    _write_jsonl(data / "train.jsonl", train_rows)
    _write_jsonl(data / "dev.jsonl", [_example("PEB-3", "dev")])
    _write_jsonl(data / "holdout.jsonl", [_example("PEB-4", "dev")])
    artifacts.mkdir(parents=True)
    (artifacts / "leakage_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (artifacts / "cost_estimate.json").write_text(
        json.dumps(
            {
                "under_hard_cap": True,
                "total_estimated_cost_usd": 1.0,
                "train_token_estimate": 1000,
                "training_expansion_factor": 1,
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "dataset_report.json").write_text(json.dumps({"sft_examples": len(train_rows)}), encoding="utf-8")
    (artifacts / "source_coverage.json").write_text(json.dumps({"total_cases": 1}), encoding="utf-8")
    return config, data, artifacts, run


def _human_prediction(benchmark_id: str) -> dict:
    return {
        "prediction_id": f"the_spice:{benchmark_id}",
        "benchmark_id": benchmark_id,
        "track": "human_effect",
        "model_name": "the-spice",
        "category": "no_known_human_effect_evidence",
        "evidence_level": "unsupported_contradicted_or_unsafe_claim",
        "evidence_direction": "not_applicable",
        "claim_status": "insufficient_information",
        "safety_status": "insufficient_safety_data",
        "abstained": True,
        "rationale_source_ids": [],
    }


def _binding_prediction(benchmark_id: str) -> dict:
    return {
        "prediction_id": f"the_spice:{benchmark_id}",
        "benchmark_id": benchmark_id,
        "track": "binding_rank",
        "model_name": "the-spice",
        "scores": [{"item_id": "item-1", "score": 0.0, "rank": 1}],
    }


def _write_resume_fixture(
    root: Path,
    *,
    human_existing_count: int = 190,
    duplicate_existing: bool = False,
) -> tuple[Path, Path, Path, Path, list[str]]:
    release = root / "release"
    output = root / "run" / "eval" / "peb_predictions"
    leaderboard = root / "leaderboard"
    run_dir = root / "run"
    human_ids = [f"PEB-HFX-{index:05d}" for index in range(1, 201)]
    binding_ids = [f"PEB-BIND-{index:05d}" for index in range(1, 26)]
    _write_jsonl(
        release / "human_effect" / "cases.jsonl",
        [{"benchmark_id": benchmark_id, "track": "human_effect"} for benchmark_id in human_ids],
    )
    _write_jsonl(
        release / "binding_rank" / "cases.jsonl",
        [
            {
                "benchmark_id": benchmark_id,
                "track": "binding_rank",
                "items": [{"item_id": "item-1"}],
            }
            for benchmark_id in binding_ids
        ],
    )
    existing_human = [_human_prediction(benchmark_id) for benchmark_id in human_ids[:human_existing_count]]
    if duplicate_existing and existing_human:
        existing_human.append(dict(existing_human[-1]))
    _write_jsonl(output / "base_human_effect.jsonl", existing_human)
    _write_jsonl(output / "base_binding_rank.jsonl", [_binding_prediction(item) for item in binding_ids])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "training_completed", "model_name": "openai/gpt-oss-20b"}),
        encoding="utf-8",
    )
    (run_dir / "hf_upload").mkdir(parents=True, exist_ok=True)
    (run_dir / "hf_upload" / "adapter_export_manifest.json").write_text(
        json.dumps({"status": "remote_export_available", "remote_sampler_path": "tinker://test/sampler"}),
        encoding="utf-8",
    )
    return release, output, leaderboard, run_dir, human_ids[human_existing_count:]


def test_strict_response_template_contains_training_sections() -> None:
    payload = strict_response_template()
    assert "known_source_backed_facts" in payload
    assert "pose_contact_assessment" in payload
    assert "target_binding" in payload
    assert "human_effect_estimate" in payload


def test_extracts_first_balanced_json_object() -> None:
    result = infer_peb_engineer._extract_json_payload_with_metadata(
        'prefix <json>{"outer":{"inner":1},"items":[1,2]}</json> suffix'
    )
    assert result["payload"] == {"outer": {"inner": 1}, "items": [1, 2]}
    assert result["method"] == "tagged_json:full_json"


def test_extracts_json_before_trailing_junk() -> None:
    result = infer_peb_engineer._extract_json_payload_with_metadata(
        '{"ok":true,"nested":{"value":"}"}} trailing text {"ignored":true}'
    )
    assert result["payload"] == {"ok": True, "nested": {"value": "}"}}


def test_extracts_json_before_repeated_fields_after_closing_brace() -> None:
    result = infer_peb_engineer._extract_json_payload_with_metadata(
        '{"unsupported_claims":[]}, "unsupported_claims":[]}, "unsupported_claims":[]}'
    )
    assert result["payload"] == {"unsupported_claims": []}


def test_truncated_json_is_invalid() -> None:
    with pytest.raises(json.JSONDecodeError):
        infer_peb_engineer._extract_json_payload_with_metadata('{"ok": true, "items": [1, 2]')


def test_noinput_output_is_degenerate() -> None:
    assert infer_peb_engineer._degenerate_output_reason("<noinput provided.>") == "noinput_output"


def test_repeated_struct_pose_loop_is_degenerate() -> None:
    loop = "<no-" + "-".join(
        f"STRUCT-{157 + index % 20:05d}-POSE-00022" for index in range(80)
    )
    assert infer_peb_engineer._degenerate_output_reason(loop) == "repeated_struct_pose_identifier_loop"


def test_degenerate_known_output_triggers_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    valid = json.dumps(strict_response_template(), sort_keys=True)
    calls = []

    def fake_sample_text(**kwargs):
        calls.append(kwargs)
        return "<noinput provided.>" if len(calls) == 1 else valid

    monkeypatch.setattr(infer_peb_engineer, "_sample_text", fake_sample_text)
    text, metadata, error = sanity_eval_peb_engineer._sample_sanity_text(
        module=object(),
        sampler=object(),
        tokenizer=object(),
        prompt=[{"role": "user", "content": "known report"}],
        source_row={"benchmark_id": "PEB-STRUCT-00005", "source_id": "3G6N_2_F"},
        task_type="known_peptide_report",
        timeout=1,
    )
    assert text == valid
    assert error is None
    assert metadata["known_report_retry_attempted"] is True
    assert metadata["known_report_retry_status"] == "recovered"
    assert metadata["initial_degenerate_reason"] == "noinput_output"
    assert calls[0]["use_json_wrapper"] is False
    assert calls[1]["max_tokens"] == sanity_eval_peb_engineer.KNOWN_REPORT_RETRY_MAX_TOKENS


def test_degenerate_known_output_after_retry_remains_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(
        [
            "<noinput provided.>",
            "<no-STRUCT-00157-POSE-00022-STRUCT-00157-POSE-00022" * 20,
        ]
    )

    def fake_sample_text(**kwargs):
        del kwargs
        return next(outputs)

    monkeypatch.setattr(infer_peb_engineer, "_sample_text", fake_sample_text)
    text, metadata, error = sanity_eval_peb_engineer._sample_sanity_text(
        module=object(),
        sampler=object(),
        tokenizer=object(),
        prompt=[{"role": "user", "content": "known report"}],
        source_row={"benchmark_id": "PEB-STRUCT-00026", "source_id": "1VTP_1_A"},
        task_type="known_peptide_report",
        timeout=1,
    )
    record = sanity_eval_peb_engineer._prediction_record(
        index=0,
        source_row={"task_type": "known_peptide_report", "benchmark_id": "PEB-STRUCT-00026"},
        text=text,
        prompt_hash="prompt",
        error=error,
    )
    record.update(metadata)
    assert record["json_valid"] is False
    assert record["parse_valid"] is False
    assert record["parsed_json"] is None
    assert record["normalization_applied"] is False
    assert record["known_report_retry_status"] == "still_degenerate"
    assert str(record["error"]).startswith("degenerate_non_json_output:")


def test_pose_contacts_are_capped() -> None:
    contacts = [
        {"target_residue": f"A:{index}", "peptide_residue": f"B:{index}"}
        for index in range(100)
    ]
    cleaned = infer_peb_engineer._clean_contacts(contacts)
    assert len(cleaned) == infer_peb_engineer.MAX_PREDICTED_CONTACTS
    assert cleaned[-1] == {"target_residue": "A:49", "peptide_residue": "B:49"}


def test_pose_missing_assessment_normalizes_safely() -> None:
    payload, notes = infer_peb_engineer._normalise_report_payload({"overall_confidence": "high"})
    assert "pose_contact_assessment_defaulted" in notes
    assert payload["overall_confidence"] == "high"
    assert payload["pose_contact_assessment"] == {
        "status": "not_assessed",
        "confidence": "low",
        "interface_residues": [],
        "predicted_contacts": [],
        "notes": "",
    }
    assert payload["known_source_backed_facts"] == []


def test_developability_only_object_gets_full_empty_skeleton() -> None:
    payload, notes = infer_peb_engineer._normalise_report_payload(
        {
            "stability_risk": "low",
            "solubility_risk": "medium",
            "toxicity_risk": "low",
            "hemolysis_risk": "low",
            "cytotoxicity_risk": "low",
            "synthesis_complexity": "low",
        }
    )
    assert "top_level_developability_wrapped" in notes
    assert set(strict_response_template()) <= set(payload)
    assert payload["developability"]["solubility_risk"] == "medium"
    assert payload["pose_contact_assessment"]["predicted_contacts"] == []
    assert payload["known_source_backed_facts"] == []


def test_missing_developability_fields_become_unknown() -> None:
    payload, _notes = infer_peb_engineer._normalise_report_payload(
        {"developability": {"stability_risk": "low"}}
    )
    developability = payload["developability"]
    assert developability["stability_risk"] == "low"
    assert developability["solubility_risk"] == "unknown"
    assert developability["toxicity_risk"] == "unknown"
    assert developability["hemolysis_risk"] == "unknown"
    assert developability["cytotoxicity_risk"] == "unknown"
    assert developability["synthesis_complexity"] == "unknown"


def test_normalizer_does_not_invent_contacts() -> None:
    payload, _notes = infer_peb_engineer._normalise_report_payload(
        {"pose_contact_assessment": {"status": "not_assessed"}}
    )
    assert payload["pose_contact_assessment"]["predicted_contacts"] == []
    assert payload["pose_contact_assessment"]["interface_residues"] == []


def test_normalizer_does_not_invent_scientific_claims() -> None:
    payload, _notes = infer_peb_engineer._normalise_report_payload({})
    assert payload["known_source_backed_facts"] == []
    assert payload["unsupported_claims"] == []
    assert payload["target_binding"]["relative_rank"] == []
    assert payload["functional_assay_estimate"]["categories"] == []
    assert payload["human_effect_estimate"]["claim_status"] == "insufficient_information"
    assert payload["pose_contact_assessment"]["predicted_contacts"] == []


def test_known_report_with_missing_optional_containers_normalizes_safely() -> None:
    payload, notes = infer_peb_engineer._normalise_report_payload(
        {
            "known_source_backed_facts": [{"source_id": "SRC-1"}],
            "overall_confidence": "medium",
        }
    )
    assert payload["known_source_backed_facts"] == [{"source_id": "SRC-1"}]
    assert payload["overall_confidence"] == "medium"
    assert payload["structure_source_reference"]["status"] == "not_applicable"
    assert payload["pose_contact_assessment"]["status"] == "not_assessed"
    assert payload["target_binding"]["relative_rank"] == []
    assert "pose_contact_assessment_defaulted" in notes


def test_existing_sanity_predictions_all_parse_and_non_pose_stay_valid() -> None:
    full_path = Path("training/runs/peb-engineer-v0/eval/sanity_predictions_reparse_clean_80of80.jsonl")
    path = full_path if full_path.exists() else Path("training/runs/peb-engineer-v0/eval/sanity_predictions.jsonl")
    if not path.exists():
        pytest.skip("sanity predictions are not present in this checkout")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if path == full_path:
        assert len(rows) == 80
    reparsed = [
        sanity_eval_peb_engineer._prediction_record(
            index=index,
            source_row=row,
            text=str(row.get("raw_prediction_text") or row.get("prediction_text") or ""),
            prompt_hash=str(row.get("prompt_hash") or f"row-{index}"),
        )
        for index, row in enumerate(rows)
    ]
    assert all(row["parse_valid"] for row in reparsed)
    assert all(row["schema_valid"] for row in reparsed if row["task_type"] != "pose_contact_prediction")


def test_base_resume_detects_and_runs_only_missing_human_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, output, leaderboard, run_dir, missing_ids = _write_resume_fixture(tmp_path)
    assert len(missing_ids) == 10
    called_ids: list[str] = []

    monkeypatch.setattr(
        infer_peb_engineer,
        "_create_sampler",
        lambda *args, **kwargs: (object(), object(), "tinker://test/sampler", object()),
    )

    def fake_predict_chunk(**kwargs):
        ids = [case["benchmark_id"] for case in kwargs["cases"]]
        called_ids.extend(ids)
        return [_human_prediction(benchmark_id) for benchmark_id in ids], {
            "chunk_status": "completed",
            "transient_api_errors": 0,
            "transient_invalid_json": 0,
            "last_error_type": "unknown",
            "last_error_message": "",
            "recovered_after_retry": True,
        }

    monkeypatch.setattr(infer_peb_engineer, "_predict_chunk", fake_predict_chunk)
    manifest = infer_peb_engineer.infer(
        run_dir,
        release,
        output,
        "the-spice",
        "base",
        leaderboard_dir=leaderboard,
        chunk_size=4,
        resume=True,
    )

    assert called_ids == missing_ids
    human_rows = [json.loads(line) for line in (output / "base_human_effect.jsonl").read_text().splitlines()]
    binding_rows = [json.loads(line) for line in (output / "base_binding_rank.jsonl").read_text().splitlines()]
    assert len(human_rows) == 200
    assert len({row["benchmark_id"] for row in human_rows}) == 200
    assert len(binding_rows) == 25
    assert (leaderboard / "predictions" / "the_spice" / "base_human_effect.jsonl").exists()
    assert manifest["status"] == "completed_clean"
    assert manifest["predictions_written"] == 225
    assert manifest["track_summaries"]["human_effect"]["missing_ids"] == []
    assert manifest["track_summaries"]["binding_rank"]["status"] == "completed_from_existing"
    assert manifest["tracks"] == ["human_effect", "binding_rank"]


def test_base_resume_dedupes_existing_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release, output, leaderboard, run_dir, _missing_ids = _write_resume_fixture(
        tmp_path, duplicate_existing=True
    )
    monkeypatch.setattr(
        infer_peb_engineer,
        "_create_sampler",
        lambda *args, **kwargs: (object(), object(), "tinker://test/sampler", object()),
    )
    monkeypatch.setattr(
        infer_peb_engineer,
        "_predict_chunk",
        lambda **kwargs: (
            [_human_prediction(case["benchmark_id"]) for case in kwargs["cases"]],
            {
                "chunk_status": "completed",
                "transient_api_errors": 0,
                "transient_invalid_json": 0,
                "last_error_type": "unknown",
                "last_error_message": "",
                "recovered_after_retry": False,
            },
        ),
    )

    infer_peb_engineer.infer(run_dir, release, output, "the-spice", "base", leaderboard_dir=leaderboard)
    rows = [json.loads(line) for line in (output / "base_human_effect.jsonl").read_text().splitlines()]
    assert len(rows) == 200
    assert len({row["benchmark_id"] for row in rows}) == 200


def test_human_effect_parse_failure_uses_final_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        "benchmark_id": "PEB-HFX-00178",
        "track": "human_effect",
        "source_database": "test-source",
        "claim_text": "claim not provided",
        "peptide": {"sequence": "ACD"},
    }
    fallback_ids: list[str] = []

    monkeypatch.setattr(infer_peb_engineer, "_sample_text", lambda **kwargs: "not json")

    def fake_final_fallback(**kwargs):
        fallback_ids.append(kwargs["case"]["benchmark_id"])
        return _human_prediction(kwargs["case"]["benchmark_id"])

    monkeypatch.setattr(infer_peb_engineer, "_predict_human_effect_final_fallback", fake_final_fallback)
    predictions, summary = infer_peb_engineer._predict_chunk(
        module=object(),
        sampler=object(),
        tokenizer=object(),
        track="human_effect",
        mode="base",
        cases=[case],
        model_name="the-spice",
        timeout=1,
        max_tokens=128,
        retries=2,
    )

    assert fallback_ids == ["PEB-HFX-00178"]
    assert predictions == [_human_prediction("PEB-HFX-00178")]
    assert summary["chunk_status"] == "completed"
    assert summary["transient_invalid_json"] == 1
    assert summary["recovered_after_retry"] is True


def test_human_effect_failed_final_fallback_writes_failed_row(monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        "benchmark_id": "PEB-HFX-00179",
        "track": "human_effect",
        "source_database": "test-source",
        "claim_text": "claim not provided",
        "peptide": {"sequence": "ACD"},
    }

    monkeypatch.setattr(infer_peb_engineer, "_sample_text", lambda **kwargs: "not json")

    def fake_final_fallback(**kwargs):
        del kwargs
        raise ValueError("degenerate_non_json_output")

    monkeypatch.setattr(infer_peb_engineer, "_predict_human_effect_final_fallback", fake_final_fallback)
    predictions, summary = infer_peb_engineer._predict_chunk(
        module=object(),
        sampler=object(),
        tokenizer=object(),
        track="human_effect",
        mode="base",
        cases=[case],
        model_name="the-spice",
        timeout=1,
        max_tokens=128,
        retries=2,
    )

    assert summary["chunk_status"] == "completed"
    assert summary["last_error_type"] == "invalid_json_recovered_with_failed_rows"
    assert predictions is not None
    assert len(predictions) == 1
    assert predictions[0]["benchmark_id"] == "PEB-HFX-00179"
    assert predictions[0]["status"] == "failed"
    assert predictions[0]["json_valid"] is False
    assert predictions[0]["schema_valid"] is False
    assert predictions[0]["error_type"] == "unresolved_invalid_json"


def test_complete_base_resume_does_not_require_pose_structure_or_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, output, leaderboard, run_dir, missing_ids = _write_resume_fixture(
        tmp_path, human_existing_count=200
    )
    assert missing_ids == []

    def fail_sampler(*args, **kwargs):
        raise AssertionError("sampler should not be created for complete resume")

    monkeypatch.setattr(infer_peb_engineer, "_create_sampler", fail_sampler)
    manifest = infer_peb_engineer.infer(
        run_dir,
        release,
        output,
        "the-spice",
        "base",
        leaderboard_dir=leaderboard,
        resume=True,
    )

    assert manifest["status"] == "completed_clean"
    assert manifest["reason"] == "completed_clean"
    assert manifest["predictions_written"] == 225
    assert manifest["unresolved_api_error_count"] == 0
    assert manifest["unresolved_invalid_json_count"] == 0
    assert manifest["tracks"] == ["human_effect", "binding_rank"]


def test_unresolved_invalid_rows_are_accounted_as_failed_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, output, leaderboard, run_dir, missing_ids = _write_resume_fixture(
        tmp_path, human_existing_count=199
    )
    assert len(missing_ids) == 1
    monkeypatch.setattr(
        infer_peb_engineer,
        "_create_sampler",
        lambda *args, **kwargs: (object(), object(), "tinker://test/sampler", object()),
    )

    def fake_failed_chunk(**kwargs):
        del kwargs
        return None, {
            "chunk_status": "failed",
            "transient_api_errors": 0,
            "transient_invalid_json": 1,
            "last_error_type": "invalid_json_or_schema",
            "last_error_message": "degenerate_non_json_output",
            "recovered_after_retry": False,
        }

    monkeypatch.setattr(infer_peb_engineer, "_predict_chunk", fake_failed_chunk)
    manifest = infer_peb_engineer.infer(
        run_dir,
        release,
        output,
        "the-spice",
        "base",
        leaderboard_dir=leaderboard,
        resume=True,
    )

    assert manifest["status"] == "completed_with_failures"
    assert manifest["reason"] == "explicit_failed_prediction_rows"
    assert manifest["track_summaries"]["human_effect"]["completed"] == 200
    assert manifest["track_summaries"]["human_effect"]["missing_ids"] == []
    assert manifest["unresolved_invalid_json_count"] == 1
    rows = [json.loads(line) for line in (output / "base_human_effect.jsonl").read_text().splitlines()]
    assert len(rows) == 200
    assert len({row["benchmark_id"] for row in rows}) == 200
    failed = [row for row in rows if row.get("status") == "failed"]
    assert len(failed) == 1
    assert failed[0]["benchmark_id"] == missing_ids[0]
    assert failed[0]["json_valid"] is False
    assert failed[0]["schema_valid"] is False
    assert failed[0]["error_type"] == "unresolved_invalid_json"


def test_failed_human_effect_prediction_is_schema_valid_and_marked_failed() -> None:
    prediction = infer_peb_engineer._failed_human_effect_prediction(
        {"benchmark_id": "PEB-HFX-00178"},
        model_name="the-spice",
        error_message="degenerate_non_json_output",
    )
    parsed = HumanEffectPrediction.model_validate(prediction)
    assert parsed.status == "failed"
    assert parsed.json_valid is False
    assert parsed.schema_valid is False
    assert parsed.error_type == "unresolved_invalid_json"


def test_failed_human_effect_predictions_score_as_wrong() -> None:
    policy = {
        "raw_data_redistribution": "allowed",
        "processed_label_redistribution": "allowed",
        "commercial_use": "unknown",
        "attribution_required": False,
        "share_alike_obligation": False,
        "use_in_public_leaderboard": "allowed",
    }
    case = HumanEffectCase.model_validate(
        {
            "benchmark_id": "PEB-HFX-FAIL",
            "track": "human_effect",
            "release_mode": "derived",
            "source_database": "test-source",
            "source_id": "SRC-1",
            "source_version": "test",
            "retrieval_date": "2026-01-01",
            "license_or_usage_note": "test fixture",
            "redistribution_policy_snapshot": policy,
            "qc_status": "source_checked",
            "split": "test",
            "leakage_group": {},
            "peptide": {"sequence": "ACD"},
            "claim_text": "unsupported claim",
            "category": "toxic_adverse_effect_concern",
            "evidence_level": "human_clinical_evidence",
            "evidence_direction": "mixed",
            "claim_status": "unsafe_to_claim",
            "safety_status": "known_risk",
            "source_evidence_type": "test",
        }
    )
    failed_prediction = HumanEffectPrediction.model_validate(
        infer_peb_engineer._failed_human_effect_prediction(
            {"benchmark_id": "PEB-HFX-FAIL"},
            model_name="the-spice",
        )
    )

    result = evaluate_human_effect([case], [failed_prediction])

    assert result.n_predictions == 1
    assert result.metrics["failed_prediction_count"] == 1
    assert result.metrics["valid_prediction_rate"] == 0.0
    assert result.metrics["category_macro_f1"] == 0.0
    assert result.metrics["claim_status_accuracy"] == 0.0
    assert any("failed prediction" in warning for warning in result.warnings)


def test_completed_with_failures_can_be_scored() -> None:
    assert "completed_with_failures" in SCOREABLE_MANIFEST_STATUSES
    assert "completed_clean" in SCOREABLE_MANIFEST_STATUSES


def test_cost_estimate_expands_compact_corpus_under_cap() -> None:
    config = {
        "model_name": "GPT-OSS-20B",
        "target_train_tokens_min": 60_000_000,
        "target_train_tokens_max": 75_000_000,
        "hard_cost_cap_usd": 35,
        "pricing": {
            "estimated_sft_usd_per_million_tokens": 0.35,
            "estimated_sampling_usd_per_million_tokens": 0.20,
            "estimated_storage_usd": 1.5,
        },
    }
    train_rows = [{"token_estimate": 3_000_000}]
    dev_rows = [{"token_estimate": 200_000}]
    estimate = _cost_estimate(config, train_rows, dev_rows)
    assert estimate["train_token_estimate"] >= 60_000_000
    assert estimate["under_hard_cap"] is True


def test_leakage_check_rejects_test_case_in_training_rows(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_jsonl(release / "structure" / "splits" / "test.jsonl", [{"benchmark_id": "PEB-TEST"}])
    for track in ("pose", "binding_rank", "human_effect"):
        _write_jsonl(release / track / "splits" / "test.jsonl", [])
    for track in ("structure", "pose", "binding_rank", "human_effect"):
        for split in ("train", "dev"):
            _write_jsonl(release / track / "splits" / f"{split}.jsonl", [])

    data_dir = tmp_path / "sft"
    _write_jsonl(data_dir / "train.jsonl", [_example("PEB-TEST", "train")])
    _write_jsonl(data_dir / "dev.jsonl", [])
    _write_jsonl(data_dir / "holdout.jsonl", [])
    report = check_leakage(data_dir, release, tmp_path / "artifacts")
    assert report["passed"] is False
    assert any(issue["type"] == "test_case_label_in_training_data" for issue in report["issues"])


def test_leakage_check_rejects_answer_like_prompt_fields(tmp_path: Path) -> None:
    release = tmp_path / "release"
    for track in ("structure", "pose", "binding_rank", "human_effect"):
        for split in ("train", "dev", "test"):
            _write_jsonl(release / track / "splits" / f"{split}.jsonl", [])

    data_dir = tmp_path / "sft"
    _write_jsonl(data_dir / "train.jsonl", [_example("PEB-TRAIN", "train", "normalized_rank: 1.0")])
    _write_jsonl(data_dir / "dev.jsonl", [])
    _write_jsonl(data_dir / "holdout.jsonl", [])
    report = check_leakage(data_dir, release, tmp_path / "artifacts")
    assert report["passed"] is False
    assert any(issue["type"] == "answer_like_field_in_prompt" for issue in report["issues"])


def test_mock_tinker_training_calls_low_level_primitives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeServiceClient.calls = []
    _FakeTrainingClient.calls = []
    monkeypatch.setitem(sys.modules, "tinker", _fake_tinker_module())
    monkeypatch.setenv(TINKER_ENV_NAME, "unit-test-key")
    config, data, artifacts, run = _write_training_fixture(tmp_path)

    result = launch_training(
        config,
        run,
        data,
        artifacts,
        run_to_completion=True,
        max_cost_usd=35,
        export_adapter=True,
        batch_size=1,
    )

    assert result["status"] == "completed"
    assert ("create_lora_training_client", "openai/gpt-oss-20b", 32, 42) in _FakeServiceClient.calls
    assert any(call[0] == "forward_backward" for call in _FakeTrainingClient.calls if isinstance(call, tuple))
    assert any(call[0] == "optim_step" for call in _FakeTrainingClient.calls if isinstance(call, tuple))
    assert any(call[0] == "save_state" for call in _FakeTrainingClient.calls if isinstance(call, tuple))
    assert json.loads((run / "run_manifest.json").read_text())["status"] == "training_completed"
    assert (run / "hf_upload" / "model_card.json").exists()


def test_mock_tinker_skips_malformed_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTrainingClient.calls = []
    monkeypatch.setitem(sys.modules, "tinker", _fake_tinker_module())
    monkeypatch.setenv(TINKER_ENV_NAME, "unit-test-key")
    malformed = {"example_id": "bad", "benchmark_id": "PEB-BAD", "messages": []}
    config, data, artifacts, run = _write_training_fixture(tmp_path, [_example("PEB-OK", "train"), malformed])

    result = launch_training(config, run, data, artifacts, run_to_completion=True, batch_size=4)

    assert result["valid_train_examples"] == 1
    assert result["skipped_examples"] == 1


def test_training_missing_key_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", _fake_tinker_module())
    monkeypatch.delenv(TINKER_ENV_NAME, raising=False)
    config, data, artifacts, run = _write_training_fixture(tmp_path)

    with pytest.raises(SystemExit):
        launch_training(config, run, data, artifacts, run_to_completion=True)

    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["hard_blocker"] == "missing_tinker_api_key"


def test_training_respects_cost_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", _fake_tinker_module())
    monkeypatch.setenv(TINKER_ENV_NAME, "unit-test-key")
    config, data, artifacts, run = _write_training_fixture(tmp_path)

    with pytest.raises(SystemExit):
        launch_training(config, run, data, artifacts, run_to_completion=True, max_cost_usd=0.5)

    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["hard_blocker"] == "cost_cap_exceeded"


def test_mock_training_does_not_write_key_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", _fake_tinker_module())
    fake_key = "unit-test-secret-value"
    monkeypatch.setenv(TINKER_ENV_NAME, fake_key)
    config, data, artifacts, run = _write_training_fixture(tmp_path)

    launch_training(config, run, data, artifacts, run_to_completion=True, export_adapter=True)

    combined = "\n".join(path.read_text(encoding="utf-8") for path in run.rglob("*") if path.is_file())
    assert fake_key not in combined


def test_only_root_readme_markdown_exists() -> None:
    markdown = [
        path
        for path in Path(".").rglob("*.md")
        if ".git" not in path.parts and ".venv" not in path.parts and ".pytest_cache" not in path.parts
    ]
    assert markdown == [Path("README.md")]


def test_prepare_upload_writes_the_spice_metadata_without_local_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TINKER_ENV_NAME, raising=False)
    config, _data, artifacts, run = _write_training_fixture(tmp_path)
    (run / "run_manifest.json").parent.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "training_completed", "model_name": "openai/gpt-oss-20b"}),
        encoding="utf-8",
    )
    (run / "adapter_export_manifest.json").write_text(
        json.dumps(
            {
                "status": "remote_export_available",
                "remote_sampler_path": "tinker://example/sampler",
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_upload(
        run,
        config,
        artifacts,
        model_name="the-spice",
        export_adapter=True,
    )

    model_card = json.loads((run / "hf_upload" / "model_card.json").read_text())
    adapter = json.loads((run / "hf_upload" / "adapter_export_manifest.json").read_text())
    assert manifest["model_name"] == "the-spice"
    assert manifest["local_download_status"] == "blocked_missing_auth_env"
    assert model_card["model_name"] == "the-spice"
    assert model_card["base_model"] == "openai/gpt-oss-20b"
    assert adapter["adapter_type"] == "LoRA"


def test_prepare_upload_uses_cookbook_download_and_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class _FakeWeights:
        @staticmethod
        def download(*, tinker_path, output_dir):
            calls.append(("download", tinker_path, output_dir))
            raw = Path(output_dir)
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "adapter_config.json").write_text("{}", encoding="utf-8")
            (raw / "adapter_model.safetensors").write_text("raw", encoding="utf-8")
            return str(raw)

        @staticmethod
        def build_lora_adapter(*, base_model, adapter_path, output_path):
            calls.append(("build_lora_adapter", base_model, adapter_path, output_path))
            out = Path(output_path)
            out.mkdir(parents=True)
            (out / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "target_modules": ["q_proj"]}),
                encoding="utf-8",
            )
            (out / "adapter_model.safetensors").write_text("peft", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "tinker_cookbook", types.SimpleNamespace(weights=_FakeWeights))
    monkeypatch.setenv(TINKER_ENV_NAME, "unit-test-key")
    monkeypatch.setenv("PEB_TINKER_FULL_BASE_EXPORT", "1")
    config, _data, artifacts, run = _write_training_fixture(tmp_path)
    (run / "run_manifest.json").parent.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "training_completed", "model_name": "openai/gpt-oss-20b"}),
        encoding="utf-8",
    )
    (run / "adapter_export_manifest.json").write_text(
        json.dumps(
            {
                "status": "remote_export_available",
                "remote_sampler_path": "tinker://example/sampler_weights/final",
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_upload(
        run,
        config,
        artifacts,
        model_name="the-spice",
        export_adapter=True,
    )

    assert calls[0] == (
        "download",
        "tinker://example/sampler_weights/final",
        str(run / "adapter_raw"),
    )
    assert calls[1][0] == "build_lora_adapter"
    assert calls[1][1] == "openai/gpt-oss-20b"
    assert manifest["adapter_status"] == "local_peft_adapter_available"
    assert (run / "hf_upload" / "adapter_model.safetensors").exists()
    peft_config = json.loads((run / "hf_upload" / "adapter_config.json").read_text())
    assert peft_config["peft_type"] == "LORA"
    assert (run / "hf_upload" / "training_adapter_config.json").exists()
    adapter = json.loads((run / "hf_upload" / "adapter_export_manifest.json").read_text())
    assert adapter["adapter_model_exists"] is True


def test_sanity_eval_preserves_completed_remote_results(tmp_path: Path) -> None:
    _config, data, _artifacts, run = _write_training_fixture(tmp_path)
    (run / "run_manifest.json").parent.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "training_completed"}),
        encoding="utf-8",
    )
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "sanity_eval_results.json").write_text(
        json.dumps({"status": "completed", "prediction_count": 1}),
        encoding="utf-8",
    )
    _write_jsonl(run / "eval" / "sanity_predictions.jsonl", [{"prediction_text": "{}"}])

    result = sanity_eval(run, data, 1, "the-spice")

    assert result["status"] == "completed"
    assert result["model_name"] == "the-spice"


def test_peb_eval_records_blocker_without_fake_leaderboard_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TINKER_ENV_NAME, raising=False)
    _config, _data, _artifacts, run = _write_training_fixture(tmp_path)
    release = tmp_path / "release"
    leaderboard = tmp_path / "leaderboard"
    (run / "run_manifest.json").parent.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "training_completed", "model_name": "openai/gpt-oss-20b"}),
        encoding="utf-8",
    )
    (run / "adapter_export_manifest.json").write_text(
        json.dumps({"status": "remote_export_available", "remote_sampler_path": "tinker://example"}),
        encoding="utf-8",
    )

    result = evaluate(run, release, leaderboard, "the-spice")

    assert result["status"] == "eval_blocked"
    assert result["reason"] == "missing_inference_manifest"
    assert result["leaderboard_row_added"] is False
    assert (run / "eval" / "peb_scores.json").exists()
    assert (leaderboard / "results" / "the_spice" / "eval_blocked.json").exists()


def test_peb_eval_scores_base_artifacts_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, output, leaderboard, run_dir, _missing_ids = _write_resume_fixture(
        tmp_path, human_existing_count=200
    )
    manifest = {
        "status": "completed_with_failures",
        "reason": "explicit_failed_prediction_rows",
        "mode": "base",
        "tracks": ["human_effect", "binding_rank"],
        "predictions_written": 225,
        "track_summaries": {
            "human_effect": {
                "attempted": 200,
                "completed": 200,
                "unresolved_api_errors": 0,
                "unresolved_invalid_json": 1,
                "retry_recovered_count": 0,
            },
            "binding_rank": {
                "attempted": 25,
                "completed": 25,
                "unresolved_api_errors": 0,
                "unresolved_invalid_json": 0,
                "retry_recovered_count": 0,
            },
        },
    }
    (output / "base_inference_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    def fail_infer(*args, **kwargs):
        raise AssertionError("evaluation must not call inference")

    monkeypatch.setattr(eval_peb_engineer.infer_peb_engineer, "infer", fail_infer)
    monkeypatch.setattr(
        eval_peb_engineer,
        "_cases_for_track",
        lambda _release, track: [{"benchmark_id": f"{track}-case"}],
    )
    monkeypatch.setattr(
        eval_peb_engineer,
        "_evaluate_track",
        lambda track, cases, predictions: {
            "track": track,
            "n_cases": len(cases),
            "n_predictions": len(predictions),
            "metrics": {
                "category_macro_f1": 0.5,
                "evidence_level_ordinal_accuracy": 0.5,
                "overclaim_penalty": 0,
                "valid_prediction_rate": 0.995,
                "failed_prediction_count": 1,
            }
            if track == "human_effect"
            else {
                "spearman": 1.0,
                "kendall_tau": 1.0,
                "pairwise_ranking_accuracy": 1.0,
            },
            "warnings": [],
        },
    )
    monkeypatch.setattr(eval_peb_engineer, "_copy_leaderboard_to_release", lambda *_args, **_kwargs: None)

    result = evaluate(run_dir, release, leaderboard, "the-spice", modes=("base",))

    assert result["status"] == "completed"
    assert result["leaderboard_row_added"] is True
    assert result["complete_modes"] == ["base"]
    assert result["manifests"]["base"]["status"] == "completed_with_failures"
    assert (run_dir / "eval" / "peb_scores.json").exists()
    rows = json.loads((leaderboard / "leaderboard.json").read_text(encoding="utf-8"))
    assert any(row["model_id"] == "the-spice" and row["mode"] == "base" for row in rows)
