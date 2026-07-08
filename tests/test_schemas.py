import pytest
from pydantic import ValidationError

from peb.io import read_jsonl
from peb.schemas import (
    HumanEffectPrediction,
    Track,
    validate_case_record,
    validate_prediction_record,
)


def test_valid_fixtures_pass():
    for path in [
        "data/fixtures/structure_fixture.jsonl",
        "data/fixtures/pose_fixture.jsonl",
        "data/fixtures/binding_rank_fixture.jsonl",
        "data/fixtures/human_effect_fixture.jsonl",
    ]:
        for record in read_jsonl(path):
            assert validate_case_record(record).track in Track


def test_invalid_enum_fails():
    record = read_jsonl("data/fixtures/human_effect_fixture.jsonl")[0]
    record["claim_status"] = "too_strong"
    with pytest.raises(ValidationError):
        validate_case_record(record)


def test_missing_required_field_fails():
    record = read_jsonl("data/fixtures/pose_fixture.jsonl")[0]
    del record["source_id"]
    with pytest.raises(ValidationError):
        validate_case_record(record)


def test_prediction_schema_validates():
    prediction = HumanEffectPrediction(
        prediction_id="pred-1",
        benchmark_id="case-1",
        category="no_known_human_effect_evidence",
        evidence_level="unsupported_contradicted_or_unsafe_claim",
        evidence_direction="not_applicable",
        claim_status="insufficient_information",
        safety_status="insufficient_safety_data",
    )
    assert validate_prediction_record(prediction.model_dump(mode="json")).benchmark_id == "case-1"

