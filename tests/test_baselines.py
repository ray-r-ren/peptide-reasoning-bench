from peb.baselines import make_baseline_predictions
from peb.io import read_jsonl
from peb.metrics.evidence_metrics import evaluate_human_effect
from peb.schemas import (
    ClaimStatus,
    EvidenceDirection,
    EvidenceLevel,
    HumanEffectCategory,
    SafetyStatus,
    Track,
    validate_case_record,
    validate_prediction_record,
)


def test_baseline_is_deterministic_and_valid():
    cases = [validate_case_record(record) for record in read_jsonl("data/fixtures/human_effect_fixture.jsonl")]
    first = make_baseline_predictions(Track.human_effect, cases, seed=7)
    second = make_baseline_predictions(Track.human_effect, cases, seed=7)
    assert first == second
    predictions = [validate_prediction_record(record) for record in first]
    result = evaluate_human_effect(cases, predictions)
    assert result.n_predictions == len(cases)


def test_non_oracle_human_effect_baseline_does_not_read_gold_labels():
    case = validate_case_record(read_jsonl("data/fixtures/human_effect_fixture.jsonl")[0])
    mutated = case.model_copy(
        update={
            "category": HumanEffectCategory.endocrine_hormonal,
            "evidence_level": EvidenceLevel.approved_human_indication,
            "evidence_direction": EvidenceDirection.positive,
            "claim_status": ClaimStatus.supported,
            "safety_status": SafetyStatus.known_acceptable_under_approved_use,
        }
    )

    original_pred = make_baseline_predictions(
        Track.human_effect,
        [case],
        model_name="human_effect_non_oracle_baseline",
    )[0]
    mutated_pred = make_baseline_predictions(
        Track.human_effect,
        [mutated],
        model_name="human_effect_non_oracle_baseline",
    )[0]

    for field_name in ("category", "evidence_level", "evidence_direction", "claim_status", "safety_status"):
        assert original_pred[field_name] == mutated_pred[field_name]


def test_oracle_human_effect_baseline_is_label_copying_sanity_check():
    case = validate_case_record(read_jsonl("data/fixtures/human_effect_fixture.jsonl")[0])
    pred = make_baseline_predictions(
        Track.human_effect,
        [case],
        model_name="human_effect_oracle_source_reference_baseline",
    )[0]

    assert pred["evidence_level"] == case.evidence_level.value
    assert pred["evidence_direction"] == case.evidence_direction.value
