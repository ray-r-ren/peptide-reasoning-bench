from peb.io import read_jsonl
from peb.metrics.evidence_metrics import evaluate_human_effect
from peb.schemas import (
    ClaimStatus,
    EvidenceLevel,
    HumanEffectPrediction,
    SafetyStatus,
    validate_case_record,
)


def _cases():
    return [validate_case_record(record) for record in read_jsonl("data/fixtures/human_effect_fixture.jsonl")]


def test_evidence_metrics_credit_abstention():
    cases = _cases()
    predictions = [
        HumanEffectPrediction(
            prediction_id=f"pred-{case.benchmark_id}",
            benchmark_id=case.benchmark_id,
            category=case.category,
            evidence_level=case.evidence_level,
            evidence_direction=case.evidence_direction,
            claim_status=case.claim_status,
            safety_status=case.safety_status,
            abstained=case.claim_status == ClaimStatus.insufficient_information,
        )
        for case in cases
    ]
    result = evaluate_human_effect(cases, predictions)
    assert result.metrics["category_macro_f1"] == 1
    assert result.metrics["correct_abstention_credit"] == 1


def test_trial_without_results_penalty():
    case = _cases()[1]
    pred = HumanEffectPrediction(
        prediction_id="overclaim",
        benchmark_id=case.benchmark_id,
        category=case.category,
        evidence_level=EvidenceLevel.approved_human_indication,
        evidence_direction=case.evidence_direction,
        claim_status=ClaimStatus.supported,
        safety_status=case.safety_status,
    )
    result = evaluate_human_effect([case], [pred])
    assert result.metrics["trial_without_results_penalty"] == 1


def test_pathway_only_overclaim_penalty():
    case = _cases()[2]
    pred = HumanEffectPrediction(
        prediction_id="pathway-overclaim",
        benchmark_id=case.benchmark_id,
        category=case.category,
        evidence_level=EvidenceLevel.human_clinical_evidence,
        evidence_direction=case.evidence_direction,
        claim_status=ClaimStatus.supported,
        safety_status=case.safety_status,
    )
    result = evaluate_human_effect([case], [pred])
    assert result.metrics["pathway_only_overclaim_penalty"] == 1


def test_unsafe_claim_penalty():
    case = _cases()[0].model_copy(update={"safety_status": SafetyStatus.not_for_human_use})
    pred = HumanEffectPrediction(
        prediction_id="unsafe-overclaim",
        benchmark_id=case.benchmark_id,
        category=case.category,
        evidence_level=case.evidence_level,
        evidence_direction=case.evidence_direction,
        claim_status=case.claim_status,
        safety_status=SafetyStatus.known_acceptable_under_approved_use,
    )
    result = evaluate_human_effect([case], [pred])
    assert result.metrics["safety_overclaim_penalty"] == 1

