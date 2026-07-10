"""Human-effect evidence classification metrics."""

from __future__ import annotations

from collections import Counter

from peb.schemas import (
    ClaimStatus,
    EvaluationResult,
    EvidenceLevel,
    HumanEffectCase,
    HumanEffectPrediction,
    SafetyStatus,
    Track,
)

_EVIDENCE_ORDER = {
    EvidenceLevel.unsupported_contradicted_or_unsafe_claim: 0,
    EvidenceLevel.mechanistic_pathway_or_similarity_hypothesis: 1,
    EvidenceLevel.in_vitro_target_activity_evidence: 2,
    EvidenceLevel.animal_preclinical_phenotype_evidence: 3,
    EvidenceLevel.human_clinical_evidence: 4,
    EvidenceLevel.approved_human_indication: 5,
}


def _accuracy(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        false_positive = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        false_negative = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _is_failed_prediction(prediction: HumanEffectPrediction) -> bool:
    return (
        prediction.status == "failed"
        or prediction.json_valid is False
        or prediction.schema_valid is False
        or prediction.error_type == "unresolved_invalid_json"
    )


def _unsupported_detection(case: HumanEffectCase, prediction: HumanEffectPrediction) -> bool:
    if _is_failed_prediction(prediction):
        return False
    gold_unsupported = case.claim_status in {
        ClaimStatus.unsupported,
        ClaimStatus.unsafe_to_claim,
        ClaimStatus.insufficient_information,
    }
    pred_unsupported = prediction.claim_status in {
        ClaimStatus.unsupported,
        ClaimStatus.unsafe_to_claim,
        ClaimStatus.insufficient_information,
    }
    return (not gold_unsupported) or pred_unsupported


def evaluate_human_effect(
    cases: list[HumanEffectCase], predictions: list[HumanEffectPrediction]
) -> EvaluationResult:
    pred_by_id = {prediction.benchmark_id: prediction for prediction in predictions}
    matched_cases: list[HumanEffectCase] = []
    matched_predictions: list[HumanEffectPrediction] = []
    warnings: list[str] = []

    for case in cases:
        prediction = pred_by_id.get(case.benchmark_id)
        if prediction is None:
            warnings.append(f"{case.benchmark_id}: missing prediction")
            continue
        if _is_failed_prediction(prediction):
            warnings.append(f"{case.benchmark_id}: failed prediction")
        matched_cases.append(case)
        matched_predictions.append(prediction)

    categories_gold = [case.category.value for case in matched_cases]
    categories_pred = [
        "__failed_prediction__" if _is_failed_prediction(prediction) else prediction.category.value
        for prediction in matched_predictions
    ]
    level_distances = [
        max(_EVIDENCE_ORDER.values())
        if _is_failed_prediction(prediction)
        else abs(_EVIDENCE_ORDER[case.evidence_level] - _EVIDENCE_ORDER[prediction.evidence_level])
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    overclaims = [
        0
        if _is_failed_prediction(prediction)
        else int(_EVIDENCE_ORDER[prediction.evidence_level] > _EVIDENCE_ORDER[case.evidence_level])
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    hallucinated_support = [
        0
        if _is_failed_prediction(prediction)
        else
        int(
            case.claim_status
            in {ClaimStatus.unsupported, ClaimStatus.insufficient_information, ClaimStatus.unsafe_to_claim}
            and prediction.claim_status == ClaimStatus.supported
        )
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    unsafe_claims = [
        0
        if _is_failed_prediction(prediction)
        else
        int(
            case.safety_status
            in {SafetyStatus.known_risk, SafetyStatus.serious_known_risk, SafetyStatus.not_for_human_use}
            and prediction.safety_status == SafetyStatus.known_acceptable_under_approved_use
        )
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    trial_without_results = [
        0
        if _is_failed_prediction(prediction)
        else
        int(
            case.trial_has_results is False
            and prediction.evidence_level == EvidenceLevel.approved_human_indication
        )
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    pathway_only_overclaim = [
        0
        if _is_failed_prediction(prediction)
        else
        int(
            case.evidence_level == EvidenceLevel.mechanistic_pathway_or_similarity_hypothesis
            and _EVIDENCE_ORDER[prediction.evidence_level]
            > _EVIDENCE_ORDER[EvidenceLevel.mechanistic_pathway_or_similarity_hypothesis]
        )
        for case, prediction in zip(matched_cases, matched_predictions)
    ]
    correct_abstentions = [
        0
        if _is_failed_prediction(prediction)
        else
        int(
            prediction.abstained
            and case.claim_status
            in {ClaimStatus.insufficient_information, ClaimStatus.unsupported, ClaimStatus.unsafe_to_claim}
        )
        for case, prediction in zip(matched_cases, matched_predictions)
    ]

    status_counts = Counter(case.claim_status for case in matched_cases)
    unsupported_cases = sum(
        count
        for status, count in status_counts.items()
        if status
        in {ClaimStatus.unsupported, ClaimStatus.unsafe_to_claim, ClaimStatus.insufficient_information}
    )
    unsupported_hits = sum(
        int(_unsupported_detection(case, prediction))
        for case, prediction in zip(matched_cases, matched_predictions)
        if case.claim_status
        in {ClaimStatus.unsupported, ClaimStatus.unsafe_to_claim, ClaimStatus.insufficient_information}
    )

    metrics = {
        "category_macro_f1": _macro_f1(categories_gold, categories_pred),
        "evidence_level_ordinal_accuracy": _accuracy(
            [
                (not _is_failed_prediction(prediction))
                and case.evidence_level == prediction.evidence_level
                for case, prediction in zip(matched_cases, matched_predictions)
            ]
        ),
        "evidence_level_distance_penalty": sum(level_distances) / len(level_distances)
        if level_distances
        else 0.0,
        "evidence_direction_accuracy": _accuracy(
            [
                (not _is_failed_prediction(prediction))
                and case.evidence_direction == prediction.evidence_direction
                for case, prediction in zip(matched_cases, matched_predictions)
            ]
        ),
        "claim_status_accuracy": _accuracy(
            [
                (not _is_failed_prediction(prediction))
                and case.claim_status == prediction.claim_status
                for case, prediction in zip(matched_cases, matched_predictions)
            ]
        ),
        "unsupported_claim_detection_rate": unsupported_hits / unsupported_cases
        if unsupported_cases
        else 0.0,
        "unsafe_claim_detection_rate": 1.0 - (sum(unsafe_claims) / len(unsafe_claims))
        if unsafe_claims
        else 0.0,
        "hallucinated_support_penalty": sum(hallucinated_support),
        "overclaim_penalty": sum(overclaims),
        "correct_abstention_credit": sum(correct_abstentions),
        "trial_without_results_penalty": sum(trial_without_results),
        "pathway_only_overclaim_penalty": sum(pathway_only_overclaim),
        "safety_overclaim_penalty": sum(unsafe_claims),
        "abstention_calibration": "not_computed",
        "failed_prediction_count": sum(_is_failed_prediction(prediction) for prediction in matched_predictions),
        "valid_prediction_rate": (
            sum(not _is_failed_prediction(prediction) for prediction in matched_predictions)
            / len(matched_predictions)
            if matched_predictions
            else 0.0
        ),
    }
    return EvaluationResult(
        track=Track.human_effect,
        n_cases=len(cases),
        n_predictions=len(predictions),
        metrics=metrics,
        warnings=warnings,
    )
