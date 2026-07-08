"""Binding-rank metrics."""

from __future__ import annotations

import math
from itertools import combinations

from peb.schemas import (
    BindingRankCase,
    BindingRankPrediction,
    EvaluationResult,
    MeasurementDirection,
    Track,
)


def _rank(values: list[float]) -> list[float]:
    sorted_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(sorted_values):
        end = cursor
        while end + 1 < len(sorted_values) and sorted_values[end + 1][0] == sorted_values[cursor][0]:
            end += 1
        average = (cursor + end + 2) / 2.0
        for _, index in sorted_values[cursor : end + 1]:
            ranks[index] = average
        cursor = end + 1
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    mean_left = sum(left_rank) / len(left_rank)
    mean_right = sum(right_rank) / len(right_rank)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_rank, right_rank))
    var_left = sum((a - mean_left) ** 2 for a in left_rank)
    var_right = sum((b - mean_right) ** 2 for b in right_rank)
    if var_left == 0 or var_right == 0:
        return 0.0
    return covariance / math.sqrt(var_left * var_right)


def kendall_tau(left: list[float], right: list[float]) -> float:
    pairs = list(combinations(range(len(left)), 2))
    if not pairs:
        return 0.0
    concordant = 0
    discordant = 0
    for i, j in pairs:
        left_delta = left[i] - left[j]
        right_delta = right[i] - right[j]
        product = left_delta * right_delta
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def pairwise_accuracy(gold: list[float], pred: list[float]) -> float:
    pairs = list(combinations(range(len(gold)), 2))
    scored = 0
    correct = 0
    for i, j in pairs:
        gold_delta = gold[i] - gold[j]
        pred_delta = pred[i] - pred[j]
        if gold_delta == 0 or pred_delta == 0:
            continue
        scored += 1
        correct += int(gold_delta * pred_delta > 0)
    return correct / scored if scored else 0.0


def _gold_strengths(case: BindingRankCase) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for item in case.items:
        value = item.measured_value
        strengths[item.item_id] = (
            -value if case.measurement_direction == MeasurementDirection.lower_is_stronger else value
        )
    return strengths


def _prediction_scores(prediction: BindingRankPrediction) -> dict[str, float]:
    return {score.item_id: score.score for score in prediction.scores}


def _top_k_enrichment(gold: list[float], pred: list[float], k: int = 1) -> float:
    if not gold:
        return 0.0
    k = min(k, len(gold))
    gold_top = set(sorted(range(len(gold)), key=lambda index: gold[index], reverse=True)[:k])
    pred_top = set(sorted(range(len(pred)), key=lambda index: pred[index], reverse=True)[:k])
    return len(gold_top & pred_top) / k


def evaluate_binding_rank(
    cases: list[BindingRankCase], predictions: list[BindingRankPrediction]
) -> EvaluationResult:
    pred_by_id = {prediction.benchmark_id: prediction for prediction in predictions}
    warnings: list[str] = []
    per_panel: list[dict[str, float]] = []

    for case in cases:
        if not case.comparable_panel:
            warnings.append(f"{case.benchmark_id}: incomparable panel flagged")
            continue
        prediction = pred_by_id.get(case.benchmark_id)
        if prediction is None:
            warnings.append(f"{case.benchmark_id}: missing prediction")
            continue
        gold_by_item = _gold_strengths(case)
        pred_by_item = _prediction_scores(prediction)
        shared = sorted(set(gold_by_item) & set(pred_by_item))
        if len(shared) < 2:
            warnings.append(f"{case.benchmark_id}: fewer than two shared items")
            continue
        gold = [gold_by_item[item] for item in shared]
        pred = [pred_by_item[item] for item in shared]
        per_panel.append(
            {
                "spearman": spearman(gold, pred),
                "kendall_tau": kendall_tau(gold, pred),
                "pairwise_accuracy": pairwise_accuracy(gold, pred),
                "top_1_enrichment": _top_k_enrichment(gold, pred, 1),
                "concordance_index": pairwise_accuracy(gold, pred),
            }
        )

    def average(key: str) -> float:
        return sum(panel[key] for panel in per_panel) / len(per_panel) if per_panel else 0.0

    metrics = {
        "spearman": average("spearman"),
        "kendall_tau": average("kendall_tau"),
        "pairwise_ranking_accuracy": average("pairwise_accuracy"),
        "top_1_enrichment": average("top_1_enrichment"),
        "concordance_index": average("concordance_index"),
        "assay_aware_subgroup_count": len(per_panel),
    }
    return EvaluationResult(
        track=Track.binding_rank,
        n_cases=len(cases),
        n_predictions=len(predictions),
        metrics=metrics,
        warnings=warnings,
    )

