"""Structure metrics."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from peb.schemas import Coordinate, EvaluationResult, StructureCase, StructurePrediction, Track


def _by_atom(coordinates: Iterable[Coordinate]) -> dict[str, Coordinate]:
    return {coordinate.atom_id: coordinate for coordinate in coordinates}


def rmsd(gold: list[Coordinate], pred: list[Coordinate]) -> Optional[float]:
    gold_by_atom = _by_atom(gold)
    pred_by_atom = _by_atom(pred)
    shared = sorted(set(gold_by_atom) & set(pred_by_atom))
    if not shared:
        return None
    squared = 0.0
    for atom_id in shared:
        left = gold_by_atom[atom_id]
        right = pred_by_atom[atom_id]
        squared += (left.x - right.x) ** 2
        squared += (left.y - right.y) ** 2
        squared += (left.z - right.z) ** 2
    return math.sqrt(squared / len(shared))


def evaluate_structure(
    cases: list[StructureCase], predictions: list[StructurePrediction]
) -> EvaluationResult:
    pred_by_id = {prediction.benchmark_id: prediction for prediction in predictions}
    values: list[float] = []
    skipped = 0
    warnings: list[str] = []

    for case in cases:
        prediction = pred_by_id.get(case.benchmark_id)
        if prediction is None:
            skipped += 1
            warnings.append(f"{case.benchmark_id}: missing prediction")
            continue
        value = rmsd(case.gold_coordinates, prediction.coordinates)
        if value is None:
            skipped += 1
            warnings.append(f"{case.benchmark_id}: no shared coordinates for RMSD")
            continue
        values.append(value)

    mean_rmsd = sum(values) / len(values) if values else "not_computed"
    metrics = {
        "backbone_rmsd": mean_rmsd,
        "all_atom_rmsd": mean_rmsd,
        "confidence_calibration": "not_computed",
        "ensemble_support": "not_computed",
        "stereochemical_validity": "not_computed",
        "skipped_cases": skipped,
        "rmsd_evaluated_cases": len(values),
        "rmsd_skipped_count": skipped,
        "rmsd_not_computed_reason": "gold or predicted coordinates unavailable for source-reference cases"
        if skipped
        else "not_applicable",
    }
    return EvaluationResult(
        track=Track.structure,
        n_cases=len(cases),
        n_predictions=len(predictions),
        metrics=metrics,
        warnings=warnings,
    )
