"""Pose metrics."""

from __future__ import annotations

from peb.schemas import ContactPair, EvaluationResult, PoseCase, PosePrediction, Track


def _contact_key(contact: ContactPair) -> tuple[str, str]:
    return (contact.target_residue, contact.peptide_residue)


def _site_from_contacts(contacts: list[ContactPair]) -> set[str]:
    return {contact.target_residue for contact in contacts}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_pose(cases: list[PoseCase], predictions: list[PosePrediction]) -> EvaluationResult:
    pred_by_id = {prediction.benchmark_id: prediction for prediction in predictions}
    contact_precisions: list[float] = []
    contact_recalls: list[float] = []
    site_recoveries: list[float] = []
    orientation_matches = 0
    orientation_total = 0
    contact_metric_denominator = 0
    coordinate_reference_only = 0
    warnings: list[str] = []

    for case in cases:
        prediction = pred_by_id.get(case.benchmark_id)
        if prediction is None:
            warnings.append(f"{case.benchmark_id}: missing prediction")
            continue
        gold_contacts = {_contact_key(contact) for contact in case.native_contacts}
        if not gold_contacts:
            coordinate_reference_only += 1
            warnings.append(
                f"{case.benchmark_id}: no contact labels; excluded from contact metrics"
            )
            continue
        contact_metric_denominator += 1
        pred_contacts = {_contact_key(contact) for contact in prediction.predicted_contacts}
        overlap = len(gold_contacts & pred_contacts)
        contact_precisions.append(_safe_ratio(overlap, len(pred_contacts)))
        contact_recalls.append(_safe_ratio(overlap, len(gold_contacts)))

        gold_site = set(case.binding_site_residues) or _site_from_contacts(case.native_contacts)
        pred_site = set(prediction.binding_site_residues) or _site_from_contacts(
            prediction.predicted_contacts
        )
        site_recoveries.append(_safe_ratio(len(gold_site & pred_site), len(gold_site)))

        if case.orientation_label and prediction.orientation_label:
            orientation_total += 1
            orientation_matches += int(case.orientation_label == prediction.orientation_label)

    metrics = {
        "binding_site_recovery": sum(site_recoveries) / len(site_recoveries)
        if site_recoveries
        else 0.0,
        "interface_contact_precision": sum(contact_precisions) / len(contact_precisions)
        if contact_precisions
        else 0.0,
        "interface_contact_recall": sum(contact_recalls) / len(contact_recalls)
        if contact_recalls
        else 0.0,
        "native_contact_recovery": sum(contact_recalls) / len(contact_recalls)
        if contact_recalls
        else 0.0,
        "peptide_orientation_accuracy": _safe_ratio(orientation_matches, orientation_total),
        "interface_rmsd": "not_computed",
        "clash_penalty": "not_computed",
        "contact_metric_denominator": contact_metric_denominator,
        "pose_contact_labeled_subset_count": contact_metric_denominator,
        "pose_coordinate_reference_subset_count": coordinate_reference_only,
        "contact_metric_exclusion_reason": "coordinate-reference-only cases lack native contact labels",
    }
    return EvaluationResult(
        track=Track.pose,
        n_cases=len(cases),
        n_predictions=len(predictions),
        metrics=metrics,
        warnings=warnings,
    )
