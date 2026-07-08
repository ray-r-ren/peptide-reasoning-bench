import pytest
from pydantic import ValidationError

from peb.io import read_jsonl
from peb.metrics.pose_metrics import evaluate_pose
from peb.schemas import ContactPair, PosePrediction, validate_case_record


def test_pose_contact_recovery():
    case = validate_case_record(read_jsonl("data/fixtures/pose_fixture.jsonl")[0])
    pred = PosePrediction(
        prediction_id="pose-pred",
        benchmark_id=case.benchmark_id,
        predicted_contacts=case.native_contacts,
        binding_site_residues=case.binding_site_residues,
        orientation_label=case.orientation_label,
    )
    result = evaluate_pose([case], [pred])
    assert result.metrics["interface_contact_recall"] == 1
    assert result.metrics["binding_site_recovery"] == 1


def test_pose_contact_metrics_exclude_coordinate_reference_only_cases():
    contact_case = validate_case_record(read_jsonl("data/fixtures/pose_fixture.jsonl")[0])
    reference_case = contact_case.model_copy(
        update={
            "benchmark_id": "PEB-FIX-POSE-COORD-ONLY",
            "native_contacts": [],
            "binding_site_residues": [],
            "pose_subset": "pose_coordinate_reference",
            "contact_label_status": "coordinate_reference_only",
            "scoring_subset": "source_reference_only",
        }
    )
    predictions = [
        PosePrediction(
            prediction_id="pose-pred-contact",
            benchmark_id=contact_case.benchmark_id,
            predicted_contacts=contact_case.native_contacts,
            binding_site_residues=contact_case.binding_site_residues,
            orientation_label=contact_case.orientation_label,
        ),
        PosePrediction(
            prediction_id="pose-pred-reference",
            benchmark_id=reference_case.benchmark_id,
            predicted_contacts=[],
            binding_site_residues=[],
            orientation_label=reference_case.orientation_label,
        ),
    ]

    result = evaluate_pose([contact_case, reference_case], predictions)

    assert result.metrics["contact_metric_denominator"] == 1
    assert result.metrics["pose_contact_labeled_subset_count"] == 1
    assert result.metrics["pose_coordinate_reference_subset_count"] == 1


def test_malformed_contact_map_fails():
    with pytest.raises(ValidationError):
        ContactPair(target_residue="A10", peptide_residue="B:1")
