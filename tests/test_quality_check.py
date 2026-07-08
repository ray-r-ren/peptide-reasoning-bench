from peb.quality_check import (
    QC_PASS_A,
    QC_PASS_B,
    QCOutcome,
    _merge_updates,
    _resolve_disagreements,
)


def test_disagreement_with_overclaim_gets_downgraded():
    record = {"benchmark_id": "case-1", "track": "human_effect"}
    pass_a = QCOutcome(
        benchmark_id="case-1",
        track="human_effect",
        checker=QC_PASS_A,
        result="pass",
        updates={"evidence_level": "approved_human_indication"},
    )
    pass_b = QCOutcome(
        benchmark_id="case-1",
        track="human_effect",
        checker=QC_PASS_B,
        result="downgrade",
        updates={"evidence_level": "mechanistic_pathway_or_similarity_hypothesis"},
    )
    disagreements = [
        {
            "field": "evidence_level",
            "pass_a": "approved_human_indication",
            "pass_b": "mechanistic_pathway_or_similarity_hypothesis",
        }
    ]

    updates = _merge_updates(pass_a, pass_b)
    resolution_updates, resolution = _resolve_disagreements(
        record,
        pass_a,
        pass_b,
        disagreements,
        "downgrade",
    )

    assert updates["evidence_level"] == "mechanistic_pathway_or_similarity_hypothesis"
    assert resolution == "downgraded"
    assert resolution_updates["qc_disagreement"] is True


def test_unresolved_disagreement_is_excluded_from_primary_scoring():
    record = {"benchmark_id": "case-2", "track": "pose"}
    pass_a = QCOutcome(
        benchmark_id="case-2",
        track="pose",
        checker=QC_PASS_A,
        result="warn",
        updates={"pose_subset": "pose_contact_labeled"},
    )
    pass_b = QCOutcome(
        benchmark_id="case-2",
        track="pose",
        checker=QC_PASS_B,
        result="warn",
        updates={"pose_subset": "pose_coordinate_reference"},
    )

    updates, resolution = _resolve_disagreements(
        record,
        pass_a,
        pass_b,
        [{"field": "pose_subset", "pass_a": "pose_contact_labeled", "pass_b": "pose_coordinate_reference"}],
        "warn",
    )

    assert resolution == "excluded_from_primary_scoring"
    assert updates["scoring_subset"] == "warning_only"
    assert updates["qc_result"] == "passed_with_warnings"
