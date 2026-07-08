from peb.io import read_jsonl
from peb.processing.audit import audit_records


def test_audit_rejects_unbacked_approved_indication():
    record = read_jsonl("data/fixtures/human_effect_fixture.jsonl")[0]
    record["evidence_level"] = "approved_human_indication"
    record["source_database"] = "synthetic_fixture"
    result = audit_records([record])
    assert not result.passed
    assert any("approved indication" in error for error in result.errors)


def test_audit_accepts_fixture_pose():
    record = read_jsonl("data/fixtures/pose_fixture.jsonl")[0]
    result = audit_records([record])
    assert result.passed

