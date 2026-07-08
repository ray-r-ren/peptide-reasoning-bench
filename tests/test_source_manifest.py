from peb.registry import validate_source_manifest


def test_source_manifest_required_fields():
    entries = validate_source_manifest()
    assert len(entries) >= 7
    assert {entry.source_bucket.value for entry in entries} <= {"A", "B", "C"}
    for entry in entries:
        assert entry.name
        assert entry.expected_fields
        assert entry.license_or_usage_note
        assert entry.redistribution_policy.use_in_public_leaderboard.value in {
            "allowed",
            "caution",
            "avoid",
        }

