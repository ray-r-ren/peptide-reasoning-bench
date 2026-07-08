from peb.processing.validate import validate_records


def test_fixture_files_validate():
    for path in [
        "data/fixtures/structure_fixture.jsonl",
        "data/fixtures/pose_fixture.jsonl",
        "data/fixtures/binding_rank_fixture.jsonl",
        "data/fixtures/human_effect_fixture.jsonl",
    ]:
        count, errors = validate_records(path)
        assert count >= 1
        assert errors == []

