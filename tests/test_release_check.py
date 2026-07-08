from peb.processing.release_check import release_check
from peb.release import build_release


def test_build_release_and_release_check(tmp_path):
    release_dir = tmp_path / "peb-v1.0-rc"
    build_release(release_dir)
    passed, errors, warnings = release_check(release_dir)
    assert not passed
    assert any("below required minimum" in error for error in errors)
    assert isinstance(warnings, list)
    assert (release_dir / "release_manifest.json").exists()
    assert (release_dir / "references" / "nonredistributable_source_index.jsonl").exists()
