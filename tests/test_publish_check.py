import json
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

from peb.io import read_jsonl, write_jsonl
from peb.quality_check import package_release, publish_check

RELEASE_DIR = Path("data/releases/peb-v1.0-rc")


def _copy_release(tmp_path):
    output = tmp_path / "release"
    shutil.copytree(RELEASE_DIR, output)
    return output


def test_publish_check_qced_release_passes():
    passed, errors, warnings, summary = publish_check(RELEASE_DIR)
    assert passed, errors
    assert summary["case_counts"]["structure"] >= 200
    assert summary["case_counts"]["pose"] >= 100
    assert summary["case_counts"]["binding_rank"] >= 25
    assert summary["case_counts"]["human_effect"] >= 200
    assert summary["contact_labeled_pose_cases"] >= 25
    assert isinstance(warnings, list)
    manifest = json.loads((RELEASE_DIR / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "PEB v1.0 release candidate"
    assert manifest["external_qc_claimed"] is False
    assert manifest["direct_human_effect_prediction_claimed"] is False
    assert manifest["medical_use_claimed"] is False
    assert manifest["qced"] is True
    assert manifest["source_metadata"] is True


def test_publish_check_rejects_false_external_qc_marker(tmp_path):
    output = _copy_release(tmp_path)
    cases_path = output / "human_effect" / "cases.jsonl"
    records = read_jsonl(cases_path)
    modified = deepcopy(records)
    modified[0]["qc_status"] = "external_qc"
    write_jsonl(cases_path, modified)

    passed, errors, _, _ = publish_check(output)
    assert not passed
    assert any("false external qc marker" in error for error in errors)


def test_publish_check_rejects_trial_without_results_positive(tmp_path):
    output = _copy_release(tmp_path)
    cases_path = output / "human_effect" / "cases.jsonl"
    records = read_jsonl(cases_path)
    modified = deepcopy(records)
    for record in modified:
        if record["source_database"] == "clinicaltrials" and record.get("trial_has_results") is False:
            record["evidence_direction"] = "positive"
            break
    write_jsonl(cases_path, modified)

    passed, errors, _, _ = publish_check(output)
    assert not passed
    assert any("clinical-trial no-results case marked positive" in error for error in errors)


def test_publish_check_rejects_contradictory_release_language(tmp_path):
    output = _copy_release(tmp_path)
    (output / "README.md").write_text(
        "This is release-ready.",
        encoding="utf-8",
    )

    passed, errors, _, _ = publish_check(output)

    assert not passed
    assert any("contains contradictory release language" in error for error in errors)


def test_publish_check_rejects_competitive_oracle_baseline(tmp_path):
    output = _copy_release(tmp_path)
    result_path = output / "baselines" / "results" / "human_effect_oracle_sanity_check.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["competitive"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    passed, errors, _, _ = publish_check(output)

    assert not passed
    assert any("oracle baseline appears in competitive leaderboard results" in error for error in errors)


def test_publish_check_rejects_unresolved_qc_disagreement(tmp_path):
    output = _copy_release(tmp_path)
    cases_path = output / "human_effect" / "cases.jsonl"
    cases = read_jsonl(cases_path)
    cases[0]["qc_disagreement"] = True
    cases[0]["qc_resolution"] = None
    write_jsonl(cases_path, cases)

    cards_path = output / "references" / "case_qc_cards.jsonl"
    cards = read_jsonl(cards_path)
    cards[0]["disagreements"] = [{"field": "evidence_level", "pass_a": "pass", "pass_b": "warn"}]
    cards[0]["qc_resolution"] = None
    write_jsonl(cards_path, cards)

    passed, errors, _, _ = publish_check(output)

    assert not passed
    assert any("qc disagreement lacks valid resolution" in error for error in errors)
    assert any("qc card disagreement lacks resolution" in error for error in errors)


def test_package_release_creates_archive_and_checksum(tmp_path):
    archive = tmp_path / "release.tar.gz"

    result = package_release(RELEASE_DIR, archive)

    assert archive.exists()
    assert Path(result["sha256_file"]).exists()
    assert result["sha256"]
    with tarfile.open(archive) as package:
        names = set(package.getnames())
    assert "peb-v1.0-rc/release_manifest.json" in names
    assert "peb-v1.0-rc/release_metadata_summary.json" in names
    assert not any(name.endswith(".md") for name in names)


def test_publish_check_rejects_legacy_public_release_doc(tmp_path):
    output = _copy_release(tmp_path)
    (output / "legacy_status.md").write_text(
        "Legacy compatibility note.",
        encoding="utf-8",
    )

    passed, errors, _, _ = publish_check(output)

    assert not passed
    assert any("markdown file is not allowed" in error for error in errors)


def test_package_release_fails_when_publish_check_fails(tmp_path):
    output = _copy_release(tmp_path)
    (output / "README.md").write_text(
        "This is release-ready.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="publish-check failed"):
        package_release(output, tmp_path / "bad-release.tar.gz")
