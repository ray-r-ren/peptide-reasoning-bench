import json
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "peb.cli", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_cli_help():
    result = _run_cli("--help")
    assert "Peptide Engineering Benchmark" in result.stdout


def test_cli_validation_baseline_eval_and_reports(tmp_path):
    _run_cli("validate", "data/fixtures/human_effect_fixture.jsonl")
    pred = tmp_path / "human_effect_baseline.jsonl"
    _run_cli(
        "make-baseline",
        "--track",
        "human_effect",
        "--input",
        "data/fixtures/human_effect_fixture.jsonl",
        "--output",
        str(pred),
    )
    result = _run_cli(
        "eval",
        "--track",
        "human_effect",
        "--gold",
        "data/fixtures/human_effect_fixture.jsonl",
        "--pred",
        str(pred),
    )
    assert json.loads(result.stdout)["track"] == "human_effect"

    _run_cli("manifest-check")
    card = tmp_path / "pdb.txt"
    _run_cli("source-card", "--source", "pdb", "--output", str(card))
    assert card.exists()

    templates = tmp_path / "templates"
    _run_cli("create-curation-templates", "--output-dir", str(templates))
    assert (templates / "pdb_pose_cases_template.csv").exists()

    release_dir = tmp_path / "release"
    _run_cli("build-release", "--output-dir", str(release_dir))
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "peb.cli",
            "release-check",
            "--input-dir",
            str(release_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    assert "below required minimum" in failed.stderr
    report = tmp_path / "release_report.txt"
    _run_cli("release-report", "--release-dir", str(release_dir), "--output", str(report))
    assert report.exists()
