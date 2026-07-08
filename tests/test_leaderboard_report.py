import json

from peb.reports.leaderboard import render_leaderboard


def test_leaderboard_deduplicates_and_excludes_oracle(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    competitive = {
        "track": "human_effect",
        "model_name": "human_effect_non_oracle_baseline",
        "competitive": True,
        "metrics": {"category_macro_f1": 0.25},
    }
    duplicate = dict(competitive)
    oracle = {
        "track": "human_effect",
        "model_name": "human_effect_oracle_source_reference_baseline",
        "competitive": False,
        "leaderboard_group": "oracle_sanity_check",
        "metrics": {"category_macro_f1": 1.0},
    }
    for name, payload in {
        "a.json": competitive,
        "b.json": duplicate,
        "oracle.json": oracle,
    }.items():
        (result_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    rendered = render_leaderboard(result_dir)

    assert rendered.count("human_effect_non_oracle_baseline") == 1
    assert "| human_effect_non_oracle_baseline | human_effect | category_macro_f1 | 0.25 |" in rendered
    assert "| human_effect_oracle_source_reference_baseline | human_effect | category_macro_f1 | 1.0 | excluded |" in rendered
