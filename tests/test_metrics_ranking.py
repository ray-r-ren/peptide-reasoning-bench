from peb.io import read_jsonl
from peb.metrics.ranking_metrics import evaluate_binding_rank, pairwise_accuracy, spearman
from peb.schemas import BindingRankPrediction, BindingRankScore, validate_case_record


def test_ranking_metrics_perfect_order():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1
    assert pairwise_accuracy([1, 2, 3], [1, 2, 3]) == 1


def test_binding_rank_eval_fixture():
    case = validate_case_record(read_jsonl("data/fixtures/binding_rank_fixture.jsonl")[0])
    pred = BindingRankPrediction(
        prediction_id="bind-pred",
        benchmark_id=case.benchmark_id,
        scores=[
            BindingRankScore(item_id="pep-a", score=3, rank=1),
            BindingRankScore(item_id="pep-b", score=2, rank=2),
            BindingRankScore(item_id="pep-c", score=1, rank=3),
        ],
    )
    result = evaluate_binding_rank([case], [pred])
    assert result.metrics["pairwise_ranking_accuracy"] == 1


def test_mixed_assay_panel_is_flagged():
    case = validate_case_record(read_jsonl("data/fixtures/binding_rank_fixture.jsonl")[0])
    changed = case.model_copy(update={"comparable_panel": False, "panel_exclusion_reason": "mixed"})
    pred = BindingRankPrediction(
        prediction_id="bind-pred",
        benchmark_id=case.benchmark_id,
        scores=[BindingRankScore(item_id=item.item_id, score=1.0) for item in case.items],
    )
    result = evaluate_binding_rank([changed], [pred])
    assert result.metrics["assay_aware_subgroup_count"] == 0
    assert result.warnings

