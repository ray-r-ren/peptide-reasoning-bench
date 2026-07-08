from peb.io import read_jsonl
from peb.metrics.structure_metrics import evaluate_structure, rmsd
from peb.schemas import Coordinate, StructureCase, StructurePrediction, validate_case_record


def test_structure_rmsd_simple_coordinates():
    gold = [Coordinate(atom_id="a", residue_id="A:1", x=0, y=0, z=0)]
    pred = [Coordinate(atom_id="a", residue_id="A:1", x=1, y=0, z=0)]
    assert rmsd(gold, pred) == 1


def test_structure_eval_fixture():
    case = validate_case_record(read_jsonl("data/fixtures/structure_fixture.jsonl")[0])
    prediction = StructurePrediction(
        prediction_id="p",
        benchmark_id=case.benchmark_id,
        coordinates=case.gold_coordinates,
    )
    result = evaluate_structure([case], [prediction])
    assert result.metrics["backbone_rmsd"] == 0
    assert isinstance(case, StructureCase)

