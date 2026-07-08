"""Deterministic weak baselines."""

from __future__ import annotations

import random
from typing import Any

from peb.schemas import (
    BindingRankCase,
    BindingRankPrediction,
    BindingRankScore,
    ClaimStatus,
    EvidenceDirection,
    EvidenceLevel,
    HumanEffectCase,
    HumanEffectCategory,
    HumanEffectPrediction,
    PoseCase,
    PosePrediction,
    SafetyStatus,
    StructureCase,
    StructurePrediction,
    Track,
)


def _prediction_id(case_id: str, name: str) -> str:
    return f"{name}:{case_id}"


def _is_oracle_name(name: str) -> bool:
    return "oracle" in name and "non_oracle" not in name


def _structure(case: StructureCase, name: str) -> dict[str, Any]:
    return StructurePrediction(
        prediction_id=_prediction_id(case.benchmark_id, name),
        benchmark_id=case.benchmark_id,
        model_name=name,
        coordinates=case.gold_coordinates,
        confidence=0.5,
    ).model_dump(mode="json", exclude_none=True)


def _pose(case: PoseCase, name: str) -> dict[str, Any]:
    return PosePrediction(
        prediction_id=_prediction_id(case.benchmark_id, name),
        benchmark_id=case.benchmark_id,
        model_name=name,
        predicted_contacts=case.native_contacts[:1],
        binding_site_residues=case.binding_site_residues[:1],
        orientation_label=case.orientation_label,
        clash_score=0.0,
    ).model_dump(mode="json", exclude_none=True)


def _binding_rank(case: BindingRankCase, name: str, rng: random.Random) -> dict[str, Any]:
    scores = []
    for index, item in enumerate(case.items, start=1):
        scores.append(BindingRankScore(item_id=item.item_id, score=rng.random(), rank=index))
    return BindingRankPrediction(
        prediction_id=_prediction_id(case.benchmark_id, name),
        benchmark_id=case.benchmark_id,
        model_name=name,
        scores=scores,
    ).model_dump(mode="json", exclude_none=True)


def _human_effect_oracle(case: HumanEffectCase, name: str) -> dict[str, Any]:
    abstain = case.claim_status in {
        ClaimStatus.unsupported,
        ClaimStatus.unsafe_to_claim,
        ClaimStatus.insufficient_information,
    }
    return HumanEffectPrediction(
        prediction_id=_prediction_id(case.benchmark_id, name),
        benchmark_id=case.benchmark_id,
        model_name=name,
        category=case.category if not abstain else HumanEffectCategory.no_known_human_effect_evidence,
        evidence_level=case.evidence_level,
        evidence_direction=case.evidence_direction,
        claim_status=ClaimStatus.insufficient_information if abstain else case.claim_status,
        safety_status=SafetyStatus.insufficient_safety_data if abstain else case.safety_status,
        abstained=abstain,
        rationale_source_ids=[case.source_id],
    ).model_dump(mode="json", exclude_none=True)


def _human_effect(case: HumanEffectCase, name: str) -> dict[str, Any]:
    source = case.source_database.lower()
    category = HumanEffectCategory.no_known_human_effect_evidence
    evidence_level = EvidenceLevel.unsupported_contradicted_or_unsafe_claim
    evidence_direction = EvidenceDirection.not_applicable
    claim_status = ClaimStatus.insufficient_information
    safety_status = SafetyStatus.insufficient_safety_data
    abstain = True

    if source == "dailymed":
        category = HumanEffectCategory.endocrine_hormonal
        evidence_level = EvidenceLevel.approved_human_indication
        evidence_direction = EvidenceDirection.positive
        claim_status = ClaimStatus.supported
        safety_status = SafetyStatus.known_acceptable_under_approved_use
        abstain = False
    elif source == "clinicaltrials":
        evidence_level = EvidenceLevel.human_clinical_evidence
        evidence_direction = EvidenceDirection.not_reported
        claim_status = ClaimStatus.plausible_but_unproven
    elif source in {"reactome", "gene_ontology", "opentargets"}:
        evidence_level = EvidenceLevel.mechanistic_pathway_or_similarity_hypothesis
        evidence_direction = EvidenceDirection.not_applicable
        claim_status = ClaimStatus.plausible_but_unproven

    return HumanEffectPrediction(
        prediction_id=_prediction_id(case.benchmark_id, name),
        benchmark_id=case.benchmark_id,
        model_name=name,
        category=category,
        evidence_level=evidence_level,
        evidence_direction=evidence_direction,
        claim_status=claim_status,
        safety_status=safety_status,
        abstained=abstain,
        rationale_source_ids=[case.source_id],
    ).model_dump(mode="json", exclude_none=True)


def make_baseline_predictions(
    track: Track, cases: list[Any], seed: int = 13, model_name: str = "weak_seeded_baseline"
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if track == Track.structure:
        return [_structure(case, model_name) for case in cases]
    if track == Track.pose:
        return [_pose(case, model_name) for case in cases]
    if track == Track.binding_rank:
        return [_binding_rank(case, model_name, rng) for case in cases]
    if track == Track.human_effect:
        if _is_oracle_name(model_name):
            return [_human_effect_oracle(case, model_name) for case in cases]
        return [_human_effect(case, model_name) for case in cases]
    raise ValueError(f"unsupported track: {track}")
