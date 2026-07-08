"""Pydantic models and schema dispatch for PEB."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Track(str, Enum):
    structure = "structure"
    pose = "pose"
    binding_rank = "binding_rank"
    human_effect = "human_effect"


class Split(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


class QCStatus(str, Enum):
    unchecked = "unchecked"
    source_checked = "source_checked"
    external_qc = "external_qc"
    excluded = "excluded"


class SourceBucket(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class RedistributionValue(str, Enum):
    allowed = "allowed"
    restricted = "restricted"
    unknown = "unknown"


class LeaderboardUse(str, Enum):
    allowed = "allowed"
    caution = "caution"
    avoid = "avoid"


class AdapterStatus(str, Enum):
    planned = "planned"
    stub = "stub"
    implemented = "implemented"


class ReleaseMode(str, Enum):
    bundled = "bundled"
    derived = "derived"
    source_reference_only = "source_reference_only"
    excluded = "excluded"


class QCResult(str, Enum):
    passed_qc = "passed_qc"
    passed_with_warnings = "passed_with_warnings"
    downgraded_by_qc = "downgraded_by_qc"
    excluded_by_qc = "excluded_by_qc"


class ContactLabelStatus(str, Enum):
    computed_contacts = "computed_contacts"
    coordinate_reference_only = "coordinate_reference_only"
    source_interface_reference = "source_interface_reference"
    excluded = "excluded"


class AssayCompatibilityStatus(str, Enum):
    compatible = "compatible"
    compatible_with_normalization = "compatible_with_normalization"
    incompatible_excluded = "incompatible_excluded"


class EvidenceValidationStatus(str, Enum):
    source_supported = "source_supported"
    downgraded = "downgraded"
    insufficient_information = "insufficient_information"
    excluded = "excluded"


class QCResolution(str, Enum):
    kept_conservative_label = "kept_conservative_label"
    downgraded = "downgraded"
    excluded_from_primary_scoring = "excluded_from_primary_scoring"
    excluded = "excluded"


class ScoringSubset(str, Enum):
    primary = "primary"
    source_reference_only = "source_reference_only"
    contact_labeled_subset = "contact_labeled_subset"
    warning_only = "warning_only"
    excluded = "excluded"


class HumanEffectCategory(str, Enum):
    metabolic_weight_glucose = "metabolic_weight_glucose"
    endocrine_hormonal = "endocrine_hormonal"
    antimicrobial_antiinfective = "antimicrobial_antiinfective"
    anticancer_tumor_homing = "anticancer_tumor_homing"
    immune_vaccine_antigen_presentation = "immune_vaccine_antigen_presentation"
    neuro_cns = "neuro_cns"
    cardiovascular = "cardiovascular"
    reproductive = "reproductive"
    diagnostic_imaging = "diagnostic_imaging"
    toxic_adverse_effect_concern = "toxic_adverse_effect_concern"
    no_known_human_effect_evidence = "no_known_human_effect_evidence"


class EvidenceLevel(str, Enum):
    approved_human_indication = "approved_human_indication"
    human_clinical_evidence = "human_clinical_evidence"
    animal_preclinical_phenotype_evidence = "animal_preclinical_phenotype_evidence"
    in_vitro_target_activity_evidence = "in_vitro_target_activity_evidence"
    mechanistic_pathway_or_similarity_hypothesis = (
        "mechanistic_pathway_or_similarity_hypothesis"
    )
    unsupported_contradicted_or_unsafe_claim = "unsupported_contradicted_or_unsafe_claim"


class EvidenceDirection(str, Enum):
    positive = "positive"
    negative = "negative"
    mixed = "mixed"
    inconclusive = "inconclusive"
    not_reported = "not_reported"
    not_applicable = "not_applicable"


class ClaimStatus(str, Enum):
    supported = "supported"
    plausible_but_unproven = "plausible_but_unproven"
    unsupported = "unsupported"
    contradicted = "contradicted"
    unsafe_to_claim = "unsafe_to_claim"
    insufficient_information = "insufficient_information"


class SafetyStatus(str, Enum):
    known_acceptable_under_approved_use = "known_acceptable_under_approved_use"
    known_risk = "known_risk"
    serious_known_risk = "serious_known_risk"
    insufficient_safety_data = "insufficient_safety_data"
    not_for_human_use = "not_for_human_use"


class MeasurementDirection(str, Enum):
    lower_is_stronger = "lower_is_stronger"
    higher_is_stronger = "higher_is_stronger"
    unknown = "unknown"


class Peptide(StrictModel):
    sequence: str = Field(min_length=1)
    modifications: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def sequence_is_compact(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("peptide sequence must not contain whitespace")
        return value


class Protein(StrictModel):
    name: str
    accession: Optional[str] = None
    chain_id: Optional[str] = None
    sequence: Optional[str] = None


class Target(StrictModel):
    protein: Protein
    organism: Optional[str] = None
    target_family: Optional[str] = None


class EvidenceSource(StrictModel):
    source_database: str
    source_id: str
    source_url: Optional[str] = None
    citation: Optional[str] = None
    retrieval_date: Optional[str] = None


class SourceReference(StrictModel):
    source_database: str
    source_id: str
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    retrieval_date: Optional[str] = None
    citation: Optional[str] = None


class RedistributionPolicy(StrictModel):
    raw_data_redistribution: RedistributionValue
    processed_label_redistribution: RedistributionValue
    commercial_use: RedistributionValue
    attribution_required: bool
    share_alike_obligation: bool
    use_in_public_leaderboard: LeaderboardUse


class LeakageGroup(StrictModel):
    peptide_cluster: Optional[str] = None
    target_family: Optional[str] = None
    interface_cluster: Optional[str] = None
    source_release_date: Optional[str] = None
    panel_id: Optional[str] = None


class Coordinate(StrictModel):
    atom_id: str
    residue_id: str
    x: float
    y: float
    z: float


class ContactPair(StrictModel):
    target_residue: str
    peptide_residue: str
    distance_angstrom: Optional[float] = Field(default=None, ge=0)

    @field_validator("target_residue", "peptide_residue")
    @classmethod
    def residue_pair_has_chain_separator(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("residue labels must use '<chain>:<residue>' format")
        return value


class BenchmarkCase(StrictModel):
    benchmark_id: str
    track: Track
    release_mode: ReleaseMode
    source_database: str
    source_id: str
    source_url: Optional[str] = None
    source_version: str
    retrieval_date: str
    license_or_usage_note: str
    redistribution_policy_snapshot: RedistributionPolicy
    curator_notes: str = ""
    qc_status: QCStatus
    qc_result: Optional[QCResult] = None
    qc_notes: list[str] = Field(default_factory=list)
    qc_disagreement: bool = False
    qc_resolution: Optional[QCResolution] = None
    scoring_subset: Optional[ScoringSubset] = None
    exclusion_reason: Optional[str] = None
    citation: Optional[str] = None
    evidence_quote_or_label: Optional[str] = None
    source_record_hash: Optional[str] = None
    processed_record_hash: Optional[str] = None
    split: Split
    leakage_group: LeakageGroup

    @field_validator("evidence_quote_or_label")
    @classmethod
    def concise_quote_only(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and len(value.split()) > 35:
            raise ValueError("evidence quote or label must be concise")
        return value


class StructureCase(BenchmarkCase):
    track: Literal[Track.structure] = Track.structure
    peptide: Peptide
    structure_id: str
    chain_id: Optional[str] = None
    entity_id: Optional[str] = None
    peptide_length: Optional[int] = None
    structure_file_url: Optional[str] = None
    experimental_method: str
    resolution_angstrom: Optional[float] = Field(default=None, gt=0)
    gold_structure_reference: SourceReference
    gold_coordinates: list[Coordinate] = Field(default_factory=list)
    confidence_notes: Optional[str] = None


class PoseCase(BenchmarkCase):
    track: Literal[Track.pose] = Track.pose
    peptide: Peptide
    target: Target
    pdb_id: str
    target_chain_id: str
    peptide_chain_id: str
    peptide_length: Optional[int] = None
    target_length: Optional[int] = None
    contact_map_method: Optional[str] = None
    coordinate_reference: Optional[str] = None
    native_contacts: list[ContactPair] = Field(default_factory=list)
    binding_site_residues: list[str] = Field(default_factory=list)
    orientation_label: Optional[str] = None
    pose_subset: Optional[str] = None
    contact_label_status: Optional[ContactLabelStatus] = None


class BindingRankItem(StrictModel):
    item_id: str
    peptide: Peptide
    measured_value: float
    normalized_rank: float = Field(ge=0, le=1)


class BindingRankCase(BenchmarkCase):
    track: Literal[Track.binding_rank] = Track.binding_rank
    panel_id: str
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    assay_type: str
    assay_unit: str
    assay_conditions: str
    measurement_direction: MeasurementDirection
    normalization_method: str
    comparable_panel: bool
    panel_exclusion_reason: Optional[str] = None
    items: list[BindingRankItem] = Field(min_length=2)
    source_ids: list[str] = Field(default_factory=list)
    assay_compatibility_status: Optional[AssayCompatibilityStatus] = None


class HumanEffectCase(BenchmarkCase):
    track: Literal[Track.human_effect] = Track.human_effect
    peptide: Peptide
    claim_text: str
    category: HumanEffectCategory
    evidence_level: EvidenceLevel
    evidence_direction: EvidenceDirection
    claim_status: ClaimStatus
    safety_status: SafetyStatus
    source_evidence_type: str
    target: Optional[str] = None
    mechanism_support: Optional[str] = None
    source_result_count: Optional[int] = None
    trial_status: Optional[str] = None
    trial_phase: Optional[str] = None
    trial_has_results: Optional[bool] = None
    evidence_validation_status: Optional[EvidenceValidationStatus] = None


Case = Annotated[
    Union[StructureCase, PoseCase, BindingRankCase, HumanEffectCase],
    Field(discriminator="track"),
]


class PredictionBase(StrictModel):
    prediction_id: str
    benchmark_id: str
    track: Track
    model_name: str = "unspecified"


class StructurePrediction(PredictionBase):
    track: Literal[Track.structure] = Track.structure
    coordinates: list[Coordinate] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class PosePrediction(PredictionBase):
    track: Literal[Track.pose] = Track.pose
    predicted_contacts: list[ContactPair] = Field(default_factory=list)
    binding_site_residues: list[str] = Field(default_factory=list)
    orientation_label: Optional[str] = None
    clash_score: Optional[float] = Field(default=None, ge=0)


class BindingRankScore(StrictModel):
    item_id: str
    score: float
    rank: Optional[int] = Field(default=None, ge=1)


class BindingRankPrediction(PredictionBase):
    track: Literal[Track.binding_rank] = Track.binding_rank
    scores: list[BindingRankScore] = Field(min_length=1)


class HumanEffectPrediction(PredictionBase):
    track: Literal[Track.human_effect] = Track.human_effect
    category: HumanEffectCategory
    evidence_level: EvidenceLevel
    evidence_direction: EvidenceDirection
    claim_status: ClaimStatus
    safety_status: SafetyStatus
    abstained: bool = False
    rationale_source_ids: list[str] = Field(default_factory=list)


Prediction = Annotated[
    Union[StructurePrediction, PosePrediction, BindingRankPrediction, HumanEffectPrediction],
    Field(discriminator="track"),
]


class EvaluationResult(StrictModel):
    track: Track
    n_cases: int
    n_predictions: int
    metrics: dict[str, Union[float, int, str]]
    warnings: list[str] = Field(default_factory=list)


class SourceManifestEntry(StrictModel):
    name: str
    module: str
    url: HttpUrl
    source_category: str
    expected_fields: list[str]
    license_or_usage_note: str
    citation: str
    source_version: str
    retrieval_date_policy: str
    redistribution_policy: RedistributionPolicy
    adapter_status: AdapterStatus
    comments: str
    source_bucket: SourceBucket


class ReleaseManifest(StrictModel):
    benchmark_name: str
    benchmark_abbreviation: str
    release_id: str
    release_date: str
    release_status: str
    case_counts: dict[str, int]
    source_manifest_snapshot: str
    limitations: list[str] = Field(default_factory=list)
    files: list[str]


class AuditResult(StrictModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_case_record(record: dict[str, Any]) -> BenchmarkCase:
    track = record.get("track")
    model = {
        Track.structure.value: StructureCase,
        Track.pose.value: PoseCase,
        Track.binding_rank.value: BindingRankCase,
        Track.human_effect.value: HumanEffectCase,
    }.get(track)
    if model is None:
        raise ValueError(f"unknown case track: {track!r}")
    return model.model_validate(record)


def validate_prediction_record(record: dict[str, Any]) -> PredictionBase:
    track = record.get("track")
    model = {
        Track.structure.value: StructurePrediction,
        Track.pose.value: PosePrediction,
        Track.binding_rank.value: BindingRankPrediction,
        Track.human_effect.value: HumanEffectPrediction,
    }.get(track)
    if model is None:
        raise ValueError(f"unknown prediction track: {track!r}")
    return model.model_validate(record)


def model_to_record(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def json_schema_for_cases() -> dict[str, Any]:
    return BenchmarkCase.model_json_schema()


def json_schema_for_predictions() -> dict[str, Any]:
    return PredictionBase.model_json_schema()
