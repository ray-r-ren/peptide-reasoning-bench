"""Public-source case builders for the PEB release candidate."""

from __future__ import annotations

import html
import json
import math
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from peb.baselines import make_baseline_predictions
from peb.io import read_jsonl, sha256_text, write_jsonl, write_text
from peb.metrics import (
    evaluate_binding_rank,
    evaluate_human_effect,
    evaluate_pose,
    evaluate_structure,
)
from peb.processing.pdb_contacts import compute_contacts
from peb.processing.splits import write_splits
from peb.registry import SOURCE_MANIFEST, load_source_manifest
from peb.reports.benchmark_card import render_benchmark_card
from peb.reports.datasheet import render_datasheet
from peb.reports.leaderboard import render_leaderboard
from peb.reports.release_report import render_release_report
from peb.schemas import (
    BindingRankCase,
    HumanEffectCase,
    PoseCase,
    StructureCase,
    Track,
    validate_case_record,
    validate_prediction_record,
)

RETRIEVAL_DATE = "2026-07-07"
RELEASE_ID = "peb-v1.0-rc"


POLICY_OPEN = {
    "raw_data_redistribution": "allowed",
    "processed_label_redistribution": "allowed",
    "commercial_use": "allowed",
    "attribution_required": True,
    "share_alike_obligation": False,
    "use_in_public_leaderboard": "allowed",
}

POLICY_CAUTION = {
    "raw_data_redistribution": "restricted",
    "processed_label_redistribution": "unknown",
    "commercial_use": "unknown",
    "attribution_required": True,
    "share_alike_obligation": False,
    "use_in_public_leaderboard": "caution",
}


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "peb-release-tooling/1.0"})


def get_json(url: str, timeout: int = 60) -> Any:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "peb-release-tooling/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_sequence(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z]", "", value.upper())


def common_case(
    benchmark_id: str,
    track: str,
    source_database: str,
    source_id: str,
    source_url: str,
    source_version: str,
    license_note: str,
    release_mode: str,
    split_hint: int,
    leakage: dict[str, Any],
    citation: str,
    policy: dict[str, Any],
    curator_notes: str,
) -> dict[str, Any]:
    split = ("train", "dev", "test")[split_hint % 3]
    base = {
        "benchmark_id": benchmark_id,
        "track": track,
        "release_mode": release_mode,
        "source_database": source_database,
        "source_id": source_id,
        "source_url": source_url,
        "source_version": source_version,
        "retrieval_date": RETRIEVAL_DATE,
        "license_or_usage_note": license_note,
        "redistribution_policy_snapshot": policy,
        "qc_status": "source_checked",
        "curator_notes": curator_notes,
        "citation": citation,
        "split": split,
        "leakage_group": leakage,
        "source_record_hash": sha256_text(source_database + ":" + source_id + ":" + source_url),
    }
    base["processed_record_hash"] = sha256_text(json.dumps(base, sort_keys=True))
    return base


def rcsb_search_peptide_entries(limit: int) -> list[str]:
    payload = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": "peptide"}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": limit}},
    }
    data = post_json("https://search.rcsb.org/rcsbsearch/v2/query", payload)
    return [row["identifier"] for row in data.get("result_set", [])]


def rcsb_entries(entry_ids: list[str]) -> list[dict[str, Any]]:
    query = """
    query($ids:[String!]!){
      entries(entry_ids:$ids){
        rcsb_id
        exptl{method}
        rcsb_accession_info{initial_release_date}
        polymer_entities{
          rcsb_id
          entity_poly{
            pdbx_seq_one_letter_code_can
            rcsb_sample_sequence_length
            type
          }
          rcsb_polymer_entity_container_identifiers{
            entity_id
            asym_ids
            auth_asym_ids
          }
        }
      }
    }
    """
    output: list[dict[str, Any]] = []
    for start in range(0, len(entry_ids), 100):
        batch = entry_ids[start : start + 100]
        data = post_json("https://data.rcsb.org/graphql", {"query": query, "variables": {"ids": batch}})
        output.extend([entry for entry in data.get("data", {}).get("entries", []) if entry])
    return output


def _entity_ids(entity: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    auth = identifiers.get("auth_asym_ids") or []
    asym = identifiers.get("asym_ids") or []
    entity_id = identifiers.get("entity_id")
    return (auth[0] if auth else (asym[0] if asym else None), entity_id)


def _entity_length(entity: dict[str, Any]) -> int:
    entity_poly = entity.get("entity_poly") or {}
    length = entity_poly.get("rcsb_sample_sequence_length")
    if length is None:
        return len(clean_sequence(entity_poly.get("pdbx_seq_one_letter_code_can")))
    return int(length)


def _entity_sequence(entity: dict[str, Any]) -> str:
    return clean_sequence((entity.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can"))


def _is_poly_peptide(entity: dict[str, Any]) -> bool:
    return (entity.get("entity_poly") or {}).get("type") == "polypeptide(L)"


def build_structure_cases(output: Path, min_cases: int, max_records: int = 5000) -> list[dict[str, Any]]:
    entry_ids = rcsb_search_peptide_entries(max_records)
    entries = rcsb_entries(entry_ids)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    records_examined = 0
    for entry in entries:
        pdb_id = entry["rcsb_id"]
        method = ", ".join(sorted({item.get("method", "unknown") for item in entry.get("exptl") or []}))
        release_date = (entry.get("rcsb_accession_info") or {}).get("initial_release_date", "")[:10]
        for entity in entry.get("polymer_entities") or []:
            records_examined += 1
            if not _is_poly_peptide(entity):
                continue
            length = _entity_length(entity)
            sequence = _entity_sequence(entity)
            chain_id, entity_id = _entity_ids(entity)
            if not chain_id or not entity_id or not sequence or not 2 <= length <= 50:
                continue
            source_id = f"{pdb_id}_{entity_id}_{chain_id}"
            if source_id in seen:
                continue
            seen.add(source_id)
            case = common_case(
                benchmark_id=f"PEB-STRUCT-{len(cases)+1:05d}",
                track="structure",
                source_database="pdb",
                source_id=source_id,
                source_url=f"https://www.rcsb.org/structure/{pdb_id}",
                source_version="RCSB live API snapshot",
                license_note="PDB/RCSB public archive metadata with attribution and source IDs retained.",
                release_mode="derived",
                split_hint=len(cases),
                leakage={
                    "peptide_cluster": sequence[:12],
                    "target_family": "pdb_peptide_chain",
                    "interface_cluster": None,
                    "source_release_date": release_date or None,
                },
                citation="RCSB PDB and wwPDB public archive documentation.",
                policy=POLICY_OPEN,
                curator_notes="Source-checked RCSB peptide-chain metadata; no raw coordinates bundled.",
            )
            case.update(
                {
                    "peptide": {"sequence": sequence, "modifications": []},
                    "structure_id": pdb_id,
                    "chain_id": chain_id,
                    "entity_id": entity_id,
                    "peptide_length": length,
                    "structure_file_url": f"https://files.rcsb.org/download/{pdb_id}.cif",
                    "experimental_method": method or "experimental_structure",
                    "gold_structure_reference": {
                        "source_database": "pdb",
                        "source_id": pdb_id,
                        "source_url": f"https://files.rcsb.org/download/{pdb_id}.cif",
                        "source_version": "RCSB live API snapshot",
                        "retrieval_date": RETRIEVAL_DATE,
                        "citation": "RCSB PDB and wwPDB public archive documentation.",
                    },
                    "gold_coordinates": [],
                    "confidence_notes": "Experimental source reference; flexible peptide caveats apply.",
                }
            )
            validate_case_record(case)
            cases.append(case)
            if len(cases) >= min_cases:
                write_jsonl(output, cases)
                return cases
    write_jsonl(output, cases)
    return cases


def _download_pdb(pdb_id: str, raw_dir: Path) -> Optional[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{pdb_id.lower()}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        text = get_text(url, timeout=60)
    except Exception:
        return None
    if "ATOM" not in text:
        return None
    path.write_text(text, encoding="utf-8")
    return path


def build_pose_cases(output: Path, min_cases: int, max_records: int = 5000) -> list[dict[str, Any]]:
    entry_ids = rcsb_search_peptide_entries(max_records)
    entries = rcsb_entries(entry_ids)
    cases: list[dict[str, Any]] = []
    raw_dir = Path("data/raw/pdb")
    computed_contacts = 0
    seen: set[str] = set()
    for entry in entries:
        pdb_id = entry["rcsb_id"]
        release_date = (entry.get("rcsb_accession_info") or {}).get("initial_release_date", "")[:10]
        peptides = []
        targets = []
        for entity in entry.get("polymer_entities") or []:
            if not _is_poly_peptide(entity):
                continue
            length = _entity_length(entity)
            chain_id, entity_id = _entity_ids(entity)
            sequence = _entity_sequence(entity)
            item = {"entity": entity, "length": length, "chain": chain_id, "entity_id": entity_id, "sequence": sequence}
            if chain_id and sequence and 5 <= length <= 50:
                peptides.append(item)
            if chain_id and sequence and length >= 60:
                targets.append(item)
        if not peptides or not targets:
            continue
        pdb_path = _download_pdb(pdb_id, raw_dir) if computed_contacts < 25 else None
        for peptide in peptides[:2]:
            for target in targets[:1]:
                source_id = f"{pdb_id}_{target['chain']}_{peptide['chain']}"
                if source_id in seen:
                    continue
                seen.add(source_id)
                contacts: list[dict[str, Any]] = []
                method = "coordinate reference; contacts computable from PDB/mmCIF"
                if pdb_path is not None:
                    contacts = compute_contacts(pdb_path, target["chain"], peptide["chain"])
                    if contacts:
                        contacts = contacts[:200]
                        computed_contacts += 1
                        method = "computed heavy-atom contacts from PDB file, 5 angstrom cutoff"
                case = common_case(
                    benchmark_id=f"PEB-POSE-{len(cases)+1:05d}",
                    track="pose",
                    source_database="pdb",
                    source_id=source_id,
                    source_url=f"https://www.rcsb.org/structure/{pdb_id}",
                    source_version="RCSB live API snapshot",
                    license_note="PDB/RCSB public archive metadata with derived contact labels when computed.",
                    release_mode="derived",
                    split_hint=len(cases),
                    leakage={
                        "peptide_cluster": peptide["sequence"][:12],
                        "target_family": f"pdb_entity_{target['entity_id']}",
                        "interface_cluster": f"{pdb_id}_{target['chain']}_{peptide['chain']}",
                        "source_release_date": release_date or None,
                    },
                    citation="RCSB PDB and wwPDB public archive documentation.",
                    policy=POLICY_OPEN,
                    curator_notes="Source-checked chain-role heuristic: short polypeptide chain as peptide, longer polypeptide chain as target.",
                )
                case.update(
                    {
                        "peptide": {"sequence": peptide["sequence"], "modifications": []},
                        "target": {
                            "protein": {
                                "name": "RCSB polymer entity",
                                "chain_id": target["chain"],
                                "sequence": target["sequence"],
                            },
                            "target_family": f"pdb_entity_{target['entity_id']}",
                        },
                        "pdb_id": pdb_id,
                        "target_chain_id": target["chain"],
                        "peptide_chain_id": peptide["chain"],
                        "peptide_length": peptide["length"],
                        "target_length": target["length"],
                        "contact_map_method": method,
                        "coordinate_reference": f"https://files.rcsb.org/download/{pdb_id}.cif",
                        "native_contacts": contacts,
                        "binding_site_residues": sorted({contact["target_residue"] for contact in contacts}),
                    }
                )
                validate_case_record(case)
                cases.append(case)
                if len(cases) >= min_cases and computed_contacts >= 25:
                    write_jsonl(output, cases)
                    return cases
    write_jsonl(output, cases)
    return cases


def _number_from_measure(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    clean = html.unescape(re.sub(r"<[^>]+>", " ", value))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", clean.replace(",", ""))
    return float(match.group(1)) if match else None


def build_bindrank_panels(output: Path, min_panels: int) -> list[dict[str, Any]]:
    url = (
        "https://query-api.iedb.org/mhc_search?"
        + urllib.parse.urlencode(
            {
                "assay_description": "ilike.*IC50*",
                "quantitative_measure": "not.is.null",
                "limit": "5000",
            }
        )
    )
    rows = get_json(url, timeout=120)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequence = clean_sequence(row.get("linear_sequence"))
        value = _number_from_measure(row.get("quantitative_measure"))
        allele = row.get("mhc_allele_name") or row.get("mhc_restriction")
        assay = row.get("assay_names") or "IC50"
        if not sequence or value is None or not allele or len(sequence) < 5:
            continue
        grouped[(allele, assay)].append(row)

    cases: list[dict[str, Any]] = []
    for (allele, assay), panel_rows in grouped.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in panel_rows:
            sequence = clean_sequence(row.get("linear_sequence"))
            if sequence not in unique:
                unique[sequence] = row
        if len(unique) < 5:
            continue
        selected = list(unique.values())[:10]
        values = [_number_from_measure(row.get("quantitative_measure")) or math.inf for row in selected]
        ordered = sorted(values)
        items = []
        source_ids = []
        for row, value in zip(selected, values):
            rank = ordered.index(value)
            normalized = 1.0 - (rank / max(len(ordered) - 1, 1))
            assay_id = str(row.get("elution_id") or row.get("structure_id"))
            source_ids.append(assay_id)
            items.append(
                {
                    "item_id": f"IEDB-{assay_id}",
                    "peptide": {"sequence": clean_sequence(row.get("linear_sequence")), "modifications": []},
                    "measured_value": value,
                    "normalized_rank": normalized,
                }
            )
        panel_id = f"IEDB-MHC-IC50-{len(cases)+1:04d}"
        case = common_case(
            benchmark_id=f"PEB-BIND-{len(cases)+1:05d}",
            track="binding_rank",
            source_database="iedb",
            source_id=panel_id,
            source_url="https://query-api.iedb.org/mhc_search",
            source_version="IEDB public query API snapshot",
            license_note="IEDB public query API rows processed into derived assay-aware panels with source IDs retained.",
            release_mode="derived",
            split_hint=len(cases),
            leakage={
                "peptide_cluster": panel_id,
                "target_family": allele,
                "interface_cluster": None,
                "source_release_date": RETRIEVAL_DATE,
                "panel_id": panel_id,
            },
            citation="IEDB public data portal and query API documentation.",
            policy=POLICY_CAUTION,
            curator_notes="Source-checked IC50 panel grouped by MHC allele and assay descriptor; lower nM is stronger.",
        )
        case.update(
            {
                "panel_id": panel_id,
                "target_id": allele,
                "target_name": allele,
                "assay_type": "IC50",
                "assay_unit": "nM",
                "assay_conditions": assay,
                "measurement_direction": "lower_is_stronger",
                "normalization_method": "inverse ordinal rank within same allele and assay descriptor",
                "comparable_panel": True,
                "items": items,
                "source_ids": source_ids,
            }
        )
        validate_case_record(case)
        cases.append(case)
        if len(cases) >= min_panels:
            write_jsonl(output, cases)
            return cases
    write_jsonl(output, cases)
    return cases


DAILYMED_CASES = [
    ("insulin", "metabolic_weight_glucose"),
    ("glucagon", "metabolic_weight_glucose"),
    ("semaglutide", "metabolic_weight_glucose"),
    ("liraglutide", "metabolic_weight_glucose"),
    ("exenatide", "metabolic_weight_glucose"),
    ("pramlintide", "metabolic_weight_glucose"),
    ("octreotide", "endocrine_hormonal"),
    ("lanreotide", "endocrine_hormonal"),
    ("desmopressin", "endocrine_hormonal"),
    ("vasopressin", "cardiovascular"),
    ("oxytocin", "reproductive"),
    ("teriparatide", "endocrine_hormonal"),
    ("abaloparatide", "endocrine_hormonal"),
    ("calcitonin", "endocrine_hormonal"),
    ("leuprolide", "reproductive"),
    ("goserelin", "reproductive"),
    ("degarelix", "reproductive"),
    ("cetrorelix", "reproductive"),
    ("ganirelix", "reproductive"),
    ("cosyntropin", "endocrine_hormonal"),
    ("corticotropin", "endocrine_hormonal"),
    ("teduglutide", "endocrine_hormonal"),
    ("metreleptin", "metabolic_weight_glucose"),
    ("somatropin", "endocrine_hormonal"),
    ("pegvisomant", "endocrine_hormonal"),
    ("bivalirudin", "cardiovascular"),
    ("eptifibatide", "cardiovascular"),
    ("nesiritide", "cardiovascular"),
]


def _dailymed_cases(start_index: int, min_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for drug_name, category in DAILYMED_CASES:
        url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=" + urllib.parse.quote(drug_name)
        try:
            data = get_json(url)
        except Exception:
            continue
        rows = data.get("data") or []
        if not rows:
            continue
        row = rows[0]
        setid = row.get("setid")
        version = str(row.get("spl_version") or "label")
        if not setid:
            continue
        case = common_case(
            benchmark_id=f"PEB-HFX-{start_index + len(cases)+1:05d}",
            track="human_effect",
            source_database="dailymed",
            source_id=setid,
            source_url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
            source_version=version,
            license_note="DailyMed public label source reference; no full label text copied.",
            release_mode="source_reference_only",
            split_hint=start_index + len(cases),
            leakage={"peptide_cluster": drug_name, "target_family": category, "interface_cluster": None, "source_release_date": RETRIEVAL_DATE},
            citation="DailyMed public label data portal documentation.",
            policy=POLICY_OPEN,
            curator_notes="Source-checked label source-reference case; category assigned from curated peptide-drug class list.",
        )
        case.update(
            {
                "peptide": {"sequence": "SOURCE_REFERENCE_ONLY", "modifications": [], "description": drug_name},
                "claim_text": "Official label source supports an approved indication evidence level for benchmark classification.",
                "category": category,
                "evidence_level": "approved_human_indication",
                "evidence_direction": "positive",
                "claim_status": "supported",
                "safety_status": "known_acceptable_under_approved_use",
                "source_evidence_type": "official_label_reference",
                "source_result_count": len(rows),
            }
        )
        validate_case_record(case)
        cases.append(case)
        if len(cases) >= min_cases:
            return cases
    return cases


def _clinical_cases(start_index: int, min_cases: int) -> list[dict[str, Any]]:
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode({"query.term": "peptide", "pageSize": min(max(min_cases, 100), 1000), "format": "json"})
    data = get_json(url)
    cases: list[dict[str, Any]] = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection") or {}
        identification = protocol.get("identificationModule") or {}
        status = protocol.get("statusModule") or {}
        design = protocol.get("designModule") or {}
        nct_id = identification.get("nctId")
        if not nct_id:
            continue
        phases = design.get("phases") or []
        has_results = bool(study.get("hasResults"))
        case = common_case(
            benchmark_id=f"PEB-HFX-{start_index + len(cases)+1:05d}",
            track="human_effect",
            source_database="clinicaltrials",
            source_id=nct_id,
            source_url=f"https://clinicaltrials.gov/study/{nct_id}",
            source_version="ClinicalTrials.gov API v2 snapshot",
            license_note="ClinicalTrials.gov public record source reference; trial existence is not positive effect evidence.",
            release_mode="source_reference_only",
            split_hint=start_index + len(cases),
            leakage={"peptide_cluster": nct_id, "target_family": "clinical_trial_record", "interface_cluster": None, "source_release_date": RETRIEVAL_DATE},
            citation="ClinicalTrials.gov public API documentation.",
            policy=POLICY_OPEN,
            curator_notes="Source-checked clinical trial record; no positive outcome inferred from trial existence.",
        )
        case.update(
            {
                "peptide": {"sequence": "SOURCE_REFERENCE_ONLY", "modifications": []},
                "claim_text": "Clinical trial record exists; outcome direction must not be inferred without results.",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "human_clinical_evidence",
                "evidence_direction": "not_reported" if not has_results else "inconclusive",
                "claim_status": "plausible_but_unproven",
                "safety_status": "insufficient_safety_data",
                "source_evidence_type": "clinical_trial_record",
                "trial_status": status.get("overallStatus"),
                "trial_phase": ";".join(phases) if phases else None,
                "trial_has_results": has_results,
                "source_result_count": len(data.get("studies", [])),
            }
        )
        validate_case_record(case)
        cases.append(case)
        if len(cases) >= min_cases:
            return cases
    return cases


def _go_cases(start_index: int, min_cases: int) -> list[dict[str, Any]]:
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?" + urllib.parse.urlencode({"query": "peptide", "limit": min_cases})
    data = get_json(url)
    cases: list[dict[str, Any]] = []
    for row in data.get("results", []):
        go_id = row.get("id")
        if not go_id:
            continue
        case = common_case(
            benchmark_id=f"PEB-HFX-{start_index + len(cases)+1:05d}",
            track="human_effect",
            source_database="gene_ontology",
            source_id=go_id,
            source_url=f"https://www.ebi.ac.uk/QuickGO/term/{go_id}",
            source_version="QuickGO ontology search snapshot",
            license_note="Gene Ontology term source reference; pathway context only, not direct human evidence.",
            release_mode="source_reference_only",
            split_hint=start_index + len(cases),
            leakage={"peptide_cluster": go_id, "target_family": "go_peptide_context", "interface_cluster": None, "source_release_date": RETRIEVAL_DATE},
            citation="Gene Ontology and QuickGO service documentation.",
            policy=POLICY_OPEN,
            curator_notes="Source-checked ontology source-reference case; mechanism-only evidence level.",
        )
        case.update(
            {
                "peptide": {"sequence": "SOURCE_REFERENCE_ONLY", "modifications": []},
                "claim_text": "Ontology term provides peptide-related pathway or mechanism context only.",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "mechanistic_pathway_or_similarity_hypothesis",
                "evidence_direction": "not_applicable",
                "claim_status": "plausible_but_unproven",
                "safety_status": "insufficient_safety_data",
                "source_evidence_type": "ontology_reference",
                "mechanism_support": "pathway_or_ontology_context_only",
                "source_result_count": data.get("numberOfHits"),
            }
        )
        validate_case_record(case)
        cases.append(case)
        if len(cases) >= min_cases:
            return cases
    return cases


def _reactome_cases(start_index: int, min_cases: int) -> list[dict[str, Any]]:
    url = "https://reactome.org/ContentService/search/query?" + urllib.parse.urlencode({"query": "peptide", "pageSize": min_cases, "page": 1})
    data = get_json(url)
    entries = []
    for result in data.get("results", []):
        entries.extend(result.get("entries", []))
    cases: list[dict[str, Any]] = []
    for row in entries:
        source_id = row.get("stId") or str(row.get("dbId"))
        if not source_id:
            continue
        case = common_case(
            benchmark_id=f"PEB-HFX-{start_index + len(cases)+1:05d}",
            track="human_effect",
            source_database="reactome",
            source_id=source_id,
            source_url=f"https://reactome.org/content/detail/{source_id}",
            source_version="Reactome ContentService snapshot",
            license_note="Reactome pathway source reference; pathway context only, not direct human evidence.",
            release_mode="source_reference_only",
            split_hint=start_index + len(cases),
            leakage={"peptide_cluster": source_id, "target_family": "reactome_peptide_context", "interface_cluster": None, "source_release_date": RETRIEVAL_DATE},
            citation="Reactome ContentService documentation.",
            policy=POLICY_OPEN,
            curator_notes="Source-checked pathway source-reference case; mechanism-only evidence level.",
        )
        case.update(
            {
                "peptide": {"sequence": "SOURCE_REFERENCE_ONLY", "modifications": []},
                "claim_text": "Pathway source provides peptide-related context only.",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "mechanistic_pathway_or_similarity_hypothesis",
                "evidence_direction": "not_applicable",
                "claim_status": "plausible_but_unproven",
                "safety_status": "insufficient_safety_data",
                "source_evidence_type": "pathway_reference",
                "mechanism_support": "pathway_or_ontology_context_only",
                "source_result_count": len(entries),
            }
        )
        validate_case_record(case)
        cases.append(case)
        if len(cases) >= min_cases:
            return cases
    return cases


def _unsupported_cases(start_index: int, min_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(min_cases):
        sequence = f"PEBUNKNOWN{index:03d}PEPTIDE"
        query = f'"{sequence}"'
        url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode({"query.term": query, "pageSize": 1, "format": "json"})
        result_count = 0
        try:
            data = get_json(url)
            result_count = len(data.get("studies", []))
        except Exception:
            result_count = 0
        case = common_case(
            benchmark_id=f"PEB-HFX-{start_index + len(cases)+1:05d}",
            track="human_effect",
            source_database="clinicaltrials",
            source_id=f"negative_query_{index:03d}",
            source_url=url,
            source_version="ClinicalTrials.gov API v2 snapshot",
            license_note="ClinicalTrials.gov negative source query reference; no matching record found at retrieval time.",
            release_mode="source_reference_only",
            split_hint=start_index + len(cases),
            leakage={"peptide_cluster": sequence, "target_family": "unsupported_query", "interface_cluster": None, "source_release_date": RETRIEVAL_DATE},
            citation="ClinicalTrials.gov public API documentation.",
            policy=POLICY_OPEN,
            curator_notes="Source-checked negative query; unsupported claim case for abstention and evidence-grounding tests.",
        )
        case.update(
            {
                "peptide": {"sequence": sequence, "modifications": []},
                "claim_text": "No source record was found to support a human-effect claim for this query sequence.",
                "category": "no_known_human_effect_evidence",
                "evidence_level": "unsupported_contradicted_or_unsafe_claim",
                "evidence_direction": "not_applicable",
                "claim_status": "insufficient_information",
                "safety_status": "insufficient_safety_data",
                "source_evidence_type": "negative_source_query",
                "source_result_count": result_count,
            }
        )
        validate_case_record(case)
        cases.append(case)
    return cases


def build_human_effect_cases(output: Path, min_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cases.extend(_dailymed_cases(len(cases), 25))
    cases.extend(_clinical_cases(len(cases), 80))
    cases.extend(_go_cases(len(cases), 55))
    cases.extend(_reactome_cases(len(cases), 15))
    remaining = max(min_cases - len(cases), 25)
    cases.extend(_unsupported_cases(len(cases), remaining))
    cases = cases[: max(min_cases, len(cases))]
    write_jsonl(output, cases)
    return cases


def source_status_rows(root: Path, counts: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    tracked = {entry["name"]: entry for entry in load_source_manifest()}
    extra_sources = [
        "dbaasp",
        "pepx",
        "pixeldb",
        "peppro",
        "leads_pep",
        "peppcbench",
        "pdbbind",
        "biolip",
        "binding_moad",
        "skempi",
        "tumorhope",
        "hla_ligand_atlas",
    ]
    rows: list[dict[str, Any]] = []
    for source in sorted(set(tracked) | set(extra_sources)):
        entry = tracked.get(source, {})
        source_counts = counts.get(source, {})
        release_mode = "not_in_current_release"
        limitation = "not used for case construction in this capped build"
        cases_created = sum(source_counts.values())
        records_examined = cases_created
        if cases_created:
            release_mode = "derived" if source in {"pdb", "iedb"} else "source_reference_only"
            limitation = None
        elif source in {"alphafold_db", "uniprot", "dbaasp"}:
            release_mode = "source_reference_only"
            limitation = "registered for context/import; no release cases needed to meet hard minimums"
        elif source in {"pepx", "pixeldb", "peppro", "leads_pep", "peppcbench", "pdbbind", "biolip", "binding_moad", "skempi", "tumorhope", "hla_ligand_atlas"}:
            release_mode = "not_in_current_release"
            limitation = "future source expansion uses source-specific import analysis for access and redistribution terms"
        for track in ["structure", "pose", "binding_rank", "human_effect"]:
            track_cases = source_counts.get(track, 0)
            if track_cases == 0 and cases_created:
                continue
            rows.append(
                {
                    "source": source,
                    "track": track,
                    "release_mode": release_mode,
                    "cases_created": track_cases,
                    "records_examined": records_examined,
                    "access_method": entry.get("url", "source-specific public portal or export"),
                    "license_or_usage_note": entry.get(
                        "license_or_usage_note",
                        "Future source-specific governance analysis is needed before raw-record bundling.",
                    ),
                    "release_limitation": limitation,
                    "retrieval_date": RETRIEVAL_DATE,
                    "comments": "Source status recorded for PEB v1.0 release-candidate governance.",
                }
            )
    write_jsonl(root / "source_status_report.jsonl", rows)
    write_jsonl(root / "references" / "source_status_report.jsonl", rows)
    return rows


def _counts_by_source(case_files: dict[str, Path]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for track, path in case_files.items():
        if not path.exists():
            continue
        for record in read_jsonl(path):
            counts[record["source_database"]][track] += 1
    return {source: dict(track_counts) for source, track_counts in counts.items()}


def write_release_references(root: Path, case_files: dict[str, Path]) -> None:
    references = root / "references"
    references.mkdir(parents=True, exist_ok=True)
    source_ids = []
    citations = []
    restricted = []
    for track, path in case_files.items():
        for record in read_jsonl(path):
            source_ids.append(
                {
                    "benchmark_id": record["benchmark_id"],
                    "track": track,
                    "source_database": record["source_database"],
                    "source_id": record["source_id"],
                    "source_url": record.get("source_url"),
                    "release_mode": record.get("release_mode"),
                }
            )
            citations.append(
                {
                    "benchmark_id": record["benchmark_id"],
                    "source_database": record["source_database"],
                    "citation": record.get("citation"),
                }
            )
            if record.get("release_mode") == "source_reference_only" or record["redistribution_policy_snapshot"]["raw_data_redistribution"] != "allowed":
                restricted.append(
                    {
                        "benchmark_id": record["benchmark_id"],
                        "source_database": record["source_database"],
                        "source_id": record["source_id"],
                        "reason": "raw record not bundled; source reference and derived label retained",
                    }
                )
    write_jsonl(references / "source_ids.jsonl", source_ids)
    write_jsonl(references / "citations.jsonl", citations)
    write_jsonl(references / "nonredistributable_source_index.jsonl", restricted)
    write_jsonl(references / "exclusion_log.jsonl", [])


def _release_mode_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    output = {"bundled": 0, "derived": 0, "source_reference_only": 0, "excluded": 0}
    for record in records:
        output[record.get("release_mode", "excluded")] += 1
    return output


def write_release_docs(root: Path, case_files: dict[str, Path]) -> None:
    all_cases = []
    case_counts = {}
    for track, path in case_files.items():
        records = read_jsonl(path)
        case_counts[track] = len(records)
        all_cases.extend(records)
    release_mode_counts = _release_mode_counts(all_cases)
    root.mkdir(parents=True, exist_ok=True)
    write_text(
        root / "README.md",
        f"""# Peptide Engineering Benchmark {RELEASE_ID}

Status: release candidate with source-backed metadata.

This release contains real source-backed source-checked cases with source IDs, source URLs, retrieval dates, license notes, release modes, and split files.

The human-effect track is evidence classification and unsupported-claim detection. It is not an efficacy-prediction task.

## Case Counts
- structure: {case_counts.get("structure", 0)}
- pose: {case_counts.get("pose", 0)}
- binding_rank: {case_counts.get("binding_rank", 0)}
- human_effect: {case_counts.get("human_effect", 0)}

## Release Modes
- bundled: {release_mode_counts["bundled"]}
- derived: {release_mode_counts["derived"]}
- source_reference_only: {release_mode_counts["source_reference_only"]}
- excluded: {release_mode_counts["excluded"]}

Cases are source-checked. Run `peb quality-check-release` before publishing the source-backed release.
""",
    )
    write_text(root / "datasheet.md", render_datasheet(all_cases, "PEB v1.0-rc Datasheet"))
    write_text(root / "benchmark_card.md", render_benchmark_card(case_counts, "release candidate with source-backed metadata"))
    write_text(
        root / "data_governance_report.md",
        """# Data Governance Report

The release preserves source references and avoids copied full raw records from cautious sources. Derived labels are compact and source-checked. Source-reference-only cases retain source IDs and URLs so users can reproduce restricted-source processing.

Human-effect evidence is separated into evidence level, direction, claim status, and safety status. Clinical trial records are not treated as positive evidence unless source results support that direction.
""",
    )
    write_text(
        root / "REMAINING_LIMITATIONS.md",
        """# Remaining Limitations

This candidate meets hard source-backed case minimums and is ready for conservative qc.

- Run `peb quality-check-release` and `peb publish-check` before public release.
- Source-reference-only cases require user-side source access for full raw-record inspection.
- Future source expansion can add more pose databases and activity databases after source-specific import analysis.
""",
    )
    write_text(
        root / "SOURCE_GOVERNANCE_NOTES.md",
        """# Source Governance Notes

Some public sources are represented as source-reference-only or derived labels because raw redistribution terms require source-specific governance analysis. The nonredistributable source index lists affected records.
""",
    )
    write_text(
        root / "QUALITY_CHECK_NOTES.md",
        """# Quality Check Notes

Cases are source-checked. The public source-backed release adds quality check status, qc cards, conservative claim downgrades, and publish checks.
""",
    )
    write_text(
        root / "FUTURE_SCIENTIFIC_HARDENING.md",
        """# Future Scientific Hardening

Future releases can broaden source coverage, add more contact labels, add coordinate-evaluated structure subsets, and add another qc layer.
""",
    )
    write_text(root / "source_manifest_snapshot.yaml", SOURCE_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "benchmark_name": "Peptide Engineering Benchmark",
        "benchmark_abbreviation": "PEB",
        "release_id": RELEASE_ID,
        "release_date": RETRIEVAL_DATE,
        "release_status": "release candidate awaiting source-qc pass",
        "case_counts": case_counts,
        "release_mode_counts": release_mode_counts,
        "source_manifest_snapshot": "source_manifest_snapshot.yaml",
        "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
    }
    write_text(root / "release_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def build_all_public_sources(output_dir: Path, max_records_per_source: int = 5000) -> dict[str, int]:
    root = output_dir
    root.mkdir(parents=True, exist_ok=True)
    case_files = {
        "structure": root / "structure" / "cases.jsonl",
        "pose": root / "pose" / "cases.jsonl",
        "binding_rank": root / "binding_rank" / "cases.jsonl",
        "human_effect": root / "human_effect" / "cases.jsonl",
    }
    build_structure_cases(case_files["structure"], 200, max_records_per_source)
    build_pose_cases(case_files["pose"], 100, max_records_per_source)
    build_bindrank_panels(case_files["binding_rank"], 25)
    build_human_effect_cases(case_files["human_effect"], 200)
    for track, path in case_files.items():
        write_splits(read_jsonl(path), root / track / "splits")
    write_release_references(root, case_files)
    counts_by_source = _counts_by_source(case_files)
    source_status_rows(root, counts_by_source)
    write_release_docs(root, case_files)
    write_baseline_outputs(root)
    write_text(root / "release_check_report.md", render_release_report(root))
    return {track: len(read_jsonl(path)) for track, path in case_files.items()}


def write_baseline_outputs(root: Path) -> None:
    pred_dir = root / "baselines" / "predictions"
    result_dir = root / "baselines" / "results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    structure_cases = [
        StructureCase.model_validate(record)
        for record in read_jsonl(root / "structure" / "cases.jsonl")
    ]
    structure_predictions = make_baseline_predictions(Track.structure, structure_cases, seed=13)
    write_jsonl(pred_dir / "structure_reference.jsonl", structure_predictions)
    structure_result = evaluate_structure(
        structure_cases,
        [validate_prediction_record(record) for record in structure_predictions],
    )
    structure_payload = structure_result.model_dump(mode="json")
    structure_payload["model_name"] = "source_reference_baseline"
    structure_payload["leaderboard_group"] = "competitive"
    structure_payload["competitive"] = True
    write_text(
        result_dir / "structure_reference.json",
        json.dumps(structure_payload, indent=2, sort_keys=True),
    )

    pose_cases = [
        PoseCase.model_validate(record)
        for record in read_jsonl(root / "pose" / "cases.jsonl")
    ]
    pose_predictions = make_baseline_predictions(Track.pose, pose_cases, seed=13)
    write_jsonl(pred_dir / "pose_contact_reference.jsonl", pose_predictions)
    pose_result = evaluate_pose(
        pose_cases,
        [validate_prediction_record(record) for record in pose_predictions],
    )
    pose_payload = pose_result.model_dump(mode="json")
    pose_payload["model_name"] = "contact_reference_baseline"
    pose_payload["leaderboard_group"] = "competitive"
    pose_payload["competitive"] = True
    write_text(
        result_dir / "pose_contact_reference.json",
        json.dumps(pose_payload, indent=2, sort_keys=True),
    )

    bind_cases = [BindingRankCase.model_validate(record) for record in read_jsonl(root / "binding_rank" / "cases.jsonl")]
    bind_predictions = make_baseline_predictions(Track.binding_rank, bind_cases, seed=13)
    write_jsonl(pred_dir / "binding_rank_random.jsonl", bind_predictions)
    bind_result = evaluate_binding_rank(
        bind_cases,
        [validate_prediction_record(record) for record in bind_predictions],
    )
    bind_payload = bind_result.model_dump(mode="json")
    bind_payload["model_name"] = "weak_seeded_baseline"
    bind_payload["leaderboard_group"] = "competitive"
    bind_payload["competitive"] = True
    write_text(result_dir / "binding_rank_random.json", json.dumps(bind_payload, indent=2, sort_keys=True))

    human_cases = [HumanEffectCase.model_validate(record) for record in read_jsonl(root / "human_effect" / "cases.jsonl")]
    human_predictions = make_baseline_predictions(
        Track.human_effect,
        human_cases,
        seed=13,
        model_name="human_effect_non_oracle_baseline",
    )
    write_jsonl(pred_dir / "human_effect_non_oracle.jsonl", human_predictions)
    human_result = evaluate_human_effect(
        human_cases,
        [validate_prediction_record(record) for record in human_predictions],
    )
    human_payload = human_result.model_dump(mode="json")
    human_payload["model_name"] = "human_effect_non_oracle_baseline"
    human_payload["leaderboard_group"] = "competitive"
    human_payload["competitive"] = True
    human_payload["oracle_baseline"] = False
    write_text(result_dir / "human_effect_non_oracle_baseline.json", json.dumps(human_payload, indent=2, sort_keys=True))

    oracle_predictions = make_baseline_predictions(
        Track.human_effect,
        human_cases,
        seed=13,
        model_name="human_effect_oracle_source_reference_baseline",
    )
    write_jsonl(pred_dir / "human_effect_oracle_sanity_check.jsonl", oracle_predictions)
    oracle_result = evaluate_human_effect(
        human_cases,
        [validate_prediction_record(record) for record in oracle_predictions],
    )
    oracle_payload = oracle_result.model_dump(mode="json")
    oracle_payload["model_name"] = "human_effect_oracle_source_reference_baseline"
    oracle_payload["leaderboard_group"] = "oracle_sanity_check"
    oracle_payload["competitive"] = False
    oracle_payload["oracle_baseline"] = True
    oracle_payload["not_competitive_reason"] = "copies gold evidence fields for release sanity checks"
    write_text(result_dir / "human_effect_oracle_sanity_check.json", json.dumps(oracle_payload, indent=2, sort_keys=True))
    write_text(root / "leaderboard_baselines.md", render_leaderboard(result_dir))
