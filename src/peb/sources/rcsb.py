"""RCSB PDB adapter."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Union

from peb.sources.base import SourceAdapter


class RCSBAdapter(SourceAdapter):
    name = "rcsb"
    base_url = "https://data.rcsb.org"
    adapter_status = "implemented"

    def entry_metadata(self, pdb_id: str) -> dict[str, Any]:
        return self.fetch_json(f"{self.base_url}/rest/v1/core/entry/{pdb_id.upper()}")

    def polymer_entities(self, pdb_id: str) -> list[dict[str, Any]]:
        entry = self.entry_metadata(pdb_id)
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", []
        )
        return [
            self.fetch_json(
                f"{self.base_url}/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
            )
            for entity_id in entity_ids
        ]

    def fetch_mmcif(self, pdb_id: str, output_dir: Union[str, Path]) -> Path:
        output = Path(output_dir) / f"{pdb_id.lower()}.cif"
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        return self.write_text(output, self.fetch_text(url))

    def search_peptide_complexes(self, limit: int) -> dict[str, Any]:
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match",
                    "value": "Protein",
                },
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }
        request = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(query).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "peb-release-tooling/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
