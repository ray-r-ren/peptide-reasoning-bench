"""ClinicalTrials.gov adapter."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Union

from peb.sources.base import SourceAdapter


class ClinicalTrialsAdapter(SourceAdapter):
    name = "clinicaltrials"
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    adapter_status = "implemented"

    def search(self, query: str, limit: int = 10) -> dict:
        params = urllib.parse.urlencode({"query.term": query, "pageSize": limit, "format": "json"})
        return self.fetch_json(f"{self.base_url}?{params}")

    def fetch_search(self, query: str, limit: int, output: Union[str, Path]) -> Path:
        import json

        payload = self.search(query, limit)
        return self.write_text(output, json.dumps(payload, indent=2, sort_keys=True))
