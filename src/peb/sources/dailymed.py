"""DailyMed adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from peb.sources.base import SourceAdapter, quote


class DailyMedAdapter(SourceAdapter):
    name = "dailymed"
    base_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    adapter_status = "implemented"

    def search(self, name: str) -> dict:
        return self.fetch_json(f"{self.base_url}/spls.json?drug_name={quote(name)}")

    def fetch_label(self, set_id: str, output_dir: Union[str, Path]) -> Path:
        text = self.fetch_text(f"{self.base_url}/spls/{quote(set_id)}.json")
        return self.write_text(Path(output_dir) / f"{set_id}.json", text)
