"""Base source adapter."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, Union


class SourceAdapter:
    """Minimal explicit network adapter."""

    name = "base"
    base_url = ""
    adapter_status = "implemented"

    def fetch_json(self, url: str, timeout: int = 30) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "peb-release-tooling/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_text(self, url: str, timeout: int = 30) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "peb-release-tooling/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def write_text(self, path: Union[str, Path], text: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return output

    def normalize_reference(self, source_id: str, source_url: str, **metadata: Any) -> dict[str, Any]:
        return {
            "source": self.name,
            "source_id": source_id,
            "source_url": source_url,
            "release_mode": "source_reference_only",
            "metadata": metadata,
        }

    def source_status(
        self,
        track: str,
        cases_created: int = 0,
        release_limitation: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "source": self.name,
            "track": track,
            "release_mode": "source_reference_only" if cases_created else "not_in_current_release",
            "cases_created": cases_created,
            "records_examined": cases_created,
            "access_method": self.base_url,
            "license_or_usage_note": "See source manifest for current source-specific terms.",
            "release_limitation": release_limitation,
            "comments": "Adapter supports source-reference registration.",
        }


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")
