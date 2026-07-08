"""PEPBI adapter metadata."""

from peb.sources.base import SourceAdapter


class PEPBIAdapter(SourceAdapter):
    name = "pepbi"
    base_url = "https://example.org/pepbi"
    adapter_status = "implemented"

