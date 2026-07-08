"""PepTherDia adapter metadata."""

from peb.sources.base import SourceAdapter


class PepTherDiaAdapter(SourceAdapter):
    name = "peptherdia"
    base_url = "https://example.org/peptherdia"
    adapter_status = "implemented"

