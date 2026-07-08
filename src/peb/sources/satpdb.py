"""SATPdb adapter metadata."""

from peb.sources.base import SourceAdapter


class SATPdbAdapter(SourceAdapter):
    name = "satpdb"
    base_url = "https://example.org/satpdb"
    adapter_status = "implemented"
