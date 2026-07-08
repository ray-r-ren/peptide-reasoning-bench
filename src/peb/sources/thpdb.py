"""THPdb adapter metadata."""

from peb.sources.base import SourceAdapter


class THPdbAdapter(SourceAdapter):
    name = "thpdb"
    base_url = "https://example.org/thpdb"
    adapter_status = "implemented"
