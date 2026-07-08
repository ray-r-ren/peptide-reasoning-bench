"""PepBDB adapter metadata."""

from peb.sources.base import SourceAdapter


class PepBDBAdapter(SourceAdapter):
    name = "pepbdb"
    base_url = "https://example.org/pepbdb"
    adapter_status = "implemented"
