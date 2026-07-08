"""PepBind adapter metadata."""

from peb.sources.base import SourceAdapter


class PepBindAdapter(SourceAdapter):
    name = "pepbind"
    base_url = "https://example.org/pepbind"
    adapter_status = "implemented"
