"""NeuroPep adapter metadata."""

from peb.sources.base import SourceAdapter


class NeuroPepAdapter(SourceAdapter):
    name = "neuropep"
    base_url = "https://isyslab.info/NeuroPep"
    adapter_status = "implemented"

