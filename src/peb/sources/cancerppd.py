"""CancerPPD adapter metadata."""

from peb.sources.base import SourceAdapter


class CancerPPDAdapter(SourceAdapter):
    name = "cancerppd"
    base_url = "https://example.org/cancerppd"
    adapter_status = "implemented"
