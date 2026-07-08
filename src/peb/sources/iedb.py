"""IEDB import placeholder."""

from peb.sources.base import SourceAdapter


class IEDBAdapter(SourceAdapter):
    name = "iedb"
    base_url = "https://www.iedb.org"
    adapter_status = "implemented"

