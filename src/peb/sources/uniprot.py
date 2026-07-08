"""UniProt adapter metadata."""

from peb.sources.base import SourceAdapter


class UniProtAdapter(SourceAdapter):
    name = "uniprot"
    base_url = "https://www.uniprot.org"
    adapter_status = "implemented"

