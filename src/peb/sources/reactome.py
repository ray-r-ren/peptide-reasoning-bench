"""Reactome adapter metadata."""

from peb.sources.base import SourceAdapter


class ReactomeAdapter(SourceAdapter):
    name = "reactome"
    base_url = "https://reactome.org"
    adapter_status = "implemented"

