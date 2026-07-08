"""Propedia adapter metadata."""

from peb.sources.base import SourceAdapter


class PropediaAdapter(SourceAdapter):
    name = "propedia"
    base_url = "https://bioinfo.dcc.ufmg.br/propedia"
    adapter_status = "implemented"

