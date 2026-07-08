"""SIDER adapter metadata."""

from peb.sources.base import SourceAdapter


class SIDERAdapter(SourceAdapter):
    name = "sider"
    base_url = "http://sideeffects.embl.de"
    adapter_status = "implemented"

