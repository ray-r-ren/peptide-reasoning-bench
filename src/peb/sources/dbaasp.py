"""DBAASP adapter metadata and import registration."""

from peb.sources.base import SourceAdapter


class DBAASPAdapter(SourceAdapter):
    name = "dbaasp"
    base_url = "https://dbaasp.org/"
    adapter_status = "implemented"

