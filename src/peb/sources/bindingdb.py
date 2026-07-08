"""BindingDB adapter metadata."""

from peb.sources.base import SourceAdapter


class BindingDBAdapter(SourceAdapter):
    name = "bindingdb"
    base_url = "https://www.bindingdb.org"
    adapter_status = "implemented"

