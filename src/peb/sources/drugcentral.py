"""DrugCentral adapter metadata."""

from peb.sources.base import SourceAdapter


class DrugCentralAdapter(SourceAdapter):
    name = "drugcentral"
    base_url = "https://drugcentral.org"
    adapter_status = "implemented"

