"""Open Targets adapter metadata."""

from peb.sources.base import SourceAdapter


class OpenTargetsAdapter(SourceAdapter):
    name = "opentargets"
    base_url = "https://platform.opentargets.org"
    adapter_status = "implemented"

