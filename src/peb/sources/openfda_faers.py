"""openFDA FAERS adapter metadata."""

from peb.sources.base import SourceAdapter


class OpenFDAFAERSAdapter(SourceAdapter):
    name = "openfda_faers"
    base_url = "https://open.fda.gov/apis/drug/event"
    adapter_status = "implemented"

