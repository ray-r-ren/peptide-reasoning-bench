"""ChEMBL adapter metadata."""

from peb.sources.base import SourceAdapter


class ChEMBLAdapter(SourceAdapter):
    name = "chembl"
    base_url = "https://www.ebi.ac.uk/chembl"
    adapter_status = "implemented"

