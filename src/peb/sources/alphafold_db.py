"""AlphaFold DB adapter metadata."""

from peb.sources.base import SourceAdapter


class AlphaFoldDBAdapter(SourceAdapter):
    name = "alphafold_db"
    base_url = "https://alphafold.ebi.ac.uk"
    adapter_status = "implemented"

