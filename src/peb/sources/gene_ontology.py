"""Gene Ontology adapter metadata."""

from peb.sources.base import SourceAdapter


class GeneOntologyAdapter(SourceAdapter):
    name = "gene_ontology"
    base_url = "http://geneontology.org"
    adapter_status = "implemented"

