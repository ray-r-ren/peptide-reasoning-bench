"""IUPHAR/BPS Guide to Pharmacology adapter metadata."""

from peb.sources.base import SourceAdapter


class IUPHARAdapter(SourceAdapter):
    name = "iuphar"
    base_url = "https://www.guidetopharmacology.org"
    adapter_status = "implemented"

