from peb.registry import source_by_name
from peb.reports.source_card import render_source_card


def test_source_card_generated():
    card = render_source_card(source_by_name("pdb"))
    assert "pdb" in card.lower()
    assert "Redistribution" in card

