"""Simple sequence feature helpers for weak baselines."""

from __future__ import annotations

HYDROPHOBIC = set("AILMFWYV")
CHARGED = set("DEKRH")


def sequence_features(sequence: str) -> dict[str, float]:
    length = max(len(sequence), 1)
    return {
        "length": float(len(sequence)),
        "hydrophobic_fraction": sum(1 for char in sequence if char in HYDROPHOBIC) / length,
        "charged_fraction": sum(1 for char in sequence if char in CHARGED) / length,
    }

