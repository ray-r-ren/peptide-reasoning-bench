"""Small PDB contact-map utility."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Union


def _parse_pdb_atoms(
    path: Union[str, Path],
) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    atoms: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip() or "_"
            residue = f"{chain}:{line[22:26].strip()}"
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms[chain][residue].append((x, y, z))
    return atoms


def _min_distance(left: list[tuple[float, float, float]], right: list[tuple[float, float, float]]) -> float:
    best = float("inf")
    for ax, ay, az in left:
        for bx, by, bz in right:
            best = min(best, math.dist((ax, ay, az), (bx, by, bz)))
    return best


def compute_contacts(
    structure: Union[str, Path], target_chain: str, peptide_chain: str, cutoff: float = 5.0
) -> list[dict[str, Union[str, float]]]:
    atoms = _parse_pdb_atoms(structure)
    contacts: list[dict[str, Union[str, float]]] = []
    for target_residue, target_atoms in atoms.get(target_chain, {}).items():
        for peptide_residue, peptide_atoms in atoms.get(peptide_chain, {}).items():
            distance = _min_distance(target_atoms, peptide_atoms)
            if distance <= cutoff:
                contacts.append(
                    {
                        "target_residue": target_residue,
                        "peptide_residue": peptide_residue,
                        "distance_angstrom": round(distance, 3),
                    }
                )
    return contacts
