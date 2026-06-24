"""A collection of smaller utility AiiDA CalcFunctions."""

from aiida.engine import calcfunction
from aiida.orm import Dict


@calcfunction
def create_dictionary(atoms, energies) -> Dict:
    """Collate a series of isolated atom energies into a dictionary output."""
    if len(atoms) != len(energies):
        raise ValueError(
            f"Mismatched lengths: Got {len(atoms)} atoms and {len(energies)} energies."
        )
    return Dict(dict(zip(atoms, energies, strict=False)))
