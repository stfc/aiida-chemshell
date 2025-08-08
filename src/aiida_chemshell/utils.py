"""Utility functions for the aiida-chemshell AiiDA plugin."""

from enum import Enum, auto

from aiida.orm import StructureData


class ChemShellQMTheory(Enum):
    """Enum fr the ChemShell theory interfaces."""

    NONE = auto()
    CASTEP = auto()
    CP2K = auto()
    DFTBP = auto()
    FHI_AIMS = auto()
    GAMESS_UK = auto()
    GAUSSIAN = auto()
    LSDALTON = auto()
    MNDO = auto()
    MOLPRO = auto()
    NWCHEM = auto()
    ORCA = auto()
    PYSCF = auto()
    TURBOMOLE = auto()


class ChemShellMMTheory(Enum):
    """Enum for the ChemShell MM theory interfaces."""

    NONE = auto()
    DL_POLY = auto()
    GULP = auto()
    NAMD = auto()


def chemsh_punch_to_structure_data(data: str) -> StructureData:
    """Create a AiiDA StructureData object from a ChemShell punch file."""
    structure = StructureData(pbc=[False, False, False])

    lines = data.split("\n")

    i = 0
    while i < len(lines):
        if "coordinates records" in lines[i]:
            natms = int(lines[i].split()[-1])
            for _a in range(natms):
                i += 1
                line = lines[i].split()
                atm = line[0]
                x = float(line[1])
                y = float(line[2])
                z = float(line[3])
                structure.append_atom(position=(x, y, z), symbols=atm)

        i += 1

    return structure
