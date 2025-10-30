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


# Not covered by test suite as this function is not yet used by production code
def chemsh_punch_to_structure_data(data: str) -> StructureData:  # pragma: no cover
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


def generate_parameter_string(params: dict) -> str:
    """
    Generate a input string for the ChemShell script from a dict.

    Take a dictionary of parameters and generate a comma separated string
    suitable for inclusion in a function call in the ChemShell input script.
    e.g. 'key1=value1, key2=value2'

    Parameters
    ----------
    params : dict
        Dictionary of parameters to convert.

    Returns
    -------
    s : str
        Comma separated string of parameters.
    """
    s = ""
    for key in params:
        if key == "theory":
            continue
        if isinstance(params[key], str):
            s += f"{key}='{params[key]}', "
        else:
            s += f"{key}={params[key]}, "
    return s.rstrip(", ")
