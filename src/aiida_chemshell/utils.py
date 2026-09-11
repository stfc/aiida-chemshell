"""Utility functions for the aiida-chemshell AiiDA plugin."""

import json
from enum import Enum, auto

from aiida.orm import SinglefileData, StructureData

from aiida_chemshell.periodic_table import PeriodicTable
from aiida_chemshell.units import UnitsConverter


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


def chemsh_cjson_to_structure_data(data: str | bytes) -> StructureData:
    """
    Create an AiiDA StructureData object from a ChemShell '.cjson' file.

    Parameters
    ----------
    data : str | bytes
        The contents of a ChemShell '.cjson' (Chemical JSON) structure file.

    Returns
    -------
    StructureData
        A non-periodic AiiDA StructureData node containing the parsed structure.
        Coordinates are converted to Angstrom when the file reports them in
        atomic units (Bohr).
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    cjson = json.loads(data)

    atoms = cjson.get("atoms", {})
    elements = atoms.get("elements", {})
    symbols = elements.get("symbol")
    if symbols is None:
        numbers = elements.get("number")
        if numbers is None:
            raise ValueError("CJSON file contains no atomic element information.")
        symbols = [PeriodicTable.atom_z_to_symbol(number) for number in numbers]
    # Normalise to the standard element symbol case (e.g. 'HE' -> 'He') as
    # required by AiiDA StructureData.
    symbols = [symbol.capitalize() for symbol in symbols]

    coords = atoms.get("coords", {})
    flat_coords = coords.get("3d", [])
    unit = coords.get("unit", "angstrom").lower()

    if len(flat_coords) != 3 * len(symbols):
        raise ValueError(
            "Mismatch between the number of atoms and coordinates in the CJSON file."
        )

    def _to_angstrom(value: float) -> float:
        if unit in ("au", "bohr", "a.u.", "atomic"):
            return UnitsConverter.bohr_to_angstrom(value)
        return value

    structure = StructureData(pbc=[False, False, False])
    for index, symbol in enumerate(symbols):
        position = [_to_angstrom(c) for c in flat_coords[3 * index : 3 * index + 3]]
        structure.append_atom(position=position, symbols=symbol)

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


def generate_default_mlip_fine_tune_config():
    """Generate a default configuration for mlip fine-tuning via Janus."""
    return {
        # "multiheads_finetuning": True,
        "foundation_filter_elements": True,
        "foundation_model_readout": True,
        "foundation_model_elements": False,
        "loss": "universal",
        "weight_pt_head": 10.0,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 10.0,
        "stress_key": "stress",
        "energy_key": "energy",
        "forces_key": "forces",
        "compute_stress": False,
        "compute_forces": True,
        "clip_grad": 10,
        "error_table": "PerAtomRMSE",
        "scaling": "rms_forces_scaling",
        "force_mh_ft_lr": True,
        "lr": 0.0001,
        "batch_size": 2,
        "max_num_epochs": 10,
        "ema": True,
        "ema_decay": 0.99999,
        "amsgrad": True,
        "default_dtype": "float64",
        "device": "cpu",
        "restart_latest": True,
        # "seed": 2024,
        "keep_isolated_atoms": True,
        "save_cpu": True,
        "weight_decay": 1e-8,
        "eval_interval": 1,
        # "enable_cueq": True,
    }


def xyz_file_validator(value: SinglefileData) -> str | None:
    """Check if a file is a valid XYZ file."""
    contents = value.get_content(mode="r").splitlines()
    try:
        natoms = int(contents[0].strip())
    except ValueError:
        return (
            "The first line of the XYZ file must be an integer"
            "representing the number of atoms."
        )
    else:
        if natoms != len(contents) - 2:
            return (
                f"The number of atoms specified ({natoms}) does not"
                f"match the number of atom lines ({len(contents) - 2})."
            )
    return None
