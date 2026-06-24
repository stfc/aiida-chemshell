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
