"""Workflows for isolating atomic species and calculating SP energies."""

from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm import Dict, StructureData
from aiida.plugins.factories import CalculationFactory

ChemShellCalculation = CalculationFactory("chemshell")


class IsolatedAtomicEnergiesWorkChain(WorkChain):
    """AiiDA workflow for extracting isolated atomic energies from a given structure."""

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)

        spec.expose_inputs(
            ChemShellCalculation, include=("structure", "qm_parameters", "code")
        )

        spec.output(
            "atom_energies",
            valid_type=Dict,
            required=False,
            help=(
                "The individual isolated atomic energies for every unique atom type in "
                "the given system."
            ),
        )

        spec.outline(cls.determine_unique_atom_types, cls.atom_energies, cls.result)

        return

    def determine_unique_atom_types(self) -> None:
        """Determine all unique atom types within the given structure."""
        self.unique_atoms = {}
        if isinstance(self.inputs.structure, StructureData):
            self._atom_types_from_structuredata(self.inputs.structure)
        # elif isinstance(self.inputs.structure, SinglefileData):
        else:
            self._atom_types_from_file()
        return

    def atom_energies(self):
        """Run ChemShell single point calculations for each atom type."""
        calculations = {}
        for atom_symbol in self.unique_atoms.keys():
            structure = StructureData()
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols=atom_symbol)
            structure.set_pbc((False, False, False))
            builder = self.inputs.code.get_builder()
            builder.structure = structure
            builder.qm_parameters = self.inputs.qm_parameters
            future = self.submit(builder)
            calculations[atom_symbol] = future
        return ToContext(**calculations)

    def result(self) -> None:
        """Collect the results into a dictionary."""
        results_dict = collate_energies(
            list(self.unique_atoms.keys()),
            [self.ctx.get(atom).outputs.energy for atom in self.unique_atoms.keys()],
        )
        self.out("atom_energies", results_dict)
        return

    def _atom_types_from_structuredata(self, structure: StructureData) -> None:
        """Determine the unique atom types from a StructureData object."""
        for site in structure.sites:
            self.unique_atoms[site.kind_name] = atom_symbol_to_z(site.kind_name)
        return

    def _atom_types_from_file(self) -> None:
        """Determine the unique atom types from a SinglefileData object."""
        if self.inputs.structure.filename[-4:] == ".xyz":
            self._atom_types_from_xyz()
        elif self.inputs.structure.filename[-6:] == ".cjson":
            self._atom_types_from_cjson()
        elif self.inputs.structure.filename[-4:] == ".pun":
            self._atom_types_from_pun()
        return

    def _atom_types_from_xyz(self) -> None:
        """Determine the unique atom types from an xyz structure file."""
        structure = StructureData()
        structure._parse_xyz(self.inputs.structure.content.decode("utf-8"))
        self._atom_types_from_structuredata(structure)
        return

    def _atom_types_from_cjson(self) -> None:
        """Determine the unique atom types from a cjson structure file."""
        import json

        data = json.loads(self.inputs.structure.content).get("atoms").get("elements")
        atom_symbols: list[str] = data.get("symbol", [])
        atom_numbers: list[int] = data.get("number", [])
        if len(atom_symbols) == 0:
            self.unique_atoms = {
                atom_z_to_symbol(number): number for number in atom_numbers
            }
        elif len(atom_numbers) == 0:
            self.unique_atoms = {
                symbol: atom_symbol_to_z(symbol) for symbol in atom_symbols
            }
        else:
            self.unique_atoms = dict(zip(atom_symbols, atom_numbers, strict=False))
        return

    def _atom_types_from_pun(self) -> None:
        """Determine the unique atom types from a .pun structure file."""
        return


@calcfunction
def collate_energies(atoms, energies) -> Dict:
    """Collate a series of isolated atom energies into a dictionary output."""
    if len(atoms) != len(energies):
        raise ValueError(
            f"Mismatched lengths: Got {len(atoms)} atoms and {len(energies)} energies."
        )
    return Dict(dict(zip(atoms, energies, strict=False)))


def atom_symbol_to_z(at_symb: str) -> int:
    """
    Get the corresponding nuclear charge for a given atom.

    Parameters
    ----------
    at_symb     : str
        The atom label.

    Returns
    -------
    Z           : int
        The nuclear charge for the given atom.
    """
    atom_dict = {
        "H": 1,
        "HE": 2,
        "LI": 3,
        "BE": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "NE": 10,
        "NA": 11,
        "MG": 12,
        "AL": 13,
        "SI": 14,
        "P": 15,
        "S": 16,
        "CL": 17,
        "AR": 18,
        "K": 19,
        "CA": 20,
        "SC": 21,
        "TI": 22,
        "V": 23,
        "CR": 24,
        "MN": 25,
        "FE": 26,
        "CO": 27,
        "NI": 28,
        "CU": 29,
        "ZN": 30,
        "GA": 31,
        "GE": 32,
        "AS": 33,
        "SE": 34,
        "BR": 35,
        "KR": 36,
        "RB": 37,
        "SR": 38,
        "Y": 39,
        "ZR": 40,
        "NB": 41,
        "MO": 42,
        "TC": 43,
        "RU": 44,
        "RH": 45,
        "PD": 46,
        "AG": 47,
        "CD": 48,
        "IN": 49,
        "SN": 50,
        "SB": 51,
        "TE": 52,
        "I": 53,
        "XE": 54,
        "CS": 55,
        "BA": 56,
        "LA": 57,
        "CE": 58,
        "PR": 59,
        "ND": 60,
        "PM": 61,
        "SM": 62,
        "EU": 63,
        "GD": 64,
        "TB": 65,
        "DY": 66,
        "HO": 67,
        "ER": 68,
        "TM": 69,
        "YB": 70,
        "LU": 71,
        "HF": 72,
        "TA": 73,
        "W": 74,
        "RE": 75,
        "OS": 76,
        "IR": 77,
        "PT": 78,
        "AU": 79,
        "HG": 80,
        "TL": 81,
        "PB": 82,
        "BI": 83,
        "PO": 84,
        "AT": 85,
        "RN": 86,
        "FR": 87,
        "RA": 88,
        "AC": 89,
        "TH": 90,
        "PA": 91,
        "U": 92,
        "NP": 93,
        "PU": 94,
        "AM": 95,
        "CM": 96,
        "BK": 97,
        "CF": 98,
        "ES": 99,
        "FM": 100,
        "MD": 101,
        "NO": 102,
        "LR": 103,
        "RF": 104,
        "DB": 105,
        "DG": 106,
        "BH": 107,
        "HS": 108,
        "MT": 109,
        "DS": 110,
        "RG": 111,
        "UUB": 112,
        "UUT": 113,
        "UUQ": 114,
        "UUP": 115,
        "UUH": 116,
        "UUS": 117,
        "UUO": 118,
        "GH": 0,
        "X": -1,
    }  # X denotes dummy atom in zmatrix
    try:
        z = atom_dict[at_symb.upper()]
    except KeyError:
        raise KeyError(f"Invalid atomic symbol {at_symb}.") from None
    return z


def atom_z_to_symbol(z: int) -> str:
    """
    Get the corresponding atomic symbol for a given nuclear charge.

    Parameters
    ----------
    Z           : int
        The nuclear charge for the given atom.

    Returns
    -------
    at_symb     : str
        The atom label.
    """
    atom_dict = {
        1: "H",
        2: "HE",
        3: "LI",
        4: "BE",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "NE",
        11: "NA",
        12: "MG",
        13: "AL",
        14: "SI",
        15: "P",
        16: "S",
        17: "CL",
        18: "AR",
        19: "K",
        20: "CA",
        21: "SC",
        22: "TI",
        23: "V",
        24: "CR",
        25: "MN",
        26: "FE",
        27: "CO",
        28: "NI",
        29: "CU",
        30: "ZN",
        31: "GA",
        32: "GE",
        33: "AS",
        34: "SE",
        35: "BR",
        36: "KR",
        37: "RB",
        38: "SR",
        39: "Y",
        40: "ZR",
        41: "NB",
        42: "MO",
        43: "TC",
        44: "RU",
        45: "RH",
        46: "PD",
        47: "AG",
        48: "CD",
        49: "IN",
        50: "SN",
        51: "SB",
        52: "TE",
        53: "I",
        54: "XE",
        55: "CS",
        56: "BA",
        57: "LA",
        58: "CE",
        59: "PR",
        60: "ND",
        61: "PM",
        62: "SM",
        63: "EU",
        64: "GD",
        65: "TB",
        66: "DY",
        67: "HO",
        68: "ER",
        69: "TM",
        70: "YB",
        71: "LU",
        72: "HF",
        73: "TA",
        74: "W",
        75: "RE",
        76: "OS",
        77: "IR",
        78: "PT",
        79: "AU",
        80: "HG",
        81: "TL",
        82: "PB",
        83: "BI",
        84: "PO",
        85: "AT",
        86: "RN",
        87: "FR",
        88: "RA",
        89: "AC",
        90: "TH",
        91: "PA",
        92: "U",
        93: "NP",
        94: "PU",
        95: "AM",
        96: "CM",
        97: "BK",
        98: "CF",
        99: "ES",
        100: "FM",
        101: "MD",
        102: "NO",
        103: "LR",
        104: "RF",
        105: "DB",
        106: "DG",
        107: "BH",
        108: "HS",
        109: "MT",
        110: "DS",
        111: "RG",
        112: "UUB",
        113: "UUT",
        114: "UUQ",
        115: "UUP",
        116: "UUH",
        117: "UUS",
        118: "UUO",
        0: "GH",
        -1: "X",
    }  # X denotes dummy atom in zmatrix
    try:
        atm_symb = atom_dict[z]
    except KeyError:
        raise KeyError(f"Invalid atomic number {z}.") from None
    return atm_symb
