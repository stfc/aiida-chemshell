"""Workflows for isolating atomic species and calculating SP energies."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, StructureData
from aiida.plugins.factories import CalculationFactory

from aiida_chemshell.calculations.utils import create_dictionary
from aiida_chemshell.periodic_table import PeriodicTable

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
        self.unique_atoms = []
        if isinstance(self.inputs.structure, StructureData):
            self._atom_types_from_structuredata(self.inputs.structure)
        else:
            self._atom_types_from_file()
        return

    def atom_energies(self):
        """Run ChemShell single point calculations for each atom type."""
        calculations = {}
        for atom_symbol in self.unique_atoms:
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
        results_dict = create_dictionary(
            list(self.unique_atoms),
            [self.ctx.get(atom).outputs.energy for atom in self.unique_atoms],
        )
        self.out("atom_energies", results_dict)
        return

    def _atom_types_from_structuredata(self, structure: StructureData) -> None:
        """Determine the unique atom types from a StructureData object."""
        self.unique_atoms = [site.kind_name for site in structure.sites]

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
            self.unique_atoms = [
                PeriodicTable.atom_z_to_symbol(number) for number in atom_numbers
            ]
        else:
            self.unique_atoms = atom_symbols
        return

    def _atom_types_from_pun(self) -> None:
        """Determine the unique atom types from a .pun structure file."""
        return
