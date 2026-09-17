"""Workflows for isolating atomic species and calculating SP energies."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, StructureData
from aiida.plugins.factories import CalculationFactory

from aiida_chemshell.calculations.utils import (
    create_dictionary,
    create_isolated_atom_structures,
)
from aiida_chemshell.workflows.utils import apply_default_input_node_tags

ChemShellCalculation = CalculationFactory("chemshell")


class IsolatedAtomicEnergiesWorkChain(WorkChain):
    """AiiDA workflow for extracting isolated atomic energies from a given structure."""

    # Default labels and descriptions applied to WorkChain specific input nodes
    DEFAULT_INPUT_TAGS = {
        "structure": (
            "Input Chemical Structure",
            "The input structure to extract isolated atomic energies from.",
        ),
    }

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)

        spec.expose_inputs(
            ChemShellCalculation, include=("structure", "qm_parameters", "code")
        )
        spec.expose_inputs(
            ChemShellCalculation, include=("metadata",), namespace="chemsh"
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

        spec.outline(
            cls.apply_default_input_tags,
            cls.create_atom_structures,
            cls.atom_energies,
            cls.result,
        )

        return

    def apply_default_input_tags(self) -> None:
        """Apply default labels/descriptions to WorkChain specific input nodes."""
        apply_default_input_node_tags(self.inputs, self.DEFAULT_INPUT_TAGS)

    def create_atom_structures(self) -> None:
        """Create provenance-tracked isolated atom structures for each atom type."""
        if self.inputs.qm_parameters["theory"] == "PySCF":
            raise Exception(
                "Isolated atom calculations not supported by PySCF QM backend."
            )
        self.ctx.atom_structures = create_isolated_atom_structures(
            self.inputs.structure
        )
        return

    def atom_energies(self):
        """Run ChemShell single point calculations for each atom type."""
        calculations = {}
        for atom_symbol, structure in self.ctx.atom_structures.items():
            inputs = {
                "structure": structure,
                "qm_parameters": self.inputs.qm_parameters,
                "code": self.inputs.code,
                "metadata": self.inputs.chemsh.metadata,
            }
            future = self.submit(ChemShellCalculation, **inputs)
            future.description = f"Single point energy for {atom_symbol} atom."
            calculations[atom_symbol] = future
        return ToContext(**calculations)

    def result(self) -> None:
        """Collect the results into a dictionary."""
        unique_atoms = list(self.ctx.atom_structures.keys())
        results_dict = create_dictionary(
            unique_atoms,
            [self.ctx.get(atom).outputs.energy for atom in unique_atoms],
        )
        if isinstance(self.inputs.structure, StructureData):
            results_dict.label = (
                f"Atomistic energies for all unique atoms extracted from Node: "
                f"{self.inputs.structure.pk}"
            )
        else:
            results_dict.label = (
                "Atomistic energies for all unique atoms extracted from "
                f"{self.inputs.structure.filename} (Node: {self.inputs.structure.pk})"
            )
        self.out("atom_energies", results_dict)
        return
