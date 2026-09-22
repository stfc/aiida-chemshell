"""Workflows for geometry optimisation based taks."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import (
    ArrayData,
    Bool,
    Dict,
    Float,
    SinglefileData,
    StructureData,
)

from aiida_chemshell.calculations.base import ChemShellCalculation
from aiida_chemshell.workflows.isolated_atoms import IsolatedAtomicEnergiesWorkChain
from aiida_chemshell.workflows.utils import apply_default_input_node_tags


class GeometryOptimisationWorkChain(WorkChain):
    """Geometry optimisation calculation with extended optional calculation options."""

    # Default labels and descriptions applied to WorkChain specific input nodes
    DEFAULT_INPUT_TAGS = {
        "vibrational_analysis": (
            "Vibrational Analysis Flag",
            "Whether to calculate the vibrational modes of the optimised structure.",
        ),
    }

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)

        ## Inputs ##
        spec.expose_inputs(ChemShellCalculation, namespace="chemsh")

        spec.input(
            "vibrational_analysis",
            valid_type=Bool,
            required=False,
            help="Calculate vibrational modes of resulting structure. (default=True)",
        )

        ## Outputs ##
        spec.output(
            "final_energy",
            valid_type=Float,
            required=True,
            help="The final energy for the optimised structure.",
        )
        spec.output(
            "optimised_structure",
            valid_type=(StructureData, SinglefileData),
            required=True,
            help="The final optimised geometry of the given structure.",
        )
        spec.output(
            "vibrational_energies",
            valid_type=Dict,
            required=False,
            help="The calculated thermochemical properties of the optimised structure",
        )
        spec.output(
            "vibrational_modes",
            valid_type=ArrayData,
            required=False,
            help="The calculated vibrational modes for the optimised structure.",
        )

        ## Workflow ##
        spec.outline(
            cls.apply_default_input_tags,
            cls.optimise,
            cls.energy,
            cls.isolated_atom_energies,
            # cls.generate_mlip_training_inputs,
            # cls.train_mlip,
            cls.result,
        )

        return

    def apply_default_input_tags(self) -> None:
        """Apply default labels/descriptions to WorkChain specific input nodes."""
        apply_default_input_node_tags(self.inputs, self.DEFAULT_INPUT_TAGS)

    def optimise(self):
        """Perform the geometry optimisation."""
        inputs = self.exposed_inputs(ChemShellCalculation, namespace="chemsh")
        if "qm_parameters" not in inputs:
            inputs["qm_parameters"] = Dict(
                {
                    "theory": "NWChem",
                    "method": "dft",
                    "functional": "B3LYP",
                    "basis": "cc-pvdz",
                }
            )
        if "force_field_file" in inputs:
            if "mm_parameters" not in inputs:
                inputs["mm_parameters"] = Dict({"theory": "DL_POLY"})
            if "qmmm_parameters" not in inputs:
                inputs["qmmm_parameters"] = Dict({"qm_region": []})
        elif "mm_parameters" in inputs:
            return None
        if "optimisation_parameters" not in inputs:
            inputs["optimisation_parameters"] = Dict({})
        if "mlip_model" in self.inputs:
            inputs["optimisation_parameters"]["save_path"] = True

        future = self.submit(ChemShellCalculation, **inputs)
        future.label = ChemShellCalculation.default_process_label(future)
        future.description = (
            f"Geometry optimisation step from WorkChainNode pk: {self.node.pk}"
        )
        return ToContext(optimise=future)

    def energy(self):
        """Perform a single point energy calculation on the optimised structure."""
        if self.inputs.get("vibrational_analysis", False):
            inputs = {
                "code": self.exposed_inputs(ChemShellCalculation, namespace="chemsh")[
                    "code"
                ],
                "metadata": self.exposed_inputs(
                    ChemShellCalculation, namespace="chemsh"
                )["metadata"],
                "structure": self.ctx.optimise.outputs.optimised_structure,
                "qm_parameters": self.ctx.optimise.inputs.qm_parameters,
            }
            if "force_field_file" in self.ctx.optimise.inputs:
                inputs["force_field_file"] = self.ctx.optimise.inputs.force_field_file
                inputs["mm_parameters"] = self.ctx.optimise.inputs.mm_parameters
                inputs["qmmm_parameters"] = self.ctx.optimise.inputs.qmmm_parameters
            inputs["optimisation_parameters"] = (
                self.ctx.optimise.inputs.optimisation_parameters.get_dict()
            )
            inputs["optimisation_parameters"]["thermal"] = True
            future = self.submit(ChemShellCalculation, **inputs)
            future.label = ChemShellCalculation.default_process_label(future)
            future.description = (
                f"Vibrational frequency calculation step from WorkChainNode "
                f"pk: {self.node.pk}"
            )
            return ToContext(energy=future)
        return None

    def isolated_atom_energies(self):
        """Calculate isolated atomic energies for all species in the input structure."""
        if "mlip_model" in self.inputs:
            inputs = {
                "structure": self.inputs.chemsh.structure,
                "code": self.inputs.chemsh.code,
                "qm_parameters": self.ctx.optimise.inputs.qm_parameters,
            }
            future = self.submit(IsolatedAtomicEnergiesWorkChain, **inputs)
            future.label = "Isolated Atomic Energy WorkChain"
            future.description = (
                f"Isolated atom energies extracted from Node: "
                f"{self.inputs.chemsh.structure.pk} for ChemShell optimisation "
                f"WorkChain: {self.node.pk} to be used for MLIP fine-tuning."
            )
            return ToContext(isolated_atoms=future)
        return None

    def result(self):
        """Extract the final workflow results."""
        if "mlip_model" in self.inputs:
            self.out("optimised_structure", self.ctx.optimise.outputs.trajectory_path)
            self.out("fine_tuned_model", self.ctx.mlip_training.outputs.model)
            self.out(
                "fine_tuned_model_compiled",
                self.ctx.mlip_training.outputs.compiled_model,
            )
        else:
            self.out(
                "optimised_structure", self.ctx.optimise.outputs.optimised_structure
            )
        if self.inputs.get("vibrational_analysis", False):
            self.out("final_energy", self.ctx.energy.outputs.energy)
            self.out(
                "vibrational_energies", self.ctx.energy.outputs.vibrational_energies
            )
            self.out("vibrational_modes", self.ctx.energy.outputs.vibrational_modes)
        else:
            self.out("final_energy", self.ctx.optimise.outputs.energy)
        return
