"""Workflows for geometry optimisation based taks."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import ArrayData, Bool, Dict, Float, SinglefileData, Str

from aiida_chemshell.calculations.base import ChemShellCalculation


class GeometryOptimisationWorkChain(WorkChain):
    """Geometry optimisation calculation with extended optional calculation options."""

    @classmethod
    def define(cls, spec) -> None:
        """Define the AiiDA process specification for the WorkChain."""
        super().define(spec)

        ## Inputs ##
        spec.expose_inputs(ChemShellCalculation, exclude=("metadata"))

        spec.input(
            "vibrational_analysis",
            valid_type=Bool,
            required=False,
            help="Calculate vibrational modes of resulting structure. (default=True)",
        )
        spec.input(
            "basis_quality",
            valid_type=Str,
            validator=cls.validate_basis_quality_input,
            required=False,
            help="Set basis set quality for QM calculation based on defined options.",
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
            valid_type=SinglefileData,
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
        spec.outline(cls.optimise, cls.energy, cls.result)

        return

    def optimise(self):
        """Perform the geometry optimisation."""
        inputs = self.exposed_inputs(ChemShellCalculation)
        if "qm_parameters" not in self.inputs:
            inputs["qm_parameters"] = Dict(
                {
                    "theory": "NWChem",
                    "method": "dft",
                    "functional": "B3LYP",
                    "d3": True,
                }
            )
            if "basis_quality" in self.inputs:
                inputs["qm_parameters"]["basis"] = self.get_basis_set_label(
                    self.inputs.basis_quality.value
                )
        elif "basis_quality" in self.inputs:
            inputs["qm_parameters"] = inputs.get("qm_parameters").get_dict()
            inputs["qm_parameters"]["basis"] = self.get_basis_set_label(
                self.inputs.basis_quality.value
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

        future = self.submit(ChemShellCalculation, **inputs)
        return ToContext(optimise=future)

    def energy(self):
        """Perform a single point energy calculation on the optimised structure."""
        if self.inputs.get("vibrational_analysis", False):
            inputs = {
                "code": self.exposed_inputs(ChemShellCalculation)["code"],
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
            return ToContext(energy=future)
        return None

    def result(self):
        """Extract the final workflow results."""
        self.out("optimised_structure", self.ctx.optimise.outputs.optimised_structure)
        if self.inputs.get("vibrational_analysis", False):
            self.out("final_energy", self.ctx.energy.outputs.energy)
            self.out(
                "vibrational_energies", self.ctx.energy.outputs.vibrational_energies
            )
            self.out("vibrational_modes", self.ctx.energy.outputs.vibrational_modes)
        else:
            self.out("final_energy", self.ctx.optimise.outputs.energy)
        return

    @classmethod
    def validate_basis_quality_input(cls, value: Str, _) -> str | None:
        """
        Validate the basis set quality key input.

        Parameters
        ----------
        value : Str | None
            The input key for the basis set quality mapping.

        Returns
        -------
        str | None
            Returns `None` if input is valid otherwise returns an error message.
        """
        if value.value.lower() not in ["fast", "balanced", "quality"]:
            return (
                "Invalid basis set quality key, valid keys are: 'fast', 'balanced' or "
                "'quality'."
            )
        return None

    @classmethod
    def get_basis_set_label(cls, key: str) -> str:
        """Define a custom dictionary of basis set labels for user convenience."""
        match key.lower():
            case "fast":
                return "3-21G"
            case "balanced":
                return "cc-pvdz"
            case "quality":
                return "aug-cc-pvtz"
        return ""
