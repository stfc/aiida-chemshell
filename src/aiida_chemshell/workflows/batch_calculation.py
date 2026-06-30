"""Workflow for processing a series of structures from a single input."""

from aiida.engine import ProcessSpec, ToContext, WorkChain
from aiida.orm import ProcessNode, StructureData, TrajectoryData
from aiida.plugins.factories import CalculationFactory

ChemShellCalculation = CalculationFactory("chemshell")


class BatchProcessWorkChain(WorkChain):
    """Process a series of structures with the same inputs."""

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        """Define the AiiDA process specification."""
        super().define(spec)

        # Expose Chemshell inputs
        spec.expose_inputs(ChemShellCalculation, exclude=("structure", "metadata"))

        # Input structure series
        spec.input(
            "trajectory",
            valid_type=TrajectoryData,
            required=False,
            help="The series of structures to process as a TrajectoryData node.",
        )
        spec.input_namespace(
            "structures",
            valid_type=StructureData,
            required=False,
            help="A dictionary of StructureData nodes to batch process.",
        )

        spec.exit_code(
            350,
            "ERROR_NO_INPUTS",
            message=("Must specify either 'trajectory' or 'structure' input."),
        )

        spec.outline(cls.validate_inputs, cls.submit_jobs, cls.collate_results)

    def validate_inputs(self):
        """Validate the inputs provided to the WorkChain."""
        has_trajectory = "trajectory" in self.inputs
        has_structures = "structures" in self.inputs
        if not has_trajectory and not has_structures:
            return self.exit_codes.ERROR_NO_INPUTS
        return None

    def submit_jobs(self):
        """Extract all individual structures and submit their calculations."""
        futures: dict[str, ProcessNode] = {}
        inputs = {"code": self.inputs.code}
        if "qm_parameters" in self.inputs:
            inputs["qm_parameters"] = self.inputs.qm_parameters
        if "mm_parameters" in self.inputs:
            inputs["mm_parameters"] = self.inputs.mm_parameters
            inputs["force_field_file"] = self.inputs.force_field_file
        if "qmmm_parameters" in self.inputs:
            inputs["qmmm_parameters"] = self.inputs.qmmm_parameters
        if "calculation_parameters" in self.inputs:
            inputs["calculation_parameters"] = self.inputs.calculation_parameters
        if "optimisation_parameters" in self.inputs:
            inputs["optimisation_parameters"] = self.inputs.optimisation_parameters
        if "trajectory" in self.inputs:
            for i in range(self.inputs.trajectory.numsteps):
                inputs["structure"] = self.inputs.trajectory
                inputs["structure_index"] = i
                future = self.submit(ChemShellCalculation, **inputs)
                futures[f"trajectory_frame_{i}"] = future
        if "structures" in self.inputs:
            for key, structure in self.inputs.structures.items():
                inputs["structure"] = structure
                future = self.submit(ChemShellCalculation, **inputs)
                futures[key] = future
        return ToContext(**futures)

    def collate_results(self) -> None:
        """Collect the WorkChain's results."""
        return


# def read_structures_from_file(filname, contents) -> list():
#     return []
