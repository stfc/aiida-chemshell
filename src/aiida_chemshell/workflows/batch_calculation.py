"""Workflow for processing a series of structures from a single input."""

from aiida.engine import ProcessSpec, ToContext, WorkChain
from aiida.orm import ProcessNode, SinglefileData, TrajectoryData
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
            "structure",
            valid_type=(SinglefileData, TrajectoryData),
            required=True,
            help="The series of structures to process.",
        )

        spec.outline(cls.submit_jobs, cls.collate_results)

    def submit_jobs(self):
        """Extract all individual structures and submit their calculations."""
        futures: dict[str, ProcessNode] = {}
        if isinstance(self.inputs.structure, TrajectoryData):
            for i in range(self.inputs.structure.numsteps):
                inputs = {
                    "code": self.inputs.code,
                    "structure": self.inputs.structure,
                    "structure_index": i,
                }
                if "qm_parameters" in self.inputs:
                    inputs["qm_parameters"] = self.inputs.qm_parameters
                if "mm_parameters" in self.inputs:
                    inputs["mm_parameters"] = self.inputs.mm_parameters
                    inputs["force_field_file"] = self.inputs.force_field_file
                if "qmmm_parameters" in self.inputs:
                    inputs["qmmm_parameters"] = self.inputs.qmmm_parameters
                if "calculation_parameters" in self.inputs:
                    inputs["calculation_parameters"] = (
                        self.inputs.calculation_parameters
                    )
                if "optimisation_parameters" in self.inputs:
                    inputs["optimisation_parameters"] = (
                        self.inputs.optimisation_parameters
                    )
                future = self.submit(ChemShellCalculation, **inputs)
                futures[f"structure_{i}"] = future
        return ToContext(**futures)

    def collate_results(self) -> None:
        """Collect the WorkChain's results."""
        return


# def read_structures_from_file(filname, contents) -> list():
#     return []
