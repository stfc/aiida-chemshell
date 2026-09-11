"""Workflow for processing a series of structures from a single input."""

from aiida.engine import ProcessSpec, ToContext, WorkChain
from aiida.orm import (
    ProcessNode,
    SinglefileData,
    StructureData,
    TrajectoryData,
)
from aiida.plugins.factories import CalculationFactory

from aiida_chemshell.calculations.utils import (
    combine_into_extended_xyz,
    extract_structures_from_xyz,
)
from aiida_chemshell.workflows.utils import apply_default_input_node_tags

ChemShellCalculation = CalculationFactory("chemshell")


class BatchProcessWorkChain(WorkChain):
    """Process a series of structures with the same inputs."""

    # Default labels and descriptions applied to WorkChain specific input nodes
    DEFAULT_INPUT_TAGS = {
        "structure_files": (
            "Multi-Structure Input File",
            (
                "A structure file containing multiple structures to batch process with "
                "ChemShell."
            ),
        ),
    }

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        """Define the AiiDA process specification."""
        super().define(spec)

        # Expose Chemshell inputs in the top level namespace
        spec.expose_inputs(ChemShellCalculation, exclude=("structure", "metadata"))

        # Expose the calculation metadata under a dedicated 'calc' namespace.
        spec.expose_inputs(
            ChemShellCalculation, namespace="calc", include=("metadata",)
        )

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
        spec.input_namespace(
            "structure_files",
            valid_type=SinglefileData,
            required=False,
            help=(
                "A dictionary of SinglefileData objects containing the series of "
                "input structures. The dictionary keys are not used, the node labels"
                "are determined by the filename."
            ),
        )

        # Results combination key
        spec.input(
            "combine_results",
            valid_type=bool,
            required=False,
            non_db=True,
            help=(
                "A key which tells the workchain to combine all the results from the "
                "individual batch processing tasks into one final results object which "
                'is an extended xyz file (key="xyz").'
            ),
            # validator=cls.validate_combination_input_key,
        )
        # Results combination output node
        spec.output(
            "combined_results",
            valid_type=SinglefileData,
            required=False,
            help=(
                "An extended XYZ file with all the ChemShell results from the batch "
                "processed input structures."
            ),
        )

        spec.exit_code(
            350,
            "ERROR_NO_INPUTS",
            message=(
                "Must specify either 'trajectory', 'structures' or "
                "'structure_files' input."
            ),
        )

        spec.outline(
            cls.apply_default_input_tags,
            cls.validate_inputs,
            cls.extract_structures_from_files,
            cls.submit_jobs,
            cls.collate_results,
        )

    def apply_default_input_tags(self) -> None:
        """Apply default labels/descriptions to WorkChain specific input nodes."""
        apply_default_input_node_tags(self.inputs, self.DEFAULT_INPUT_TAGS)

    # @classmethod
    # def validate_combination_input_key(cls, key: str | None, _) -> str | None:
    #     """Validate the combine_results input key."""
    #     if key in ["xyz", "trajectory"]:
    #         return None
    #     return (
    #         f"Invalid input for 'combine_results'. {key} must be either 'xyz' or "
    #         "'trajectory'"
    #     )

    def validate_inputs(self):
        """Validate the inputs provided to the WorkChain."""
        has_trajectory = "trajectory" in self.inputs
        has_structures = "structures" in self.inputs
        has_files = "structure_files" in self.inputs
        if not has_trajectory and not has_structures and not has_files:
            return self.exit_codes.ERROR_NO_INPUTS
        return None

    def extract_structures_from_files(self) -> None:
        """Extract the structures from file like input objects."""
        self.structures_from_files = {}
        if "structure_files" in self.inputs:
            for _key, file in self.inputs.structure_files.items():
                if file.filename[-4:] != ".xyz":
                    self.report(
                        "Only XYZ structured trajectory files are currently "
                        f"supported, {file.filename} will be skipped..."
                    )
                else:
                    self.report(f"Parsing structures from {file.filename}")
                    self.structures_from_files = (
                        self.structures_from_files | extract_structures_from_xyz(file)
                    )
        return

    def submit_jobs(self):
        """Extract all individual structures and submit their calculations."""
        futures: dict[str, ProcessNode] = {}
        inputs = {"code": self.inputs.code}
        # Forward the calculation options (resources, MPI processes, wallclock, ...)
        # to every job in the batch.
        if "calc" in self.inputs and "options" in self.inputs.calc.metadata:
            inputs["metadata"] = {"options": dict(self.inputs.calc.metadata.options)}
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
        if "structure_files" in self.inputs:
            for key, structure in self.structures_from_files.items():
                inputs["structure"] = structure
                future = self.submit(ChemShellCalculation, **inputs)
                futures[key] = future
        return ToContext(**futures)

    def collate_results(self) -> None:
        """Collect the WorkChain's results."""
        if self.inputs.get("combine_results", None):
            if "optimisation_parameters" in self.inputs:
                self.logger.warning(
                    "Output combination is not currently supported for optimisation "
                    "jobs."
                )
                return
            inputs = {}
            include_forces = self.inputs.get("calculation_parameters", {}).get(
                "gradients", False
            )
            if "trajectory" in self.inputs:
                inputs["structure_trajectory"] = self.inputs.trajectory
                for i in range(self.inputs.trajectory.numsteps):
                    inputs[f"energy_trajectory_frame_{i}"] = self.ctx[
                        f"trajectory_frame_{i}"
                    ].outputs.energy
                    if include_forces:
                        inputs[f"array_trajectory_frame_{i}"] = self.ctx[
                            f"trajectory_frame_{i}"
                        ].outputs.gradients
            if "structures" in self.inputs:
                for key, structure in self.inputs.structures.items():
                    inputs[f"structure_{key}"] = structure
                    inputs[f"energy_{key}"] = self.ctx[key].outputs.energy
                    if include_forces:
                        inputs[f"array_{key}"] = self.ctx[key].outputs.gradients
            if "structure_files" in self.inputs:
                for key, structure in self.structures_from_files.items():
                    inputs[f"structure_{key}"] = structure
                    inputs[f"energy_{key}"] = self.ctx[key].outputs.energy
                    if include_forces:
                        inputs[f"array_{key}"] = self.ctx[key].outputs.gradients
            combined_output_node = combine_into_extended_xyz(**inputs)
            combined_output_node.label = "ChemShell Batch Processed Structures"
            combined_output_node.description = (
                "Collection of structures processed by ChemShell from WorkChain: "
                f" {self.node.pk}"
            )
            self.out("combined_results", combined_output_node)
        return
