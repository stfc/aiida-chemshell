"""Workflow for processing a series of structures from a single input."""

import re
from io import BytesIO

from aiida.engine import ProcessSpec, ToContext, WorkChain, calcfunction
from aiida.orm import (
    ArrayData,
    Float,
    ProcessNode,
    SinglefileData,
    StructureData,
    TrajectoryData,
)
from aiida.plugins.factories import CalculationFactory

ChemShellCalculation = CalculationFactory("chemshell")


class BatchProcessWorkChain(WorkChain):
    """Process a series of structures with the same inputs."""

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
            cls.validate_inputs,
            cls.extract_structures_from_files,
            cls.submit_jobs,
            cls.collate_results,
        )

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


@calcfunction
def extract_structures_from_xyz(file: SinglefileData):
    """Parse a SinglefileData XYZ trajectory into individual StructureData nodes."""
    with file.open(mode="r") as f:
        lines = f.readlines()

    structures = {}
    line_count = len(lines)
    i = 0
    frame_idx = 0

    while i < line_count:
        line = lines[i].strip()
        try:
            natoms = int(line)
        except ValueError as e:
            raise Exception("Invalid XYZ format detected.") from e
        if (i + 2 + natoms) > line_count:
            raise Exception("XYZ file truncation detected.")

        # Create the base StructureData object
        structure = StructureData(pbc=(False, False, False))

        # Read the comment line
        i += 1
        line = lines[i].strip()
        cell = None
        if "Lattice=" in line:
            match = re.search(r'Lattice="([^"]+)"', line)
            if match:
                lat_vals = [float(x) for x in match.group(1).split()]
                if len(lat_vals) == 9:
                    cell = [lat_vals[0:3], lat_vals[3:6], lat_vals[6:9]]
            pbc = [True, True, True]
            if "pbc=" in line:
                match_pbc = re.search(r'pbc="([^"]+)"', line)
                if match_pbc:
                    pbc_vals = match_pbc.group(1).split()
                    if len(pbc_vals) == 3:
                        # Robust check: converts 'T', 'True', or '1' to True
                        pbc = [val.upper() in ["T", "TRUE", "1"] for val in pbc_vals]

            # Assign the parse cell parameters to the StructureData object
            structure.cell = cell
            structure.pbc = pbc

        i += 1
        for atmi in range(natoms):
            atom_line = lines[i + atmi].strip().split()
            if len(atom_line) < 4:
                raise Exception(
                    f"Invalid atom entry in xyz file: {line[i + atmi].strip()}"
                )

            structure.append_atom(
                position=[
                    float(atom_line[1]),
                    float(atom_line[2]),
                    float(atom_line[3]),
                ],
                symbols=atom_line[0],
            )

        structures[
            f"{file.filename.replace(' ', '_').strip('.xyz')}_frame_{frame_idx}"
        ] = structure
        frame_idx += 1
        i += natoms

    return structures


@calcfunction
def combine_into_extended_xyz(**kwargs) -> SinglefileData:
    """Combine a set of batch results into a single extxyz file."""
    structures: list[StructureData] = []
    energies: list[Float] = []
    arrays: list[ArrayData] = []

    # Extract individual structures and SP calculation results.
    for _key, node in kwargs.items():
        if isinstance(node, TrajectoryData):
            for tindex in range(node.numsteps):
                structures.append(node.get_step_structure(tindex))
        elif isinstance(node, StructureData):
            structures.append(node)
        elif isinstance(node, Float):
            energies.append(node)
        elif isinstance(node, ArrayData):  # Ensure after TrajectoryData !!
            arrays.append(node)
        else:
            raise ValueError(
                f"Invalid input of type {node} to 'combine_into_extended_xyz' "
                "calcfunction."
            )
    if not structures:
        raise ValueError("No StructureData nodes were provided.")
    if len(structures) != len(energies):
        raise ValueError(
            f"Mismatched input lengths: received {len(structures)} structures and "
            f"{len(energies)} energies."
        )
    if arrays:
        if len(arrays) != len(energies):
            raise ValueError(
                f"Mismatched input lengths: received {len(arrays)} arrays and "
                f"{len(energies)} energies."
            )

    xyz_lines = []

    for i in range(len(structures)):
        structure = structures[i]
        energy = energies[i]

        xyz_lines.append(str(len(structure.sites)))

        header_line = f"Energy={energy.value}"
        if arrays:
            forces = arrays[i].get_array("gradients")
            header_line += " Properties=species:S:1:pos:R:3:force:R:3"
        else:
            forces = None
            header_line += " Properties=species:S:1:pos:R:3"
        xyz_lines.append(header_line)

        for j, site in enumerate(structure.sites):
            kind = structure.get_kind(site.kind_name)
            symbol = kind.symbols[0]
            x, y, z = site.position
            site_line = f"{symbol:<4} {x:15.8f} {y:15.8f} {z:15.8f}"
            if forces is not None:
                fx, fy, fz = forces[j]
                site_line += f" {fx:15.8f} {fy:15.8f} {fz:15.8f}"
            xyz_lines.append(site_line)

    content = "\n".join(xyz_lines)
    stream = BytesIO(content.encode("utf-8"))

    return SinglefileData(file=stream, filename="chemshell_batch_workchain.extxyz")
