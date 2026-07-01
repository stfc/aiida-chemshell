"""Workflow for processing a series of structures from a single input."""

import re

from aiida.engine import ProcessSpec, ToContext, WorkChain, calcfunction
from aiida.orm import ProcessNode, SinglefileData, StructureData, TrajectoryData
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
