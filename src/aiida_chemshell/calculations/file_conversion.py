"""CalcJob to create individual extended XYZ files for steps in an optimisation."""

from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec, PortNamespace
from aiida.orm import ArrayData, Dict, SinglefileData, Str

from aiida_chemshell.units import UnitsConverter


class CreateJanusTrainingInputsCalcJob(CalcJob):
    """CalcJob to split an XYZ trajectory into individual ext XYZ files."""

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define the inputs, outputs and metadata for the CalcJob."""
        super().define(spec)
        spec.input(
            "path",
            valid_type=SinglefileData,
            required=True,
            help="An XYZ trajectory file containing the positions of each step.",
        )
        spec.input(
            "force",
            valid_type=SinglefileData,
            required=True,
            help="An XYZ trajectory file containing the forces at each step.",
        )
        spec.input(
            "energies",
            valid_type=ArrayData,
            required=True,
            help="The calculated dft energy of each step in the data series in a.u.",
        )
        spec.input(
            "atom_energies",
            valid_type=Dict,
            required=True,
            help="The isolated atomic energies for the training set.",
        )

        spec.input(
            "filename",
            valid_type=Str,
            required=False,
            help="The name to give the directory of output files.",
        )

        spec.output(
            "training_input",
            valid_type=SinglefileData,
            validator=cls.validate_path_file,
            required=True,
            help="The main training data set in extended XYZ format.",
        )
        spec.output(
            "validation_input",
            valid_type=SinglefileData,
            vlidatory=cls.validate_path_file,
            required=True,
            help="The validation data set in extended XYZ format.",
        )
        spec.output(
            "test_input",
            valid_type=SinglefileData,
            required=True,
            help="The testing data set in extended XYZ format.",
        )

        ## Metadata
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 2,
        }
        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "chemshell.file_conversion.mlip_training"

        # Exit Codes
        spec.exit_code(
            300,
            "ERROR_NO_TRAJECTORY_FILES",
            message="No trajectory files have been produced.",
        )

        return

    @classmethod
    def validate_path_file(
        cls, value: SinglefileData, namepsace: PortNamespace
    ) -> str | None:
        """Perform validation checks on the input path xyz files."""
        if not isinstance(value, SinglefileData):
            return "Input trajectory needs to be a SinglefileData node."
        content = value.get_content("r")
        lines = content.split("\n")
        natoms = int(lines[0])
        if len(lines) % (natoms + 2) != 0:
            return "Invalid XYZ trajectory structure detected."
        if len(lines) // (natoms + 2) < 5:
            return "Not enough individual configurations within input trajectory."
        return None

    def create_isolated_atom_energy_xyz(self) -> str:
        """Create the isolated atomistic energies in the required XYZ format."""
        xyz_str = ""
        zero = 0.0
        for atom in self.inputs.atom_energies.keys():
            xyz_str += "1\nProperties=species:S:1:pos:R:3:dft_forces:R:3 "
            energy = UnitsConverter.hartree_to_ev(self.inputs.atom_energies[atom])
            xyz_str += f"dft_energy={energy} "
            xyz_str += 'config_type=IsolatedAtom pbc="F F F"\n'
            xyz_str += f"{atom:8s} {zero:.4f} {zero:.4f} {zero:.4f} "
            xyz_str += f"{zero:.4f} {zero:.4f} {zero:.4f}\n"
        return xyz_str

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Perform the python task to split the input trajectories."""
        script = self.generate_script()
        with folder.open("input.py", "w") as f:
            f.write(script)

        with folder.open("energies.txt", "w") as f:
            for val in self.inputs.energies.get_array("energies"):
                f.write(f"{UnitsConverter.hartree_to_ev(val):.10f}\n")

        with folder.open("isolated_atoms.xyz", "w") as f:
            f.write(self.create_isolated_atom_energy_xyz())

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        if "chemsh.x" in str(self.inputs.code.filepath_executable):
            code_info.cmdline_params = [
                "input.py",
            ]
        else:
            n_machines = self.inputs.metadata.options.resources.get("num_machines")
            n_mpi_pm = self.inputs.metadata.options.resources.get(
                "num_mpiprocs_per_machine"
            )
            tot_mpi = n_machines * n_mpi_pm
            code_info.cmdline_params = [
                "-np",
                self.inputs.metadata.options.resources.get("tot_num_mpiprocs", tot_mpi),
                "input.py",
            ]

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.provenance_exclude_list = []
        calc_info.retrieve_temporary_list = ["train.xyz", "valid.xyz", "test.xyz"]
        calc_info.local_copy_list = [
            (
                self.inputs.path.uuid,
                self.inputs.path.filename,
                self.inputs.path.filename,
            ),
            (
                self.inputs.force.uuid,
                self.inputs.force.filename,
                self.inputs.force.filename,
            ),
        ]

        return calc_info

    def generate_script(self) -> str:
        """Generate the python script for splitting the trajectory files."""
        return """
with open("path.xyz", "r") as f:
    path = f.readlines()
with open("path_force.xyz", "r") as f:
    force = f.readlines()
energies = []
with open("energies.txt", 'r') as f:
    for line in f:
        energies.append(float(line))
natms = int(path[0])
nsteps = len(path) // (natms + 2)
valid_interval = 5
test_interval = 10
if nsteps < test_interval:
    test_interval = nsteps
    valid_interval = (nsteps // 2) + 1
for step in range(nsteps):
    if (step + 1) % test_interval == 0:
        fname = "test.xyz"
    elif (step + 1) % valid_interval == 0:
        fname = "valid.xyz"
    else:
        fname = "train.xyz"
    with open(fname, "a") as f:
        f.write(f"{natms}\\n")
        f.write(f'Step={step}  ')
        f.write(f'Properties=species:S:1:pos:R:3:dft_forces:R:3 pbc="F F F" ')
        f.write(f'energy={energies[step]}\\n')
        for i in range(2, natms + 2):
            index = (step * (natms + 2)) + i
            f.write(path[index].strip("\\n") + "   ")
            f.write("   ".join(force[index].split()[1:]))
            f.write("\\n")
with open("isolated_atoms.xyz", "r") as f:
    atoms = f.read()
with open("train.xyz", "a") as f:
    f.write(atoms)
"""
