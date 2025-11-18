"""CalcJob to create individual extended XYZ files for steps in an optimisation."""

from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import FolderData, SinglefileData, Str


class SplitTrajectory(CalcJob):
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
            "filename",
            valid_type=Str,
            required=False,
            help="The name to give the directory of output files.",
        )

        spec.output(
            "trajectory_folder",
            valid_type=FolderData,
            required=True,
            help="The directory containing the resulting extXYZ files.",
        )

        ## Metadata
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 2,
        }
        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "chemshell.split_traj"

        return

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Perform the python task to split the input trajectories."""
        script = self.generate_script()
        with folder.open("input.py", "w") as f:
            f.write(script)

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
        calc_info.retrieve_temporary_list = []
        calc_info.provenance_exclude_list = []
        calc_info.retrieve_temporary_list = [
            "trajectory*xyz",
        ]
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
with open("path_force.xyz", 'r') as f:
    force = f.readlines()
natms = int(path[0])
xyz_str = "\\n{0:8s} {1:12.8f} {2:12.8f} {3:12.8f} {4:12.8f} {5:12.8f} {6:12.8f}"
i = 2
step = 0
while i < len(path):
    # Write a new xyz file for given step
    with open(f"trajectory_{step:03d}.xyz", 'w') as f:
        # write the number of atoms
        f.write(f"{natms:10d}")
        # Write the header line
        f.write(f"\\nProperties=\\"species:S:1:pos:R:3:force:R:3\\" Step={step}")
        for j in range(natms):
            path_line = path[i + j].split()
            force_line = force[i + j].split()
            f.write(
                xyz_str.format(
                    path_line[0],
                    float(path_line[1]),
                    float(path_line[2]),
                    float(path_line[3]),
                    float(force_line[1]),
                    float(force_line[2]),
                    float(force_line[3])
                )
            )
    i += natms + 2
    step += 1
"""
