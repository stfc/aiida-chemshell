"""Defines the parser for the split trajectory CalcJob."""

from pathlib import Path

from aiida.engine import ExitCode
from aiida.orm import FolderData
from aiida.parsers.parser import Parser


class SplitTrajectoryParser(Parser):
    """AiiDA parser plugin for SplitTrajectory CalcJob."""

    def parse(self, **kwargs):
        """AiiDA parser plugin for SplitTrajectory CalcJob."""
        traj_folder = FolderData(
            label="ExtXYZ Trajectory Folder",
            description=(
                "Folder contianing extXYZ files for each step in from a given "
                "trajectory."
            ),
        )
        tmp_folder = Path(kwargs["retrieved_temporary_folder"])
        for file in tmp_folder.iterdir():
            traj_folder.put_object_from_file(file, file.name)
        # for file_name in self.retrieved.list_object_names():
        #     if "trajectory" in file_name:
        #         traj_folder.put_object_from_bytes(
        #             self.retrieved.get_object_content(file_name, 'rb'),
        #             file_name
        #         )
        if len(traj_folder.list_object_names()) < 1:
            return self.exit_codes.ERROR_NO_TRAJECTORY_FILES
        self.out("trajectory_folder", traj_folder)
        return ExitCode(0)
