"""Defines the parser for the split trajectory CalcJob."""

import os

from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser


class CreateJanusTrainingInputsParser(Parser):
    """AiiDA parser plugin for SplitTrajectory CalcJob."""

    def parse(self, **kwargs):
        """AiiDA parser plugin for SplitTrajectory CalcJob."""
        description_str = "MLIP {} data set extracted from ChemShell calculation."
        retrieved_temporary_folder = kwargs.get("retrieved_temporary_folder", None)

        # with self.retrieved.open("train.xyz", "r") as f:
        with open(os.path.join(retrieved_temporary_folder, "train.xyz"), "rb") as f:
            self.out(
                "training_input",
                SinglefileData(
                    file=f,
                    filename="train.xyz",
                    label="MLIP training data.",
                    description=description_str.format("training"),
                ),
            )

        # with self.retrieved.open("test.xyz", "r") as f:
        with open(os.path.join(retrieved_temporary_folder, "test.xyz"), "rb") as f:
            self.out(
                "test_input",
                SinglefileData(
                    file=f,
                    filename="test.xyz",
                    label="MLIP test data.",
                    description=description_str.format("testing"),
                ),
            )

        # with self.retrieved.open("valid.xyz", "r") as f:
        with open(os.path.join(retrieved_temporary_folder, "valid.xyz"), "rb") as f:
            self.out(
                "validation_input",
                SinglefileData(
                    file=f,
                    filename="valid.xyz",
                    label="MLIP validation data.",
                    description=description_str.format("validation"),
                ),
            )

        return ExitCode(0)
