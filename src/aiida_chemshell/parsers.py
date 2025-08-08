"""Defines the calculation parsers for the ChemShell AiiDA plugin."""

import json

import numpy
from aiida.common import ModificationNotAllowed
from aiida.engine import ExitCode
from aiida.orm import ArrayData, Float, SinglefileData
from aiida.parsers.parser import Parser

from aiida_chemshell.calculations import ChemShellCalculation


class ChemShellParser(Parser):
    """AiiDA parser plugin for ChemShell calculations."""

    def parse(self, **kwargs):
        """Parse the output of a ChemShell calculation."""
        if ChemShellCalculation.FILE_STDOUT not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_STDOUT_NOT_FOUND
        if ChemShellCalculation.FILE_RESULTS not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_RESULTS_FILE_NOT_FOUND

        # Read the 'json' formatted results file
        results = json.loads(
            self.retrieved.get_object_content(ChemShellCalculation.FILE_RESULTS)
        )

        # Extract the final energy
        try:
            self.out("energy", Float(results["energy"][0]))
        except (KeyError, ValueError):
            return self.exit_codes.ERROR_MISSING_FINAL_ENERGY
        except ModificationNotAllowed as e:
            raise e

        # Extract gradients/hessian if they are requested
        if "calculation_parameters" in self.node.inputs:
            if self.node.inputs.calculation_parameters.get("gradients", False):
                try:
                    gradients = numpy.array(results["gradients"])
                except (KeyError, ValueError):
                    return self.exit_codes.ERROR_MISSING_GRADIENTS
                else:
                    grad_data = ArrayData()
                    grad_data.set_array("gradients", gradients)
                    self.out("gradients", grad_data)
            if self.node.inputs.calculation_parameters.get("hessian", False):
                try:
                    hessian = numpy.array(results["hessian"])
                except (KeyError, ValueError):
                    return self.exit_codes.ERROR_MISSING_GRADIENTS
                else:
                    if "gradients" in self.outputs:
                        self.outputs["gradients"].set_array("hessian", hessian)
                    else:
                        grad_data = ArrayData()
                        grad_data.set_array("hessian", hessian)
                        self.out("gradients", grad_data)

        # If the calculation was a geometry optimisation, store the optimised structure
        if "optimisation_parameters" in self.node.inputs:
            if ChemShellCalculation.FILE_DLFIND in self.retrieved.list_object_names():
                # Store the optimised structure file
                with self.retrieved.open(ChemShellCalculation.FILE_DLFIND, "r") as f:
                    self.out(
                        "optimised_structure",
                        SinglefileData(
                            file=f, filename=ChemShellCalculation.FILE_DLFIND
                        ),
                    )
            else:
                return self.exit_codes.ERROR_MISSING_OPTIMISED_STRUCTURE_FILE

        return ExitCode(0)
