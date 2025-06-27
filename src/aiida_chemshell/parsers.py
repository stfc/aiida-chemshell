from aiida.engine import ExitCode
from aiida.orm import Float, SinglefileData, ArrayData
from aiida.parsers.parser import Parser 
import json
import numpy 

from aiida_chemshell.calculations import ChemShellCalculation 

class ChemShellParser(Parser):
    """
    AiiDA parser plugin for ChemShell calculations.
    """ 

    def parse(self, **kwargs):
        """
        Parse the output of a ChemShell calculation.
        """

        if ChemShellCalculation.FILE_STDOUT not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_STDOUT_NOT_FOUND
        if ChemShellCalculation.FILE_RESULTS not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_RESULTS_FILE_NOT_FOUND
        
        # Read the 'json' formatted results file 
        results = json.loads(self.retrieved.get_object_content(ChemShellCalculation.FILE_RESULTS))

        # Extract the final energy 
        try:
            self.out("energy", Float(results["energy"][0]))
        except:
            return self.exit_codes.ERROR_MISSING_FINAL_ENERGY

        # Extract gradients/hessian if they are requested 
        if "calculation_parameters" in self.node.inputs:
            if self.node.inputs.calculation_parameters.get("gradients", False):
                try:
                    gradients = numpy.array(results["gradients"])
                    gradData = ArrayData() 
                    gradData.set_array("gradients", gradients)
                    self.out("gradients", gradData)
                except:
                    return self.exit_codes.ERROR_MISSING_GRADIENTS
            if self.node.inputs.calculation_parameters.get("hessian", False):
                try:
                    hessian = numpy.array(results["hessian"])
                    if "gradients" in self.outputs:
                        self.outputs["gradients"].set_array("hessian", hessian)
                    else:
                        gradData = ArrayData() 
                        gradData.set_array("hessian", hessian)
                        self.out("gradients", gradData)
                except:
                    return self.exit_codes.ERROR_MISSING_GRADIENTS

        # If the calculation was a geometry optimisation, store the optimised structure 
        if "optimisation_parameters" in self.node.inputs:
            if ChemShellCalculation.FILE_DLFIND not in self.retrieved.list_object_names():
                return self.exit_codes.ERROR_MISSING_OPTIMISED_STRUCTURE_FILE
            # Store the optimised structure file 
            with self.retrieved.open(ChemShellCalculation.FILE_DLFIND, 'r') as f:
                self.out("optimised_structure", SinglefileData(file=f, filename=ChemShellCalculation.FILE_DLFIND))

        return ExitCode(0)
        