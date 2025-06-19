from aiida.engine import ExitCode
from aiida.orm import Float, SinglefileData
from aiida.parsers.parser import Parser 

from aiida_chemshell.calculations import ChemShellCalculation 

class ChemShellParser(Parser):
    """
    AiiDA parser plugin for ChemShell calculations.
    """ 

    def parse(self, **kwargs):
        """
        Parse the output of a ChemShell calculation.
        """

        if "optimisation_parameters" in self.node.inputs:
            # Store the optimised structure file 
            with self.retrieved.open(ChemShellCalculation.FILE_DLFIND, 'r') as f:
                self.out("optimised_structure", SinglefileData(file=f))
            # Read the final converged energy value for the optimised structure 
            with self.retrieved.open(ChemShellCalculation.FILE_STDOUT, 'r') as f:
                for line in f:
                    if "Final converged energy" in line:
                        self.out("energy", Float(line.split()[3]))
                        break 
        else:
            # Read the single point energy from the ChemShell output 
            with self.retrieved.open(ChemShellCalculation.FILE_STDOUT, 'r') as f:
                for line in f:
                    if "Final SP energy" in line:
                        self.out("energy", Float(line.split()[4]))
                        break
        