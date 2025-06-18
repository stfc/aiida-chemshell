from aiida.parsers.parser import Parser 
from aiida.orm import Float

class ChemShellParser(Parser):
    """
    AiiDA parser plugin for ChemShell calculations.
    """ 

    def parse(self, **kwargs):
        """
        Parse the output of a ChemShell calculation.
        """

        # Read the single point energy from the ChemShell output 
        with self.retrieved.open("output.log", 'r') as f:
            for line in f:
                if "Final SP energy" in line:
                    energy = Float(line.split()[4])
                    self.out("energy", energy)
                    break 

        