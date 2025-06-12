from aiida.parsers.parser import Parser 

class ChemShellParser(Parser):
    """
    AiiDA parser plugin for ChemShell calculations.
    """ 

    def parse(self, **kwargs):
        """
        Parse the output of a ChemShell calculation.
        """
