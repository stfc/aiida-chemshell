from aiida.engine import CalcJob 

class ChemShellCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapper for ChemShell calculations.
    """

    @classmethod
    def define(cls, spec):
        """
        Define the inputs and outputs of the ChemShell clalculation.
        """
        super(ChemShellCalculation, cls).define(spec)
        
        return 
    
    def prepare_for_submission(self, dir):
        
        return 