from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder 
from aiida.common import CalcInfo

class ChemShellCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapper for ChemShell calculations.
    """

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the inputs and outputs of the ChemShell clalculation.
        """
        super(ChemShellCalculation, cls).define(spec)
         
    
    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """
        Prepare the ChemShell calculation for submission.

        Params
        ------
        folder : Folder 
            An `aiida.common.folders.Folder` specifying the temporary working directory for the calculation.

        Returns
        -------
        calcInfo : CalcInfo 
            An `aiida.common.CalcInfo` instance. 
        """
        calcInfo = CalcInfo()
        return calcInfo 