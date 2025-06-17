from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder 
from aiida.common import CalcInfo, CodeInfo
from aiida.orm import SinglefileData, Dict

from aiida_chemshell.utils import ChemShellTheory

class ChemShellCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapper for ChemShell calculations.
    """

    FILE_SCRIPT = "chemshell_input.py"
    FILE_STDOUT = "output.log"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the inputs and outputs of the ChemShell calculation.
        """
        super(ChemShellCalculation, cls).define(spec)
        spec.input('structure', valid_type=SinglefileData, required=True, help="The input structure for the ChemShell calculation contained within an '.xyz', '.pun' or '.cjson' file.")
        spec.input("parameters", valid_type=Dict, validator=cls.validate_parameters, required=True, help="A dictionary of parameters for to be passed to the Theory object for the ChemShell calculation.")

        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
        spec.inputs["metadata"]["options"]["parser_name"].default = "chemshell"

    @classmethod 
    def get_valid_parameter_keys(cls) -> tuple[str]:
        """
        Return a tuple of valid parameter keys for the ChemShell calculation.

        Returns
        -------
        validKeys : tuple[str]
            A tuple of valid Theory parameter keys for the ChemShell calculation.
        """
        validKeys = ("theory", "method", "basis", "charge", "functional", "mult", "scftype")
        return validKeys

    @classmethod
    def validate_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the Theory object parameters to be passed to the ChemShell calculation.
        
        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation. If None, no validation is performed.

        Returns 
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error message string.
        """
        
        if "theory" not in value.keys():
            return "The 'theory' parameter for specifying the ChemShell theory interface must be provided."
        
        invalidKeys = set(value.keys()).difference(set(cls.get_valid_parameter_keys()))
        if invalidKeys:
            return "The following parameter keys are invalid: {0:s}. Valid keys are: {1:s}".format(
                ", ".join(invalidKeys), 
                ", ".join(cls.get_valid_parameter_keys())
            )
        
        if isinstance(value.get("theory"), str):
            if value.get("theory").upper() not in ChemShellTheory.__members__:
                return "The specified theory '{0:s}' is not a valid ChemShell theory interface within the AiiDA-ChemShell workflow.".format(value.get("theory"))
        elif isinstance(value.get("theory"), int):
            if value.get("theory") not in [t.value for t in ChemShellTheory]:
                return "The specified theory '{0:d}' is not a valid ChemShell theory interface within the AiiDA-ChemShell workflow.".format(value.get("theory"))
        elif not isinstance(value.get("theory"), ChemShellTheory):
            return "The 'theory' parameter cannot be recognised as a valid ChemShell theory interface. It must be a string, integer or ChemShellTheory enum."
        
        if "method" in value.keys():
            if value.get("method").upper() not in ["HF", "DFT"]:
                return "The specified method key ('{0:s}') us not valid.".format(value.get("method"))
        if "charge" in value.keys():
            if not isinstance(value.get("charge"), int):
                return "The 'charge' parameter must be an integer."
        if "mult" in value.keys():
            if not isinstance(value.get("mult"), int):
                return "The 'mult' parameter must be an integer."
        if "scftype" in value.keys():
            if value.get("scftype").upper() not in ["RHF", "UHF", "ROHF", "RKS", "UKS", "ROKS"]:
                return "The 'scftype' parameter must be one of 'RHF', 'UHF' or 'ROHF' (or analogous 'rks', 'uks' or 'roks')."
        
        
    def SPCalc_Script_Generator(self) -> str:
        """
        Generates the input script for a ChemShell single-point energy calculation.

        Returns
        -------
        script : str
            A string containing the ChemShell input script for the single-point energy calculation.
        """
 
        if isinstance(self.inputs.parameters.get("theory"), str):
            theory = ChemShellTheory[self.inputs.parameters.get("theory").upper()]
        elif isinstance(self.inputs.parameters.get("theory"), int):
            theory = ChemShellTheory(self.inputs.parameters.get("theory"))
        else:
            theory = self.inputs.parameters.get("theory")
            
        match theory:
            case ChemShellTheory.CASTEP:
                theory_key = "CASTEP"
            case ChemShellTheory.CP2K:
                theory_key = "CP2K"
            case ChemShellTheory.DFTBP:
                theory_key = "DFTBplus"
            case ChemShellTheory.FHI_AIMS:
                theory_key = "FHIaims"
            case ChemShellTheory.GAMESS_UK:
                theory_key = "GAMESS_UK"
            case ChemShellTheory.GAUSSIAN:
                theory_key = "Gaussian"
            case ChemShellTheory.LSDALTON:
                theory_key = "LSDalton"
            case ChemShellTheory.MNDO:
                theory_key = "MNDO"
            case ChemShellTheory.MOLPRO:
                theory_key = "Molpro"
            case ChemShellTheory.NWCHEM:
                theory_key = "NWChem"
            case ChemShellTheory.ORCA:
                theory_key = "ORCA"
            case ChemShellTheory.PYSCF:
                theory_key = "PySCF"
            case ChemShellTheory.TURBOMOLE:
                theory_key = "TURBOMOLE"

        script = "from chemsh import Fragment, {0:s}, SP\n".format(theory_key)
        script += "structure = Fragment(coords='{0:s}')\n".format(self.inputs.structure.filename)
        script += "theory = {0:s}(frag=structure, method='{1:s}', basis='{2:s}', charge={3:d}, functional='{4:s}', mult={5:d}, scftype='{6:s}')\n".format(
            theory_key, 
            self.inputs.parameters.get("method", "HF"),
            self.inputs.parameters.get("basis", "3-21G"),
            self.inputs.parameters.get("charge", 0), 
            self.inputs.parameters.get("functional", "B3LYP"),
            self.inputs.parameters.get("mult", 1), 
            self.inputs.parameters.get("scftype", "rhf")
        )

        script += "SP(theory=theory, gradients=False, hessian=False).run()\n"
        
        return script 


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
        params = self.inputs.parameters.get_dict() 
        params["structure"] = self.inputs.structure.filename
        inputScript = self.SPCalc_Script_Generator()

        with folder.open(ChemShellCalculation.FILE_SCRIPT, 'w') as f:
            f.write(inputScript)

        codeInfo = CodeInfo()
        codeInfo.code_uuid = self.inputs.code.uuid
        codeInfo.cmdline_params = [ChemShellCalculation.FILE_SCRIPT,]
        codeInfo.stdout_name = ChemShellCalculation.FILE_STDOUT
        
        calcInfo = CalcInfo()
        calcInfo.codes_info = [codeInfo]
        calcInfo.retrieve_temporary_list = []
        calcInfo.provenance_exclude_list = [] 
        calcInfo.retrieve_list = [ChemShellCalculation.FILE_STDOUT,]
        calcInfo.local_copy_list = [(self.inputs.structure.uuid, self.inputs.structure.filename, self.inputs.structure.filename),]


        return calcInfo 
    