from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder 
from aiida.common import CalcInfo, CodeInfo
from aiida.orm import SinglefileData, Dict, Float

from aiida_chemshell.utils import ChemShellQMTheory, ChemShellMMTheory

class ChemShellCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapper for ChemShell calculations.
    """

    FILE_SCRIPT = "chemshell_input.py"
    FILE_STDOUT = "output.log"
    FILE_DLFIND = "_dl_find.pun"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the inputs and outputs of the ChemShell calculation.
        """
        super(ChemShellCalculation, cls).define(spec)
        spec.input('structure', valid_type=SinglefileData, required=True, help="The input structure for the ChemShell calculation contained within an '.xyz', '.pun' or '.cjson' file.")
        
        spec.input("calculation_parameters", valid_type=Dict, validator=cls.validate_calculation_parameters, required=False, help="A dictionary of parameters for the ChemShell Task object.")
        spec.input("optimisation_parameters", valid_type=Dict, validator=cls.validate_optimisation_parameters, required=False, help="A dictionary of parameters for the ChemShell geometry optimisation task. If this input is provided, a geometry optimisation task will be configured and added to this job.")

        spec.input("QM_parameters", valid_type=Dict, validator=cls.validate_QM_parameters, required=False, help="A dictionary of parameters for to be passed to the Theory object for the ChemShell calculation.")
        
        spec.input("MM_parameters", valid_type=Dict, validator=cls.validate_MM_parameters, required=False, help="A dictionary of parameters for the ChemShell MM interface.")
        spec.input("forceFieldFile", valid_type=SinglefileData, required=False, help="A file containing the force field parameters for the ChemShell MM interface.")
        
        spec.input("QMMM_parameters", valid_type=Dict, required=False, help="A dictionary of parameters for the ChemShell QM/MM interface.")
        

        spec.output("energy", valid_type=Float, required=True, help="The total energy of the system.")
        spec.output("optimised_structure", valid_type=SinglefileData, required=False, help="The optimised structure of the given system, if a geometry optimisation task was configured and successfully completed. The structure is contained within a ChemShell '.pun' file.")

        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
        spec.inputs["metadata"]["options"]["parser_name"].default = "chemshell"



    @classmethod 
    def get_valid_calculation_parameter_keys(cls) -> tuple[str]:
        """
        Return a tuple of valid parameter keys for the ChemShell calculation.

        Returns
        -------
        validKeys : tuple[str]
            A tuple of valid parameter keys for the ChemShell calculation.
        """
        validKeys = ("gradients", "hessian")
        return validKeys
    
    @classmethod 
    def validate_calculation_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the calculation parameters to be passed to the ChemShell calculation.
        
        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation. If None, no validation is performed.

        Returns 
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error message string.
        """
            
        invalidKeys = set(value.keys()).difference(set(cls.get_valid_calculation_parameter_keys()))
        if invalidKeys:
            return "The following parameter keys are invalid: {0:s}. Valid keys are: {1:s}".format(
                ", ".join(invalidKeys), 
                ", ".join(cls.get_valid_calculation_parameter_keys())
            )
    
        if "gradients" in value.keys():
            if not isinstance(value.get("gradients"), bool):
                return "The 'gradients' parameter must be a boolean value (True or False)."
        if "hessian" in value.keys():
            if not isinstance(value.get("hessian"), bool):
                return "The 'hessian' parameter must be a boolean value (True or False)."
        
        return
    
    @classmethod 
    def get_valid_optimisation_parameter_keys(cls) -> tuple[str]:
        """
        Return a tuple of valid parameter keys for the ChemShell geometry optimisation task.

        Returns
        -------
        validKeys : tuple[str]
            A tuple of valid optimisation parameter keys for the ChemShell calculation.
        """
        validKeys = ("maxcycle", "maxene", "coordinates", "algorithm", "trust_radius", "maxstep", "tolerance", "neb", "nimages", "nebk", "dimer", "delta", "tsrelative")
        # other options not included thus far: redidues, contraints, frag2
        return validKeys
    
    @classmethod
    def validate_optimisation_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the optimisation parameters to be passed to the ChemShell geometry optimisation task.
        
        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell geometry optimisation task. If None, no validation is performed.

        Returns 
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error message string.
        """
        
        invalidKeys = set(value.keys()).difference(set(cls.get_valid_optimisation_parameter_keys()))
        if invalidKeys:
            return "The following parameter keys are invalid: {0:s}. Valid keys are: {1:s}".format(
                ", ".join(invalidKeys), 
                ", ".join(cls.get_valid_optimisation_parameter_keys())
            )
        
        return

    @classmethod 
    def get_valid_QM_parameter_keys(cls) -> tuple[str]:
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
    def validate_QM_parameters(cls, value: Dict | None, _) -> str | None:
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
            # return "The 'theory' parameter for specifying the ChemShell theory interface must be provided."
            value.set("theory", ChemShellQMTheory.NONE)
        
        invalidKeys = set(value.keys()).difference(set(cls.get_valid_QM_parameter_keys()))
        if invalidKeys:
            return "The following parameter keys are invalid: {0:s}. Valid keys are: {1:s}".format(
                ", ".join(invalidKeys), 
                ", ".join(cls.get_valid_parameter_keys())
            )
        
        if isinstance(value.get("theory"), str):
            if value.get("theory").upper() not in ChemShellQMTheory.__members__:
                return "The specified theory '{0:s}' is not a valid ChemShell theory interface within the AiiDA-ChemShell workflow.".format(value.get("theory"))
        elif isinstance(value.get("theory"), int):
            if value.get("theory") not in [t.value for t in ChemShellQMTheory]:
                return "The specified theory '{0:d}' is not a valid ChemShell theory interface within the AiiDA-ChemShell workflow.".format(value.get("theory"))
        elif not isinstance(value.get("theory"), ChemShellQMTheory):
            return "The 'theory' parameter cannot be recognised as a valid ChemShell theory interface. It must be a string, integer or ChemShellQMTheory enum."
        
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
            
        return 
    
    @classmethod 
    def get_valid_MM_parameter_keys(cls) -> tuple[str]:
        """
        Return a tuple of valid parameter keys for the ChemShell MM interface.

        Returns
        -------
        validKeys : tuple[str]
            A tuple of valid MM parameter keys for the ChemShell calculation.
        """
        validKeys = ("theory", "ff", "input", "output")
        return validKeys
    
    @classmethod
    def validate_MM_parameters(cls, value: Dict | None, _) -> str | None:
        print(value.get_dict())
        return 
    
    @classmethod
    def getQMTheoryKey(cls, theory: ChemShellQMTheory) -> str:
        """
        Get the key for the QM theory interface in ChemShell.

        Parameters
        ----------
        theory : ChemShellQMTheory
            The MM theory interface to get the key for.

        Returns
        -------
        str
            The ChemShell class key for the QM theory interface.
        """
        match theory:
            case ChemShellQMTheory.CASTEP:
                return "CASTEP"
            case ChemShellQMTheory.CP2K:
                return "CP2K"
            case ChemShellQMTheory.DFTBP:
                return "DFTBplus"
            case ChemShellQMTheory.FHI_AIMS:
                return "FHIaims"
            case ChemShellQMTheory.GAMESS_UK:
                return "GAMESS_UK"
            case ChemShellQMTheory.GAUSSIAN:
                return "Gaussian"
            case ChemShellQMTheory.LSDALTON:
                return "LSDalton"
            case ChemShellQMTheory.MNDO:
                return "MNDO"
            case ChemShellQMTheory.MOLPRO:
                return "Molpro"
            case ChemShellQMTheory.NWCHEM:
                return "NWChem"
            case ChemShellQMTheory.ORCA:
                return "ORCA"
            case ChemShellQMTheory.PYSCF:
                return "PySCF"
            case ChemShellQMTheory.TURBOMOLE:
                return "TURBOMOLE"
            
        return ''
    
    @classmethod
    def getMMTheoryKey(cls, theory: ChemShellMMTheory) -> str:
        """
        Get the key for the MM theory interface in ChemShell.

        Parameters
        ----------
        theory : ChemShellMMTheory
            The MM theory interface to get the key for.

        Returns
        -------
        str
            The ChemShell class key for the MM theory interface.
        """
        match theory:
            case ChemShellMMTheory.DL_POLY:
                return "DL_POLY"
            case ChemShellMMTheory.GULP:
                return "GULP"
            case ChemShellMMTheory.NAMD:
                return "NAMD"
        return ''
        
        
    def SPCalc_Script_Generator(self) -> str:
        """
        Generates the input script for a ChemShell single-point energy calculation.

        Returns
        -------
        script : str
            A string containing the ChemShell input script for the single-point energy calculation.
        """

        qmTheory = None
        mmTheory = None 

        script = "from chemsh import Fragment, SP\n"
        script += "structure = Fragment(coords='{0:s}')\n".format(self.inputs.structure.filename)

        if "QM_parameters" in self.inputs:
            if isinstance(self.inputs.QM_parameters.get("theory"), str):
                qmTheory = ChemShellQMTheory[self.inputs.QM_parameters.get("theory").upper()]
            elif isinstance(self.inputs.QM_parameters.get("theory"), int):
                qmTheory = ChemShellQMTheory(self.inputs.QM_parameters.get("theory"))
            else:
                qmTheory = self.inputs.QM_parameters.get("theory")
            
            if qmTheory != ChemShellQMTheory.NONE:
                qmTheoryKey = ChemShellCalculation.getQMTheoryKey(qmTheory)

                script += "from chemsh import {0:s}\n".format(qmTheoryKey)
                script += "qmtheory = {0:s}(frag=structure, method='{1:s}', basis='{2:s}', charge={3:d}, functional='{4:s}', mult={5:d}, scftype='{6:s}')\n".format(
                qmTheoryKey, 
                self.inputs.QM_parameters.get("method", "HF"),
                self.inputs.QM_parameters.get("basis", "3-21G"),
                self.inputs.QM_parameters.get("charge", 0), 
                self.inputs.QM_parameters.get("functional", "B3LYP"),
                self.inputs.QM_parameters.get("mult", 1), 
                self.inputs.QM_parameters.get("scftype", "rhf")
            )
                
        if "MM_parameters" in self.inputs:
            if isinstance(self.inputs.MM_parameters.get("theory"), str):
                mmTheory = ChemShellMMTheory[self.inputs.MM_parameters.get("theory").upper()]
            elif isinstance(self.inputs.MM_parameters.get("theory"), int):
                mmTheory = ChemShellMMTheory(self.inputs.MM_parameters.get("theory"))
            else:
                mmTheory = self.inputs.MM_parameters.get("theory")
            if mmTheory != ChemShellMMTheory.NONE:
                mmTheoryKey = ChemShellCalculation.getMMTheoryKey(mmTheory)

                script += "from chemsh import {0:s}\n".format(mmTheoryKey)
                script += "mmtheory = {0:s}(frag=structure, ff='{1:s}')\n".format(#, input='{2:s}', output='{3:s}')\n".format(
                    mmTheoryKey,
                    self.inputs.forceFieldFile.filename,
                    # '',
                    # '' 
                )

        if "optimisation_parameters" in self.inputs:
            # Run a geometry optimisation using DL_FIND 
            optStr = "Opt(theory=qmtheory, maxcycle={0:d}, maxene={1:d}, coordinates='{2:s}', algorithm='{3:s}', "
            optStr += "trust_radius='{4:s}', maxstep={5:f}, tolerance={6:f}, neb='{7:s}', nimages={8:d}, nebk={9:f}, "
            optStr += "dimer={10:s}, delta={11:f}, tsrelative={12:s}).run()\n"
            optStr = optStr.format(
                self.inputs.optimisation_parameters.get("maxcycle", 100),
                self.inputs.optimisation_parameters.get("maxene", 10000),
                self.inputs.optimisation_parameters.get("coordinates", "cartesian"),
                self.inputs.optimisation_parameters.get("algorithm", "lbfgs"),
                self.inputs.optimisation_parameters.get("trust_radius", "constant"), 
                self.inputs.optimisation_parameters.get("maxstep", 0.5), 
                self.inputs.optimisation_parameters.get("tolerance", 0.00045), 
                self.inputs.optimisation_parameters.get("neb", ""), 
                self.inputs.optimisation_parameters.get("nimages", 1), 
                self.inputs.optimisation_parameters.get("nebk", 0.01), 
                str(self.inputs.optimisation_parameters.get("dimer", False)),
                self.inputs.optimisation_parameters.get("delta", 0.01), 
                str(self.inputs.optimisation_parameters.get("tsrelative", False))
            )
            script += optStr
        else:
            # Perform a single point energy calculation 
            if "calculation_parameters" not in self.inputs:
                # Assign default values if none are given 
                self.inputs.calculation_parameters = Dict(dict={})
            
            if qmTheory and not mmTheory:
                script += "SP(theory=qmtheory, gradients={0:s}, hessian={1:s}).run()\n".format(
                    str(self.inputs.calculation_parameters.get("gradients", False)),
                    str(self.inputs.calculation_parameters.get("hessian", False))
                )
            elif mmTheory and not qmTheory:
                script += "SP(theory=mmtheory, gradients={0:s}, hessian={1:s}).run()\n".format(
                    str(self.inputs.calculation_parameters.get("gradients", False)),
                    str(self.inputs.calculation_parameters.get("hessian", False))
                )
            elif qmTheory and mmTheory:
                if self.input.QMMM_parameters:
                    script += "from chemsh import QMMM\n"
                    script += "qmmm = QMMM(frag=structure, qm=qmtheory, mm=mmtheory, qm_region=[{0:s}])\n".format('')
                    script += "SP(theory=qmmm, gradients={0:s}, hessian={1:s}).run()\n".format(
                        str(self.inputs.calculation_parameters.get("gradients", False)),
                        str(self.inputs.calculation_parameters.get("hessian", False))
                    )
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
        calcInfo.local_copy_list = [
            (self.inputs.structure.uuid, self.inputs.structure.filename, self.inputs.structure.filename),
        ]

        if "MM_parameters" in self.inputs:
            calcInfo.local_copy_list.append((self.inputs.forceFieldFile.uuid, self.inputs.forceFieldFile.filename, self.inputs.forceFieldFile.filename))

        if "optimisation_parameters" in self.inputs:
            calcInfo.retrieve_list.append(ChemShellCalculation.FILE_DLFIND)


        return calcInfo 
    