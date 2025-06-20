from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder 
from aiida.common import CalcInfo, CodeInfo
from aiida.orm import SinglefileData, Dict, Float, Str

from aiida_chemshell.utils import ChemShellQMTheory, ChemShellMMTheory

class ChemShellCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapper for ChemShell calculations.

    Currently supports the following tasks: 
      - Single point energy 
      - Geometry optimisation 

    """

    FILE_SCRIPT = "chemshell_input.py"
    FILE_STDOUT = "output.log"
    FILE_DLFIND = "_dl_find.pun"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the inputs, outputs and metadata of the ChemShell calculation.
        """
        super(ChemShellCalculation, cls).define(spec)
        spec.input('structure', valid_type=SinglefileData, required=True, help="The input structure for the ChemShell calculation contained within an '.xyz', '.pun' or '.cjson' file.")
        
        # Task object parameters 
        spec.input("calculation_parameters", valid_type=Dict, validator=cls.validate_calculation_parameters, required=False, help="A dictionary of parameters for the ChemShell Task object.")
        spec.input("optimisation_parameters", valid_type=Dict, validator=cls.validate_optimisation_parameters, required=False, help="A dictionary of parameters for the ChemShell geometry optimisation task. If this input is provided, a geometry optimisation task will be configured and added to this job.")

        # Theory objects parameters 
        spec.input("qm_theory", valid_type=Str, validator=cls.validate_qm_theory, required=False, help="Set the QM theory interface for the chemshell calculation.")
        spec.input("QM_parameters", valid_type=Dict, validator=cls.validate_QM_parameters, required=False, help="A dictionary of parameters for to be passed to the Theory object for the ChemShell calculation.")
        spec.input("MM_parameters", valid_type=Dict, validator=cls.validate_MM_parameters, required=False, help="A dictionary of parameters for the ChemShell MM interface.")
        # The force field input is specified as a file (not a string) and is not directly contained within the MM_parameters dictionary due to serialisation of SingfileData object types 
        spec.input("forceFieldFile", valid_type=SinglefileData, required=False, help="A file containing the force field parameters for the ChemShell MM interface.")
        spec.input("QMMM_parameters", valid_type=Dict, required=False, help="A dictionary of parameters for the ChemShell QM/MM interface.")
        
        # Calculation outputs 
        spec.output("energy", valid_type=Float, required=True, help="The total energy of the system.")
        spec.output("optimised_structure", valid_type=SinglefileData, required=False, help="The optimised structure of the given system, if a geometry optimisation task was configured and successfully completed. The structure is contained within a ChemShell '.pun' file.")

        # Metadata 
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
        spec.inputs["metadata"]["options"]["parser_name"].default = "chemshell"

        return 

    @classmethod 
    def get_valid_calculation_parameter_keys(cls) -> tuple[str]:
        """
        Return a tuple of valid parameter keys for the ChemShell Single Point calculation.

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
        Validate the calculation parameters to be passed to the ChemShell Single Point calculation.
        
        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation. If None, no validation is performed.

        Returns 
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error message string.
        """
        
        # Check for valid parameter keys 
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

        # Check for invalid parameters keys 
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
        validKeys = ("method", "basis", "charge", "functional", "mult", "scftype",
                     "damping", "diis", "direct", "guess", "maxiter", "path", "pseudopotential",
                     "restart", "scf")
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
        
        invalidKeys = set(value.keys()).difference(set(cls.get_valid_QM_parameter_keys()))
        if invalidKeys:
            # Checks for invalid parameter keys 
            return "The following parameter keys are invalid: {0:s}. Valid keys are: {1:s}".format(
                ", ".join(invalidKeys), 
                ", ".join(cls.get_valid_QM_parameter_keys())
            )     

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
            

        # TODO: check SCF parameters 
        return 
    
    @classmethod 
    def validate_qm_theory(cls, value: str | None, _) -> str | None:
        # Check the specified theory interface 
        if value.value.upper() not in ChemShellQMTheory.__members__:
            return "The specified theory '{0:s}' is not a valid ChemShell theory interface within the AiiDA-ChemShell workflow.".format(value)
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
        validKeys = ("theory", "input", "output")
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
        
        
    def chemsh_script_generator(self) -> str:
        """
        Generates the input script for a ChemShell single-point energy calculation.

        Returns
        -------
        script : str
            A string containing the ChemShell input script for the single-point energy calculation.
        """

        qmTheory = None
        mmTheory = None 

        script = "from chemsh import Fragment\n"
        script += "structure = Fragment(coords='{0:s}')\n".format(self.inputs.structure.filename)

        ## Setup Theory objects 

        # if "QM_parameters" in self.inputs:
        if self.inputs.get("qm_theory", Str("NONE")).value.upper() != "NONE":
            # Creates a quantum mechanics Theory object 
            qmTheory = ChemShellQMTheory[self.inputs.qm_theory.value.upper()]
            
            if qmTheory != ChemShellQMTheory.NONE:
                qmTheoryKey = ChemShellCalculation.getQMTheoryKey(qmTheory)

                script += "from chemsh import {0:s}\n".format(qmTheoryKey)
                paramStr = "" 
                if "QM_parameters" in self.inputs:
                    for key in self.inputs.QM_parameters.keys():
                        val = self.inputs.QM_parameters.get(key)
                        if isinstance(val, str):
                            paramStr += ", " + key + "='" + val + "'"
                        else:
                            paramStr += ", " + key + "=" + str(val)
                script += "qmtheory = {0:s}(frag=structure".format(qmTheoryKey) + paramStr + ")\n"
                    
                
        if "MM_parameters" in self.inputs:
            # Creates a molecular mechanics Theory object 
            mmTheory = ChemShellMMTheory[self.inputs.MM_parameters.get("theory").upper()]
            if mmTheory != ChemShellMMTheory.NONE:
                mmTheoryKey = ChemShellCalculation.getMMTheoryKey(mmTheory)

                script += "from chemsh import {0:s}\n".format(mmTheoryKey)
                script += "mmtheory = {0:s}(frag=structure, ff='{1:s}')\n".format(#, input='{2:s}', output='{3:s}')\n".format(
                    mmTheoryKey,
                    self.inputs.forceFieldFile.filename,
                    # '',
                    # '' 
                )

        ## Setup Task objects 

        if "optimisation_parameters" in self.inputs:
            # Run a geometry optimisation using DL_FIND
            
            if qmTheory and not mmTheory:
                tStr = "qmtheory"
            elif mmTheory and not qmTheory:
                tStr = "mmtheory"
            elif qmTheory and mmTheory:
                #TODO: qm/mm theory setup 
                tStr = "qmtheory" 
            else:
                #TODO: Catch exception here 
                pass 

            optStr = "Opt(theory={13:s}, maxcycle={0:d}, maxene={1:d}, coordinates='{2:s}', algorithm='{3:s}', "
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
                str(self.inputs.optimisation_parameters.get("tsrelative", False)),
                tStr
            )
            script += optStr
        else:
            # Perform a single point energy calculation (default calculation type)
            script += "from chemsh import SP\n"
            if "calculation_parameters" not in self.inputs:
                # Assign default values if none are given 
                self.inputs.calculation_parameters = Dict(dict={})
            
            if qmTheory and not mmTheory:
                # Runs a QM single point energy calculation
                script += "SP(theory=qmtheory, gradients={0:s}, hessian={1:s}).run()\n".format(
                    str(self.inputs.calculation_parameters.get("gradients", False)),
                    str(self.inputs.calculation_parameters.get("hessian", False))
                )
            elif mmTheory and not qmTheory:
                # Runs a MM single point energy calculation 
                script += "SP(theory=mmtheory, gradients={0:s}, hessian={1:s}).run()\n".format(
                    str(self.inputs.calculation_parameters.get("gradients", False)),
                    str(self.inputs.calculation_parameters.get("hessian", False))
                )
            elif qmTheory and mmTheory:
                # If both a provided assume the user wants to run a QM/MM calculation and check for require parameters
                if self.input.QMMM_parameters:
                    # Runs a QM/MM single point energy calculation 
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

        # Create the ChemShell input script 
        inputScript = self.chemsh_script_generator()
        with folder.open(ChemShellCalculation.FILE_SCRIPT, 'w') as f:
            f.write(inputScript)

        # Define the AiiDA code parameters 
        codeInfo = CodeInfo()
        codeInfo.code_uuid = self.inputs.code.uuid
        codeInfo.cmdline_params = [ChemShellCalculation.FILE_SCRIPT,]
        codeInfo.stdout_name = ChemShellCalculation.FILE_STDOUT
        
        # Setup the calculation information object 
        calcInfo = CalcInfo()
        calcInfo.codes_info = [codeInfo]
        calcInfo.retrieve_temporary_list = []
        calcInfo.provenance_exclude_list = [] 
        calcInfo.retrieve_list = [ChemShellCalculation.FILE_STDOUT,]
        calcInfo.local_copy_list = [
            (self.inputs.structure.uuid, self.inputs.structure.filename, self.inputs.structure.filename),
        ]

        # If running with an MM theory a force field file is required and copied
        if "MM_parameters" in self.inputs:
            calcInfo.local_copy_list.append((self.inputs.forceFieldFile.uuid, self.inputs.forceFieldFile.filename, self.inputs.forceFieldFile.filename))

        # If performing a geometry optimisation retrieve the generated _dl_find.pun file containing the optimised structure 
        if "optimisation_parameters" in self.inputs:
            calcInfo.retrieve_list.append(ChemShellCalculation.FILE_DLFIND)


        return calcInfo 
    