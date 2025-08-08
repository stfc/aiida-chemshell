"""Core ChemShell calculations module."""

from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import ArrayData, Dict, Float, SinglefileData, StructureData

from aiida_chemshell.utils import ChemShellMMTheory, ChemShellQMTheory


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
    FILE_TMP_STRUCTURE = "input_structure.xyz"
    FILE_RESULTS = "result.json"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """
        Define the inputs, outputs and metadata of the ChemShell calculation.

        Parameters
        ----------
        spec : CalcJobProcessSpec
            The AiiDA Process specification object for the job.
        """
        super().define(spec)
        spec.input(
            "structure",
            valid_type=(SinglefileData, StructureData),
            validator=cls.validate_structure_file,
            required=True,
            help="The input structure for the ChemShell calculation either contained \
                within an '.xyz', '.pun' or '.cjson' file or as a StructureData \
                    instance.",
        )

        ## Task object parameters
        spec.input(
            "calculation_parameters",
            valid_type=Dict,
            validator=cls.validate_calculation_parameters,
            required=False,
            help="A dictionary of parameters for the ChemShell Task object.",
        )
        spec.input(
            "optimisation_parameters",
            valid_type=Dict,
            validator=cls.validate_optimisation_parameters,
            required=False,
            help="A dictionary of parameters for the ChemShell geometry optimisation \
                task. If this input is provided, a geometry optimisation task will be \
                     configured and added to this job.",
        )

        ## Theory objects parameters
        spec.input(
            "qm_parameters",
            valid_type=Dict,
            validator=cls.validate_qm_parameters,
            required=False,
            help="A dictionary of parameters for to be passed to the Theory object \
                for the ChemShell calculation.",
        )
        spec.input(
            "mm_parameters",
            valid_type=Dict,
            validator=cls.validate_mm_parameters,
            required=False,
            help="A dictionary of parameters for the ChemShell MM interface.",
        )
        # The force field input is specified as a file (not a string) and is not
        # directly contained within the MM_parameters dictionary due to serialisation
        # of SingfileData object types
        spec.input(
            "force_field_file",
            valid_type=SinglefileData,
            required=False,
            help="A file containing the force field parameters for the ChemShell \
                MM interface.",
        )
        spec.input(
            "qmmm_parameters",
            valid_type=Dict,
            required=False,
            help="A dictionary of parameters for the ChemShell QM/MM interface.",
        )

        ## Calculation outputs
        spec.output(
            "energy",
            valid_type=Float,
            required=True,
            help="The total energy of the system.",
        )
        spec.output(
            "gradients",
            valid_type=ArrayData,
            required=False,
            help="The gradients (and hessian) of the system if requested. The \
                gradients are contained within an AiiDA ArrayData object with the \
                    key 'gradients'.",
        )
        spec.output(
            "optimised_structure",
            valid_type=SinglefileData,
            required=False,
            help="The optimised structure of the given system, if a geometry \
                optimisation task was configured and successfully completed. The \
                    structure is contained within a ChemShell '.pun' file.",
        )

        ## Metadata
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "chemshell"

        # Exit Codes
        spec.exit_code(
            300,
            "ERROR_STDOUT_NOT_FOUND",
            message="Error accessing the `output.log` ChemShell output file.",
        )
        spec.exit_code(
            301,
            "ERROR_MISSING_FINAL_ENERGY",
            message="ChemShell calculation failed to compute a final energy for \
                the given task.",
        )
        spec.exit_code(
            302,
            "ERROR_MISSING_OPTIMISED_STRUCTURE_FILE",
            message="ChemShell failed to produced the expected optimised structure \
                file.",
        )
        spec.exit_code(
            303,
            "ERROR_RESULTS_FILE_NOT_FOUND",
            message="ChemShell calculation failed to produce the expected results \
                file.",
        )
        spec.exit_code(
            304,
            "ERROR_MISSING_GRADIENTS",
            message="ChemShell calculation failed to compute the requested \
                gradients or hessian for the given task.",
        )

        return

    @classmethod
    def validate_structure_file(
        cls, value: SinglefileData | StructureData | None, _
    ) -> str | None:
        """
        Validate the ChemShell input structure file.

        Parameters
        ----------
        value : SinglefileData | StructureData | None
            The input structure to validate. This can be either a file containing
            the structure or an AiiDA StructureData object. If None, no validation
            is performed.

        Returns
        -------
        str | None
            Returns `None` if no error is found otherwise returns an error message
        """
        if isinstance(value, SinglefileData):
            if value.filename[-4:] not in [".xyz", ".pun"]:
                if value.filename[-6:] != ".cjson":
                    return "Structure file must be either an '.xyz', '.pun' or \
                        '.cjson' formatted structure file."

        return None

    @classmethod
    def get_valid_calculation_parameter_keys(cls) -> tuple[str]:
        """
        Return the valid parameter keys for the ChemShell Single Point calculation.

        Returns
        -------
         : tuple[str]
            A tuple of valid parameter keys for the ChemShell calculation.
        """
        return ("gradients", "hessian")

    @classmethod
    def validate_calculation_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the ChemShell Single Point calculation input parameters.

        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation.
            If None, no validation is performed.

        Returns
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error
            message string.
        """
        # Check for valid parameter keys
        invalid_keys = set(value.keys()).difference(
            set(cls.get_valid_calculation_parameter_keys())
        )
        if invalid_keys:
            return f"The following parameter keys are invalid: \
                {', '.join(invalid_keys):s}. Valid keys are: \
                    {', '.join(cls.get_valid_calculation_parameter_keys()):s}"

        if "gradients" in value.keys():
            if not isinstance(value.get("gradients"), bool):
                return "The 'gradients' parameter must be a Boolean value."
        if "hessian" in value.keys():
            if not isinstance(value.get("hessian"), bool):
                return "The 'hessian' parameter must be a Boolean value."

        return None

    @classmethod
    def get_valid_optimisation_parameter_keys(cls) -> tuple[str]:
        """
        Return the valid parameter keys for a ChemShell geometry optimisation task.

        Returns
        -------
         : tuple[str]
            A tuple of valid optimisation parameter keys for the ChemShell
            calculation.
        """
        # other options not included thus far: redidues, contraints, frag2
        return (
            "maxcycle",
            "maxene",
            "coordinates",
            "algorithm",
            "trust_radius",
            "maxstep",
            "tolerance",
            "neb",
            "nimages",
            "nebk",
            "dimer",
            "delta",
            "tsrelative",
        )

    @classmethod
    def validate_optimisation_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the ChemShell optimisation input parameters.

        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell geometry optimisation task.
            If None, no validation is performed.

        Returns
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error message
            string.
        """
        # Check for invalid parameters keys
        invalid_keys = set(value.keys()).difference(
            set(cls.get_valid_optimisation_parameter_keys())
        )
        if invalid_keys:
            return f"The following parameter keys are invalid: \
                {', '.join(invalid_keys):s}. Valid keys are: \
                    {', '.join(cls.get_valid_optimisation_parameter_keys()):s}"

        # TODO: check the types of the parameters

        return None

    @classmethod
    def get_valid_qm_paramater_keys(cls) -> dict[str:type]:
        """
        Return a tuple of valid parameter keys for the ChemShell calculation.

        Returns
        -------
        validKeys : dict[str: type]
            A tuple of valid Theory parameter keys for the ChemShell calculation.
        """
        return {
            "theory": str,
            "method": str,
            "basis": str,
            "charge": float | int,
            "functional": str,
            "mult": int | float,
            "scftype": str,
            "damping": bool,
            "diis": bool,
            "direct": bool,
            "guess": str,  # TODO: file???
            "maxiter": int,
            "path": str,
            "pseudopotential": str | dict,
            "restart": bool,
            "scf": float,
        }

    @classmethod
    def validate_qm_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the Theory object parameters to be passed to the ChemShell calculation.

        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation.
            If None, no validation is performed.

        Returns
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an error
            message string.
        """
        # Check the specified theory interface
        if value.get("theory", "").upper() not in ChemShellQMTheory.__members__:
            theory = value.get("theory")
            return f"The specified theory '{theory:s}' is not a valid \
                ChemShell theory interface within the AiiDA-ChemShell workflow."

        valid_keys = cls.get_valid_qm_paramater_keys()

        # Check for valid parameter keys
        invalid_keys = set(value.keys()).difference(set(valid_keys.keys()))
        if invalid_keys:
            return f"The following parameter keys are invalid: \
                {', '.join(invalid_keys):s}. Valid keys are: \
                    {', '.join(valid_keys.keys()):s}"

        # Check for valid parameter types
        for key, val in value.items():
            if not isinstance(val, valid_keys[key]):
                return f"The parameter '{key:s}' must be of type \
                    {valid_keys[key].__name__:s}."

        # Check for valid parameter values if value options are restricted
        if "method" in value.keys():
            method = value.get("method").upper()
            if method not in ["HF", "DFT"]:
                return f"The specified method key ('{method:s}') is \
                    not valid."
        if "scftype" in value.keys():
            opts = ["RHF", "UHF", "ROHF", "RKS", "UKS", "ROKS"]
            if value.get("scftype").upper() not in opts:
                return "The 'scftype' parameter must be one of 'RHF', 'UHF' or \
                    'ROHF' (or analogous 'rks', 'uks' or 'roks')."

        return None

    @classmethod
    def get_valid_mm_paramater_keys(cls, theory: str = "") -> dict[str:type]:
        """
        Return a tuple of valid parameter keys for the ChemShell MM interface.

        Returns
        -------
        validKeys : dict[str: type]
            A tuple of valid MM parameter keys for the ChemShell calculation.
        """
        if theory == "DL_POLY":
            valid_keys = {
                "theory": str,
                "input": str | tuple[str],
                "output": str,
                # general keys -> TODO: these are files which are not supported
                # by as AiiDA nodes if in a Dict object
                "berendsen": float,
                "delr": float,
                "densvar": int,
                "dl_field_charges": bool,
                "dl_field_types": bool,
                "equilibration": int,
                "ewald": float,
                "potential": bool,
                "print": int,
                "rcut": float,
                "rpad": float,
                "rvdw": float,
                "scale": int,
                "steps": int,
                "stack": int,
                "restart": str,
                "timestep": float,
            }
        elif theory == "GULP":
            valid_keys = {
                "theory": str,
                "input": str,
                "output": str,
                # general keys -> TODO: these are files which are not supported
                # by as AiiDA nodes if in a Dict object
                "molecule": bool,
                "conjugate": bool,
            }
        elif theory == "NAMD":
            valid_keys = {
                "theory": str,
                "input": str,
                "output": str,
                # general keys -> TODO: these are files which are not supported
                # by as AiiDA nodes if in a Dict object
                "binary": bool,
                "coor": str,  # TODO: file???
                "margin": float,
                "par": str | list[str],  # TODO: file???
                "pdb": str,  # TODO: file???
                "prefix": str,  # ?? how to handle this?
                "prefix_restart": str,  # ?? how to handle this?
                "psf": str,  # TODO: file???
                "psfgen_options": dict,
                "vel": str,  # TODO: file???
                "xsc": str,  # TODO: file???
                "xst": str,  # TODO: file???
                "cutoff": float,
                "exclude": str,
                "ff_dir": str,  # TODO: file??? -> folder???
                "freq_nonbonded": int,
                "freq_full_elect": int,
                "merge_cross": bool,
                "pairlist_dist": float,
                "scaling14": float,
                "switching": bool,
                "switch_dist": float,
                "nsteps_per_cycle": int,
                "seed": int,
                "constraints": str,  # TODO: file???
                "constraints_ref": str,  # TODO: file???
                "fixed_atoms": str,  # TODO: file???
                "fixed_atoms_forces": bool,
                "constant_area": bool,
                "flexible_cell": bool,
                "group_pressure": bool,
                "pme": bool,
                "pme_grid_sizes": list | tuple,
                "wrap_all": bool,
                "wrap_water": bool,
            }
        else:
            valid_keys = {"theory": str, "input": str, "output": str}
        return valid_keys

    @classmethod
    def validate_mm_parameters(cls, value: Dict | None, _) -> str | None:
        """
        Validate the MM interface parameter inputs to the ChemShell calculation.

        Parameters
        ----------
        value : Dict | None
            A dictionary of parameters for the ChemShell calculation.
            If None, no validation is performed.

        Returns
        -------
        str | None
            Returns None if the parameters are valid, otherwise returns an
            error message string.
        """
        theory = value.get("theory", "").upper()
        if theory not in ChemShellMMTheory.__members__:
            return f"The specified MM theory '{theory:s}' is not a \
                valid ChemShell MM interface within the AiiDA-ChemShell workflow."

        valid_keys = cls.get_valid_mm_paramater_keys(theory)
        invalid_keys = set(value.keys()).difference(set(valid_keys.keys()))
        if invalid_keys:
            # Checks for invalid parameter keys
            return f"The following parameter keys are invalid: \
                {', '.join(invalid_keys):s}. Valid keys are: \
                    {', '.join(valid_keys.keys()):s}"

        # Check for valid parameter types
        for key, val in value.items():
            if not isinstance(val, valid_keys[key]):
                return f"The parameter '{key:s}' must be of type \
                    {valid_keys[key].__name__:s}."
        return None

    @classmethod
    def get_qm_theory_key(cls, theory: ChemShellQMTheory) -> str:
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

        return ""

    @classmethod
    def get_mm_theory_key(cls, theory: ChemShellMMTheory) -> str:
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
        return ""

    def _build_process_label(self) -> str:
        """
        AiiDA Process label definition.

        Defines the process label to be associated with the created ProcessNode
        stored in the AiiDA database.

        Returns
        -------
        str
            The process label based on what inputs have been provided.
        """
        theory_key = ""
        if "qm_parameters" in self.inputs:
            if "mm_parameters" in self.inputs:
                theory_key = "_(QM/MM)"
            else:
                theory_key = "_(QM)"
        else:
            theory_key = "_(MM)"

        if "optimisation_parameters" in self.inputs:
            return "ChemShell_Geometry_Optimisation" + theory_key

        return "ChemShell_Single_Point_Calculation" + theory_key

    def chemsh_script_generator(self) -> str:
        """
        Generate the input script for a ChemShell calculation.

        Returns
        -------
        script : str
            A string containing the ChemShell input script for the calculation.
        """
        qm_theory = None
        mm_theory = None
        qmmm_chk = "qm_parameters" in self.inputs and "mm_parameters" in self.inputs

        script = "from chemsh import Fragment\n"
        if isinstance(self.inputs.structure, SinglefileData):
            fname = self.inputs.structure.filename
        else:
            fname = ChemShellCalculation.FILE_TMP_STRUCTURE
        script += f"structure = Fragment(coords='{fname:s}')\n"

        ## Setup Theory objects

        if "qm_parameters" in self.inputs:
            # Creates a quantum mechanics Theory object
            qm_theory = ChemShellQMTheory[
                self.inputs.qm_parameters.get("theory").upper()
            ]

            if qm_theory != ChemShellQMTheory.NONE:
                qm_theory_key = ChemShellCalculation.get_qm_theory_key(qm_theory)

                script += f"from chemsh import {qm_theory_key:s}\n"
                param_str = ""
                if "qm_parameters" in self.inputs:
                    for key in self.inputs.qm_parameters.keys():
                        if key == "theory":
                            continue
                        val = self.inputs.qm_parameters.get(key)
                        if isinstance(val, str):
                            param_str += ", " + key + "='" + val + "'"
                        else:
                            param_str += ", " + key + "=" + str(val)
                if qmmm_chk:
                    script += f"qmtheory = {qm_theory_key:s}(" + param_str[1:] + ")\n"
                else:
                    script += f"qmtheory = {qm_theory_key:s}(frag=structure"
                    script += param_str + ")\n"

        if "mm_parameters" in self.inputs:
            # Creates a molecular mechanics Theory object
            mm_theory = ChemShellMMTheory[
                self.inputs.mm_parameters.get("theory").upper()
            ]
            if mm_theory != ChemShellMMTheory.NONE:
                mm_theory_key = ChemShellCalculation.get_mm_theory_key(mm_theory)

                script += f"from chemsh import {mm_theory_key:s}\n"
                param_str = ""
                for key in self.inputs.mm_parameters.keys():
                    if key == "theory":
                        continue
                    val = self.inputs.mm_parameters.get(key)
                    if isinstance(val, str):
                        param_str += ", " + key + "='" + val + "'"
                    else:
                        param_str += ", " + key + "=" + str(val)
                if qmmm_chk:
                    script += f"mmtheory = {mm_theory_key:s}"
                    script += f"(ff='{self.inputs.force_field_file.filename:s}'"
                    script += f"{param_str:s})\n"
                else:
                    script += f"mmtheory = {mm_theory_key:s}(frag=structure, "
                    script += f"ff='{self.inputs.force_field_file.filename:s}'"
                    script += f"{param_str:s})\n"

        # If both QM and MM are specified, create a QM/MM interface object
        if qmmm_chk:
            theory_str = "qmmm"
            script += "from chemsh import QMMM\n"
            script += "qmmm = QMMM(frag=structure, qm=qmtheory, mm=mmtheory, "
            qm_region_str = str(self.inputs.qmmm_parameters.get("qm_region", []))
            script += f"qm_region={qm_region_str:s})\n"
        elif mm_theory:
            theory_str = "mmtheory"
        else:
            theory_str = "qmtheory"

        ## Setup Task objects

        if "optimisation_parameters" in self.inputs:
            # Run a geometry optimisation using DL_FIND
            script += "from chemsh import Opt\n"
            opt_str = f"job = Opt(theory={theory_str:s}"
            for key in self.inputs.optimisation_parameters.keys():
                if isinstance(self.inputs.optimisation_parameters.get(key), str):
                    opt_str += ", " + key + "='"
                    opt_str += self.inputs.optimisation_parameters.get(key) + "'"
                else:
                    opt_str += ", " + key + "="
                    opt_str += str(self.inputs.optimisation_parameters.get(key))
            script += opt_str + ")\n"
        else:
            # Perform a single point energy calculation (default calculation type)
            script += "from chemsh import SP\n"
            if "calculation_parameters" not in self.inputs:
                # Assign default values if none are given
                self.inputs.calculation_parameters = Dict(dict={})

            # Runs a QM single point energy calculation
            script += f"job = SP(theory={theory_str:s}, "
            grad_str = str(self.inputs.calculation_parameters.get("gradients", False))
            script += f"gradients={grad_str:s}, "
            hess_str = str(self.inputs.calculation_parameters.get("hessian", False))
            script += f"hessian={hess_str:s})\n"

        script += "job.run()\njob.result.save()\n"

        return script

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """
        Prepare the ChemShell calculation for submission.

        Params
        ------
        folder : Folder
            An `aiida.common.folders.Folder` specifying the temporary working
            directory for the calculation.

        Returns
        -------
        calcInfo : CalcInfo
            An `aiida.common.CalcInfo` instance.
        """
        # Create the ChemShell input script
        input_script = self.chemsh_script_generator()
        with folder.open(ChemShellCalculation.FILE_SCRIPT, "w") as f:
            f.write(input_script)

        # Define the AiiDA code parameters
        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = [
            ChemShellCalculation.FILE_SCRIPT,
        ]
        code_info.stdout_name = ChemShellCalculation.FILE_STDOUT

        # Setup the calculation information object
        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.retrieve_temporary_list = []
        calc_info.provenance_exclude_list = []
        calc_info.retrieve_list = [
            ChemShellCalculation.FILE_STDOUT,
            ChemShellCalculation.FILE_RESULTS,
        ]
        calc_info.local_copy_list = []

        if isinstance(self.inputs.structure, StructureData):
            with folder.open(ChemShellCalculation.FILE_TMP_STRUCTURE, "wb") as f:
                f.write(self.inputs.structure._prepare_xyz()[0])
        else:
            calc_info.local_copy_list.append(
                (
                    self.inputs.structure.uuid,
                    self.inputs.structure.filename,
                    self.inputs.structure.filename,
                ),
            )

        # If running with an MM theory a force field file is required and copied
        if "force_field_file" in self.inputs:
            calc_info.local_copy_list.append(
                (
                    self.inputs.force_field_file.uuid,
                    self.inputs.force_field_file.filename,
                    self.inputs.force_field_file.filename,
                )
            )

        # If performing a geometry optimisation retrieve the generated _dl_find.pun
        # file containing the optimised structure
        if "optimisation_parameters" in self.inputs:
            calc_info.retrieve_list.append(ChemShellCalculation.FILE_DLFIND)

        return calc_info
