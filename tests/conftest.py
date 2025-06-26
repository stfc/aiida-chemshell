import pytest 
pytest_plugins = "aiida.tools.pytest_fixtures"

from aiida.engine import CalcJob 
from aiida.engine.utils import instantiate_process 
from aiida.manage.manager import get_manager 
from aiida.common.folders import Folder 
from aiida.orm import Dict, SinglefileData, StructureData

import pathlib 
import os 

@pytest.fixture 
def get_data_filepath() -> pathlib.Path:
    """ Returns the path to the tests data folder. """
    return pathlib.Path(__file__).resolve().parent / "data"  

@pytest.fixture 
def get_test_data_file(get_data_filepath):
    """ Returns a SinglefileData object containing the an input chemical structure. """

    def factory(fname: str = "water.cjson") -> SinglefileData:
        return SinglefileData(file=str(get_data_filepath / fname))

    return factory 

@pytest.fixture 
def chemsh_code(aiida_code_installed):
    return aiida_code_installed(
            filepath_executable="chemsh.x",
            default_calc_job_plugin="chemshell"
        )

@pytest.fixture 
def water_structure_object() -> StructureData:
    structure = StructureData()
    structureStr = """3
    
    O   0.000 0.000 0.000
    H  -0.754606402 0.590032355 0.0
    H   0.754606402 0.590032355 0.0
    """
    structure._parse_xyz(structureStr)
    return structure 

@pytest.fixture 
def generate_inputs(chemsh_code, get_test_data_file):
    """ Returns a dictionary of inputs for the ChemShellCalculation. """
    
    def factory(sp: dict | None = None, qm: dict | None = None, mm: dict | None = None, structure_fname: str | StructureData = 'water.cjson', ff_fname: str | None = None, opt: dict | None = None):
        if isinstance(structure_fname, str):
            structure = get_test_data_file(structure_fname)
        else:
            structure = structure_fname
        inputs = {
            "code": chemsh_code,
            "structure": structure 
        }
        if sp:
            inputs["calculation_parameters"] = Dict(sp)
        if qm:
            inputs["QM_parameters"] = Dict(qm)
            if "theory" not in qm:
                inputs["QM_parameters"]["theory"] = "NWChem"
        if mm:
            inputs["MM_parameters"] = Dict(mm)
            if "theory" not in mm:
                inputs["MM_parameters"]["theory"] = "DL_POLY"
        if not qm and not mm and not ff_fname:
            inputs["QM_parameters"] = Dict({"theory": "NWChem"})

        if ff_fname:
            inputs["forceFieldFile"] = get_test_data_file(ff_fname)
            if "MM_parameters" not in inputs:
                inputs["MM_parameters"] = Dict({"theory": "DL_POLY"})

        if opt:
            inputs["optimisation_parameters"] = Dict(opt)

        if "MM_parameters" in inputs and "QM_parameters" in inputs:
            inputs["QMMM_parameters"] = Dict({"qm_region": range(3)})

        return inputs 
    
    return factory 

@pytest.fixture 
def generate_calcjob(tmp_path, generate_inputs):

    def factory(process_class: CalcJob, inputs=generate_inputs(), return_process=False):
        manager = get_manager()
        runner = manager.get_runner() 
        process = instantiate_process(runner, process_class, **inputs)

        if return_process:
            return process 
        
        calcInfo = process.prepare_for_submission(Folder(tmp_path))
        return tmp_path, calcInfo
    
    return factory 
