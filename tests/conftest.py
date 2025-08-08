"""PyTest configurations."""

import pathlib

import pytest
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.orm import Dict, SinglefileData, StructureData

pytest_plugins = "aiida.tools.pytest_fixtures"


@pytest.fixture
def get_data_filepath() -> pathlib.Path:
    """Return the path to the tests data folder."""
    return pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture
def get_test_data_file(get_data_filepath):
    """Return a SinglefileData object containing the an input chemical structure."""

    def factory(fname: str = "water.cjson") -> SinglefileData:
        return SinglefileData(file=str(get_data_filepath / fname))

    return factory


@pytest.fixture
def chemsh_code(aiida_code_installed):
    """Return a ChemShell AiiDA code instance."""
    return aiida_code_installed(
        filepath_executable="chemsh", default_calc_job_plugin="chemshell"
    )


@pytest.fixture
def water_structure_object() -> StructureData:
    """Return a AiiDA StructureData object of a water molecule."""
    structure = StructureData()
    structure_str = """3

    O   0.000 0.000 0.000
    H  -0.754606402 0.590032355 0.0
    H   0.754606402 0.590032355 0.0
    """
    structure._parse_xyz(structure_str)
    return structure


@pytest.fixture
def generate_inputs(chemsh_code, get_test_data_file):
    """Return a dictionary of inputs for the ChemShellCalculation."""

    def factory(
        sp: dict | None = None,
        qm: dict | None = None,
        mm: dict | None = None,
        structure_fname: str | StructureData = "water.cjson",
        ff_fname: str | None = None,
        opt: dict | None = None,
    ):
        if isinstance(structure_fname, str):
            structure = get_test_data_file(structure_fname)
        else:
            structure = structure_fname
        inputs = {"code": chemsh_code, "structure": structure}
        if sp:
            inputs["calculation_parameters"] = Dict(sp)
        if qm:
            inputs["qm_parameters"] = Dict(qm)
            if "theory" not in qm:
                inputs["qm_parameters"]["theory"] = "NWChem"
        if mm:
            inputs["mm_parameters"] = Dict(mm)
            if "theory" not in mm:
                inputs["mm_parameters"]["theory"] = "DL_POLY"
        if not qm and not mm and not ff_fname:
            inputs["qm_parameters"] = Dict({"theory": "NWChem"})

        if ff_fname:
            inputs["force_field_file"] = get_test_data_file(ff_fname)
            if "mm_parameters" not in inputs:
                inputs["mm_parameters"] = Dict({"theory": "DL_POLY"})

        if opt:
            inputs["optimisation_parameters"] = Dict(opt)

        if "mm_parameters" in inputs and "qm_parameters" in inputs:
            inputs["qmmm_parameters"] = Dict({"qm_region": range(3)})

        return inputs

    return factory


@pytest.fixture
def generate_calcjob(tmp_path, generate_inputs):
    """Return an initialised aiida-chemshell CalcJob instance."""

    def factory(process_class: CalcJob, inputs=generate_inputs(), return_process=False):
        manager = get_manager()
        runner = manager.get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        if return_process:
            return process

        calc_info = process.prepare_for_submission(Folder(tmp_path))
        return tmp_path, calc_info

    return factory
