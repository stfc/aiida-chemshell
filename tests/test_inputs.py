"""Tests for ChemShell input script generation based on various input parameters."""

from aiida_chemshell.calculations import ChemShellCalculation


def test_defaults(generate_calcjob):
    """Test the default for QM based single point chemshell script generation."""
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation)

    assert calc_info.retrieve_list == [
        ChemShellCalculation.FILE_STDOUT,
        ChemShellCalculation.FILE_RESULTS,
    ]
    code_info = calc_info.codes_info[0]
    assert code_info.cmdline_params == [
        ChemShellCalculation.FILE_SCRIPT,
    ]
    assert code_info.stdout_name == ChemShellCalculation.FILE_STDOUT

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "structure = Fragment(coords='water.cjson')\n" in script_txt
    assert "from chemsh import NWChem\n" in script_txt
    assert "qmtheory = NWChem(frag=structure)" in script_txt
    assert "from chemsh import SP\n" in script_txt
    assert "job = SP(theory=qmtheory, gradients=False, hessian=False)\n" in script_txt
    assert "job.run()\n" in script_txt
    assert "job.result.save()\n" in script_txt


def test_default_mm_sp(generate_calcjob, generate_inputs):
    """Test the defaults for MM based single point chemshell script generation."""
    inputs = generate_inputs(
        mm={"theory": "DL_POLY"},
        structure_fname="butanol.cjson",
        ff_fname="butanol.ff",
        sp={"gradients": True},
    )
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calc_info.local_copy_list) == 2

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "structure = Fragment(coords='butanol.cjson')\n" in script_txt
    assert "from chemsh import DL_POLY\n" in script_txt
    assert "mmtheory = DL_POLY(frag=structure, ff='butanol.ff')\n" in script_txt
    assert "from chemsh import SP\n" in script_txt
    assert "job = SP(theory=mmtheory, gradients=True, hessian=False)\n" in script_txt


def test_default_qmmm_sp(generate_calcjob, generate_inputs):
    """Test the defaults for qmmm based single point script generation."""
    inputs = generate_inputs(
        qm={"method": "HF"}, structure_fname="h2o_dimer.cjson", ff_fname="h2o_dimer.ff"
    )
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calc_info.local_copy_list) == 2

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "structure = Fragment(coords='h2o_dimer.cjson')\n" in script_txt
    assert "from chemsh import NWChem\n" in script_txt
    assert "qmtheory = NWChem( method='HF')\n" in script_txt
    assert "from chemsh import DL_POLY\n" in script_txt
    assert "mmtheory = DL_POLY(ff='h2o_dimer.ff')\n" in script_txt
    assert "from chemsh import QMMM\n" in script_txt
    assert "qmmm = QMMM(frag=structure, qm=qmtheory, " in script_txt
    assert "mm=mmtheory, qm_region=[0, 1, 2])\n" in script_txt
    assert "from chemsh import SP\n" in script_txt
    assert "job = SP(theory=qmmm, gradients=False, hessian=False)\n" in script_txt


def test_default_qm_opt(generate_calcjob, generate_inputs):
    """Test defaults for qm optimisation script generation."""
    inputs = generate_inputs(opt={"maxcycle": 100}, qm={"method": "dft", "charge": 0})
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "structure = Fragment(coords='water.cjson')\n" in script_txt
    assert "from chemsh import NWChem\n" in script_txt
    assert "qmtheory = NWChem(frag=structure, method='dft', charge=0)" in script_txt
    assert "from chemsh import Opt\n" in script_txt
    assert "job = Opt(theory=qmtheory, maxcycle=100)\n" in script_txt

    assert calc_info.retrieve_list == [
        ChemShellCalculation.FILE_STDOUT,
        ChemShellCalculation.FILE_RESULTS,
        ChemShellCalculation.FILE_DLFIND,
    ]


def test_expanded_mm_parameters(generate_calcjob, generate_inputs):
    """Test for expanded DL_POLY based MM optional parameters."""
    inputs = generate_inputs(
        mm={"theory": "DL_POLY", "timestep": 0.0001, "rcut": 10.0},
        structure_fname="butanol.cjson",
        ff_fname="butanol.ff",
        sp={"gradients": True},
    )
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calc_info.local_copy_list) == 2

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "structure = Fragment(coords='butanol.cjson')\n" in script_txt
    assert "from chemsh import DL_POLY\n" in script_txt
    assert "mmtheory = DL_POLY(frag=structure, ff='butanol.ff'," in script_txt
    assert "timestep=0.0001, rcut=10.0)\n" in script_txt
    assert "from chemsh import SP\n" in script_txt
    assert "job = SP(theory=mmtheory, gradients=True, hessian=False)\n" in script_txt


def test_structure_as_structuredata_object(
    generate_calcjob, generate_inputs, water_structure_object
):
    """Test taking a StructureData object as an input."""
    inputs = generate_inputs(structure_fname=water_structure_object)

    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    structure_file = tmp_pth / ChemShellCalculation.FILE_TMP_STRUCTURE
    assert structure_file.exists()
    chk_str = """3
Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" pbc="False False False"
O            0.0000000000       0.0000000000       0.0000000000
H           -0.7546064020       0.5900323550       0.0000000000
H            0.7546064020       0.5900323550       0.0000000000"""
    assert structure_file.read_text() == chk_str

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    tmp_structure_file = ChemShellCalculation.FILE_TMP_STRUCTURE
    assert f"structure = Fragment(coords='{tmp_structure_file:s}')\n" in script_txt
