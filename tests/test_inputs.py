"""Tests for ChemShell input script generation based on various input parameters."""

from aiida.orm import StructureData

from aiida_chemshell.calculations.base import ChemShellCalculation


def test_defaults(generate_calcjob):
    """Test the default for QM based single point chemshell script generation."""
    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation)

    assert calc_info.retrieve_list == [
        ChemShellCalculation.FILE_STDOUT,
    ]
    assert ChemShellCalculation.FILE_RESULTS in calc_info.retrieve_temporary_list
    code_info = calc_info.codes_info[0]
    assert ChemShellCalculation.FILE_SCRIPT in code_info.cmdline_params
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
    ]
    assert ChemShellCalculation.FILE_DLFIND in calc_info.retrieve_temporary_list
    assert ChemShellCalculation.FILE_RESULTS in calc_info.retrieve_temporary_list


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
    assert ChemShellCalculation.FILE_TMP_STRUCTURE in script_txt


def test_cjson_to_structure_data(get_test_data_file):
    """Test parsing a ChemShell '.cjson' file into a StructureData node."""
    from aiida_chemshell.units import UnitsConverter
    from aiida_chemshell.utils import chemsh_cjson_to_structure_data

    cjson = get_test_data_file("water.cjson")
    structure = chemsh_cjson_to_structure_data(cjson.get_content())

    assert structure.get_formula() == "H2O"
    assert len(structure.sites) == 3

    positions = [site.position for site in structure.sites]
    # Coordinates in the file are in atomic units and should be converted to
    # Angstrom.
    assert positions[0] == (0.0, 0.0, 0.0)
    assert abs(positions[1][0] - UnitsConverter.bohr_to_angstrom(-1.426)) < 1e-9
    assert abs(positions[1][1] - UnitsConverter.bohr_to_angstrom(1.115)) < 1e-9


def test_cjson_to_structure_data_symbol_normalisation():
    """Test that uppercase element symbols are normalised for AiiDA."""
    from aiida_chemshell.utils import chemsh_cjson_to_structure_data

    data = (
        '{"atoms": {"elements": {"symbol": ["NA", "CL"]}, '
        '"coords": {"unit": "angstrom", "3d": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}}}'
    )
    structure = chemsh_cjson_to_structure_data(data)

    assert {kind.symbols[0] for kind in structure.kinds} == {"Na", "Cl"}


def test_default_input_tags_applied(generate_calcjob, generate_inputs):
    """Test default labels/descriptions are applied to untagged input nodes."""
    inputs = generate_inputs(qm={"method": "HF"})
    generate_calcjob(ChemShellCalculation, inputs)

    structure = inputs["structure"]
    assert structure.label == "Input Chemical Structure"
    assert structure.description == (
        "The input structure for the ChemShell calculation."
    )

    qm_parameters = inputs["qm_parameters"]
    assert qm_parameters.label == "ChemShell QM Parameters"
    assert qm_parameters.description == (
        "Parameters for the ChemShell QM Theory object."
    )


def test_existing_input_tags_preserved(generate_calcjob, generate_inputs):
    """Test that pre-existing labels/descriptions on input nodes are not changed."""
    inputs = generate_inputs(qm={"method": "HF"})
    inputs["structure"].label = "My custom structure"
    inputs["structure"].description = "A custom description."

    generate_calcjob(ChemShellCalculation, inputs)

    structure = inputs["structure"]
    assert structure.label == "My custom structure"
    assert structure.description == "A custom description."


def test_atom_as_structuredata_object(
    generate_calcjob, generate_inputs, water_structure_object
):
    """Test taking a StructureData object as an input."""
    structure = StructureData()
    structure.pbc = [False, False, False]
    structure.append_atom(symbols="O", position=[0.0, 0.0, 0.0])
    inputs = generate_inputs(structure_fname=structure)

    tmp_pth, calc_info = generate_calcjob(ChemShellCalculation, inputs)

    structure_file = tmp_pth / ChemShellCalculation.FILE_TMP_STRUCTURE
    assert structure_file.exists()
    chk_str = """1
Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" pbc="False False False"
O            0.0000000000       0.0000000000       0.0000000000"""
    assert structure_file.read_text() == chk_str

    script_file = tmp_pth / ChemShellCalculation.FILE_SCRIPT
    assert script_file.exists()

    script_txt = script_file.read_text()
    assert "from chemsh import Fragment\n" in script_txt
    assert "Fragment(coords=[[0.0, 0.0, 0.0]]," in script_txt
