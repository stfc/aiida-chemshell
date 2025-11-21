"""Tests for performing calculation processes with aiida_chemshell."""

from aiida.engine import run
from aiida.orm import Dict
from numpy.linalg import norm

from aiida_chemshell.calculations.base import ChemShellCalculation


def test_sp_calculation_qm_hf(chemsh_code, get_test_data_file):
    """HF based single point test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = get_test_data_file()
    builder.qm_parameters = Dict({"theory": "PySCF", "method": "HF"})
    builder.calculation_parameters = Dict({"gradients": True, "hessian": True})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_SPCalculation_nwchem_hf`"

    assert "Single_Point" in node.process_label
    assert "QM" in node.process_label

    ofiles = results["retrieved"].list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    eref = -75.585287777076
    assert abs(results.get("energy") - eref) < 1e-8, (
        "Incorrect energy result for PySCF based SP calculation"
    )
    assert "Final SCF Energy" in results.get("energy").label, (
        "Incorrect energy output node label"
    )

    assert "gradients" in results, "Missing gradient array data output node."
    grad_data = results.get("gradients")

    assert "Energy Derivative Arrays" in grad_data.label, (
        f"Incorrect node label for gradient array output node. {grad_data.label}"
    )
    assert grad_data.description != "", "Missing gradient array node description"

    assert grad_data.get_arraynames() == ["gradients", "hessian"], (
        "Gradients and Hessian have not been correctly parsed for an SP calculation."
    )

    assert grad_data.get_shape("gradients") == (3, 3), (
        "Gradients have not been returned in the expected shape for a SP calculation."
    )

    # ref = 0.020629319737626634
    ref = 0.02063027856708385
    assert (norm(grad_data.get_array("gradients")) - ref) < 1e-8


def test_sp_calculation_qm_dft(chemsh_code, get_test_data_file, water_structure_object):
    """DFT based single point test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = water_structure_object
    builder.qm_parameters = Dict(
        {
            "theory": "PySCF",
            "method": "DFT",
            "functional": "BLYP",
            "charge": 0,
            # "scftype": "uks",
        }
    )

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_SPCalculation_nwchem_DFT`"

    assert "Single_Point" in node.process_label
    assert "QM" in node.process_label

    ofiles = results.get("retrieved").list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    # eref = -75.9468895533
    eref = -75.946889436563
    assert abs(results.get("energy") - eref) < 1e-8, (
        "Incorrect energy result for PySCF based SP calculation"
    )

    assert "gradients" not in results, (
        "Gradients have been returned for a SP calculation, \
            but they were not requested in the inputs."
    )


def test_sp_calculation_dlpoly(chemsh_code, get_test_data_file):
    """MM based single point test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = get_test_data_file("butanol.cjson")
    builder.mm_parameters = Dict({"theory": "DL_POLY"})
    builder.force_field_file = get_test_data_file("butanol.ff")

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_SPCalculation_dlpoly`"

    assert "Single_Point" in node.process_label
    assert "MM" in node.process_label

    ofiles = results.get("retrieved").list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    eref = 0.018194285557
    assert (abs(results.get("energy") - eref)) < 1e-8, (
        "Incorrect energy result for DL_POLY based SP calculation."
    )


def test_sp_calculation_qmmm(chemsh_code, get_test_data_file):
    """QM/MM based single point test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = get_test_data_file("h2o_dimer.cjson")
    builder.calculation_parameters = Dict({"gradients": True})
    builder.qm_parameters = Dict({"theory": "PySCF", "method": "HF"})
    builder.force_field_file = get_test_data_file("h2o_dimer.ff")
    builder.qmmm_parameters = Dict({"qm_region": [0, 1, 2]})
    builder.mm_parameters = Dict({"theory": "DL_POLY"})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_SPCalculation_qmmm`"

    assert "Single_Point" in node.process_label
    assert "QM/MM" in node.process_label

    ofiles = results.get("retrieved").list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    eref = -75.594381915214

    assert (abs(results.get("energy") - eref)) < 1e-8, (
        "Incorrect energy result for QM/MM based SP calculation."
    )


def test_opt_calculation_qm_dft(chemsh_code, get_test_data_file):
    """QM based geometry optimisation test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = get_test_data_file("water.cjson")
    builder.qm_parameters = Dict({"theory": "PySCF", "method": "DFT", "basis": "3-21G"})
    builder.optimisation_parameters = Dict({})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_OptCalculation_PySCF`"

    assert "Geometry_Optimisation" in node.process_label
    assert "QM" in node.process_label

    ofiles = results.get("retrieved").list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    assert (
        results.get("optimised_structure").filename == ChemShellCalculation.FILE_DLFIND
    )
    assert "Structure File" in results.get("optimised_structure").label
    assert "Optimised structure" in results.get("optimised_structure").description

    # eref = -75.951248996895
    eref = -75.951248407932
    assert abs(results.get("energy") - eref) < 1e-8, (
        "Incorrect energy result for PySCF based optimisation calculation."
    )


def test_opt_calculation_dlpoly(chemsh_code, get_test_data_file):
    """MM based geometry optimisation test."""
    code = chemsh_code()
    builder = code.get_builder()
    builder.structure = get_test_data_file("butanol.cjson")
    builder.mm_parameters = Dict({"theory": "DL_POLY"})
    builder.force_field_file = get_test_data_file("butanol.ff")
    builder.optimisation_parameters = Dict({})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, "CalcJob failed for `test_OptCalculation_dlpoly`"

    assert "Geometry_Optimisation" in node.process_label
    assert "MM" in node.process_label

    ofiles = results.get("retrieved").list_object_names()
    assert ChemShellCalculation.FILE_STDOUT in ofiles
    assert ChemShellCalculation.FILE_DLFIND in ofiles
    assert ChemShellCalculation.FILE_RESULTS in ofiles

    assert (
        results.get("optimised_structure").filename == ChemShellCalculation.FILE_DLFIND
    )

    eref = 0.017155540777
    assert abs(results.get("energy") - eref) < 1e-8, (
        "Incorrect energy result for DL_POLY based optimisation calculation."
    )


# def test_opt_calculation_qmmm(chemsh_code, get_test_data_file):
#     """QM/MM geometry optimisation test."""
#     code = chemsh_code()
#     builder = code.get_builder()
#     builder.structure = get_test_data_file("h2o_dimer.cjson")
#     builder.qm_parameters = Dict({"theory": "PySCF", "method": "HF"})
#     builder.force_field_file = get_test_data_file("h2o_dimer.ff")
#     # There seems to be a bug when running this with DL_POLY???
#     builder.mm_parameters = Dict({"theory": "DL_POLY"})
#     builder.qmmm_parameters = Dict({"qm_region": [0, 1, 2]})

#     builder.optimisation_parameters = Dict({})

#     results, node = run.get_node(builder)

#     assert node.is_finished_ok, "CalcJob failed for `test_OptCalculation_qmmm`"

#     assert "Geometry_Optimisation" in node.process_label
#     assert "QM/MM" in node.process_label

#     ofiles = results.get("retrieved").list_object_names()
#     assert ChemShellCalculation.FILE_STDOUT in ofiles
#     assert ChemShellCalculation.FILE_RESULTS in ofiles

#     # eref = -75.599224873736
#     eref = -75.59922485546

#     assert (abs(results.get("energy") - eref)) < 1e-8, (
#         "Incorrect energy result for QM/MM based SP calculation."
#     )
