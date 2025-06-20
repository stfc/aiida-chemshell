from aiida.engine import run 
from aiida.orm import load_code, SinglefileData, Dict 
from aiida import load_profile 

from aiida_chemshell.calculations import ChemShellCalculation
from aiida_chemshell.utils import * 

def test_SPCalculation_nwchem_hf(chemsh_code, get_test_data_file):

    code = chemsh_code
    builder = code.get_builder() 
    builder.structure = get_test_data_file() 
    builder.QM_parameters =  Dict({"method": "HF"})
    builder.qm_theory = "NWChem"
    builder.calculation_parameters = Dict({"gradients": True, "hessian": False})
     
    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_SPCalculation_nwchem_hf`"

    ofiles = results["retrieved"].list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles 

    eref = -75.585287777076
    assert abs(results.get("energy") - eref) < 1e-8, \
        "Incorrect energy result for NWChem based SP calculation"

    


def test_SPCalculation_nwchem_DFT(chemsh_code, get_test_data_file):

    code = chemsh_code
    builder = code.get_builder() 
    builder.structure = get_test_data_file() 
    builder.QM_parameters = Dict({"method": "DFT", "functional": "BLYP", "charge": 0, "scftype": "uks"})
    builder.qm_theory = "NWChem"
    
    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_SPCalculation_nwchem_DFT`"

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles         

    eref = -75.9468895721
    assert abs(results.get("energy") - eref) < 1e-8, \
        "Incorrect energy result for NWChem based SP calculation"

   


def test_SPCalculation_dlpoly(chemsh_code, get_test_data_file):
    code = chemsh_code
    builder = code.get_builder() 
    builder.structure = get_test_data_file("butanol.cjson")
    builder.MM_parameters = Dict({"theory": "dl_poly"})
    builder.forceFieldFile = get_test_data_file("butanol.ff")

    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_SPCalculation_dlpoly`"

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles  

    eref = 0.018194285557 
    assert (abs(results.get("energy") - eref)) < 1e-8, \
        "Incorrect energy result for DL_POLY based SP calculation."

    
