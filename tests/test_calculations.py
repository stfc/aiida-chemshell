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
    
    assert "Single_Point" in node.process_label 
    assert "QM" in node.process_label 

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
    
    assert "Single_Point" in node.process_label 
    assert "QM" in node.process_label 

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles         

    eref = -75.9468895721
    assert abs(results.get("energy") - eref) < 1e-8, \
        "Incorrect energy result for NWChem based SP calculation"

   


def test_SPCalculation_dlpoly(chemsh_code, get_test_data_file):
    code = chemsh_code
    builder = code.get_builder() 
    builder.structure = get_test_data_file("butanol.cjson")
    builder.MM_parameters = Dict({"theory": "DL_POLY"})
    builder.forceFieldFile = get_test_data_file("butanol.ff")

    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_SPCalculation_dlpoly`"
    
    assert "Single_Point" in node.process_label 
    assert "MM" in node.process_label 

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles  

    eref = 0.018194285557 
    assert (abs(results.get("energy") - eref)) < 1e-8, \
        "Incorrect energy result for DL_POLY based SP calculation."
    

def test_SPCalculation_qmmm(chemsh_code, get_test_data_file):
    code = chemsh_code 
    builder = code.get_builder() 
    builder.structure = get_test_data_file("h2o_dimer.cjson")
    builder.QM_parameters = Dict({"method": "HF"})
    builder.qm_theory = "NWChem"
    builder.forceFieldFile = get_test_data_file("h2o_dimer.ff")
    builder.QMMM_parameters = Dict({"qm_region": [0, 1, 2]})
    builder.MM_parameters = Dict({"theory": "DL_POLY"})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_SPCalculation_qmmm`"
    
    assert "Single_Point" in node.process_label 
    assert "QM/MM" in node.process_label 

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles

    eref = -75.603478594858

    assert (abs(results.get("energy") - eref)) < 1e-8, \
        "Incorrect energy result for QM/MM based SP calculation."

    
def test_OptCalculation_NWChem(chemsh_code, get_test_data_file):
    code = chemsh_code
    builder = code.get_builder() 
    builder.structure = get_test_data_file("water.cjson")
    builder.qm_theory = "NWChem"
    builder.QM_parameters = Dict({"method": "DFT", "basis": "3-21G"})
    builder.optimisation_parameters = Dict({})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_OptCalculation_NWChem`"
    
    assert "Geometry_Optimisation" in node.process_label 
    assert "QM" in node.process_label 
    
    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles 

    assert results.get("optimised_structure").filename == ChemShellCalculation.FILE_DLFIND

    eref = -75.951248996895
    assert abs(results.get("energy") - eref) < 1e-8, \
        "Incorrect energy result for NWChem based optimisation calculation."
    
def test_OptCalculation_dlpoly(chemsh_code, get_test_data_file):
    code = chemsh_code 
    builder = code.get_builder() 
    builder.structure = get_test_data_file("butanol.cjson")
    builder.MM_parameters = Dict({"theory": "DL_POLY"})
    builder.forceFieldFile = get_test_data_file("butanol.ff")
    builder.optimisation_parameters = Dict({})

    results, node = run.get_node(builder)

    assert node.is_finished_ok, \
        "CalcJob failed for `test_OptCalculation_dlpoly`"
    
    assert "Geometry_Optimisation" in node.process_label 
    assert "MM" in node.process_label 
    
    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles 
    assert ChemShellCalculation.FILE_DLFIND in ofiles 

    assert results.get("optimised_structure").filename == ChemShellCalculation.FILE_DLFIND

    eref = 0.017155540777
    assert abs(results.get("energy") - eref) < 1e-8, \
        "Incorrect energy result for DL_POLY based optimisation calculation."
    

def test_OptCalculation_qmmm(chemsh_code, get_test_data_file):
    code = chemsh_code 
    builder = code.get_builder() 
    builder.structure = get_test_data_file("h2o_dimer.cjson")
    builder.QM_parameters = Dict({"method": "HF"})
    builder.qm_theory = "NWChem"
    builder.forceFieldFile = get_test_data_file("h2o_dimer_gulp.ff")
    # There seems to be a bug when running this with DL_POLY???
    builder.MM_parameters = Dict({"theory": "GULP"})
    builder.QMMM_parameters = Dict({"qm_region": [0, 1, 2]})
    
    builder.optimisation_parameters = Dict({})
    # builder.metadata.options.withmpi = True
    # builder.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 4}

    results, node = run.get_node(builder)

    with open("tmp.txt", 'w') as f:
        f.write(results.get("retrieved").get_object_content(ChemShellCalculation.FILE_STDOUT))

    assert node.is_finished_ok, \
        "CalcJob failed for `test_OptCalculation_qmmm`"
    
    assert "Geometry_Optimisation" in node.process_label 
    assert "QM/MM" in node.process_label 

    ofiles = results.get("retrieved").list_object_names() 
    assert ChemShellCalculation.FILE_STDOUT in ofiles

    eref = -75.599224873736

    assert (abs(results.get("energy") - eref)) < 1e-8, \
        "Incorrect energy result for QM/MM based SP calculation."