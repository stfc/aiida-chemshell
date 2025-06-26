Example Calculations 
====================

QM Based Single Point Energy 
----------------------------

The following is a python script that will run a Quantum Mechanics (QM) based single point energy calculation 
using the NWChem ChemShell interface. 

.. code-block:: python 

    from aiida.engine import run 
    from aiida.orm import load_code, SinglefileData, Dict
    from aiida import load_profile 

    load_profile("user_profile")  # This is not required if running in a verdi shell environment 

    builder = load_code("chemsh").get_builder() 
    builder.structure = SinglefileData(file="water.cjson")
    builder.QM_parameters = Dict({"theory": "NWChem", "method": "HF", "basis": "3-21G"})
    builder.calculation_parameters = Dict({"gradients": False, "hessian": False})

    results, node = run.get_node(builder)

    print("Final Energy = ", results.get("energy"))

This can either be run as a script or as directly within a verdi shell python environment. 

QM/MM Based Geometry Optimisation
--------------------------------- 

The following is a python script that will run a combined QM/MM based geometry optimisation using the NWChem and DL_POLY ChemShell interfaces.
The associated structure and force field files can be found in the `tests/data/` folder. 

.. code-block:: python 
    
    from aiida.engine import run 
    from aiida.orm import load_code, SinglefileData, Dict
    from aiida import load_profile 

    load_profile("user_profile")  # This is not required if running in a verdi shell environment 

    builder = load_code("chemsh").get_builder()
    builder.structure = SinglefileData(file="h2o_dimer.cjson")
    builder.QM_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})
    builder.MM_parameters = Dict({"theory": "DL_POLY"})
    builder.forceFieldFile = SinglefileData(file="h2o_dimer.ff")
    builder.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcyle": 100})

    results, node = run.get_node(builder)

    print("Final Energy = ", result.get("energy"))

This can either be run as a script or as directly within a verdi shell python environment. 