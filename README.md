# aiida-chemshell

An [AiiDA](https://www.aiida.net) plugin for the [ChemShell](https://chemshell.org/) multiscale 
computational chemistry software package. 

## Installation 

This plugin should be installed alongside the AiiDA.core package, typically using [`pip`](https://pip.pypa.io/en/stable/) python package manager. 

```
cd aiida_chemshell 
pip install . 
```

## Requirements 

To use this plugin a configured AiiDA profile and computer configuration are required, see the 
[AiiDa documentation](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html) 
for instructions on how to install and configure AiiDA. 


## Setup 

To configure the ChemShell plugin an AiiDA code instance needs to be configured for the ChemShell executable. 
The following is an example of a basic YAML configuration file:

```yaml
label: chemsh
description: ChemShell 
computer: localhost 
filepath_executable: chemsh 
default_calc_job_plugin: chemshell
use_double_quotes: false 
with_mpi: false 
prepend_text: '' 
append_text: '' 
```

Write this to a file named `chemshell.yml` ensuring the value for `computer` matches the label of your computer configured in the previous 
step. The code can then be configured by running:

```bash
verdi code create core.code.installed --config chemshell.yml -n 
```

If successful this will have created a code with the label `chemsh` which can then be used to run ChemShell jobs within the AiiDA workflow. 

## Examples 

### QM Based Single Point Energy 

The following is a python script that will run a Quantum Mechanics based single point energy calculation using the NWChem ChemShell interface. 
The associated structure file can be found in the `tests/data/` folder. 

```python 
from aiida.engine import run 
from aiida.orm import load_code, SinglefileData, Dict
from aiida import load_profile 

load_profile("user_profile")  # This is not required if running in a verdi shell environment 

builder = load_code("chemsh").get_builder() 
builder.structure = SinglefileData(file="water.cjson")
builder.qm_theory = "nwchem" 
builder.QM_parameters = Dict({"method": "HF", "basis": "3-21G"})
builder.calculation_parameters = Dict({"gradients": False, "hessian": False})

results, node = run.get_node(builder)

print("Final Energy = ", results.get("energy"))
```

This can either be run as a script or as directly within a verdi shell python environment. 

### QM/MM Based Geometry Optimisation 

The following is a python script that will run a combined QM/MM based geometry optimisation using the NWChem and DL_POLY ChemShell interfaces.
The associated structure and force field files can be found in the `tests/data/` folder. 

```python 
from aiida.engine import run 
from aiida.orm import load_code, SinglefileData, Dict
from aiida import load_profile 

load_profile("user_profile")  # This is not required if running in a verdi shell environment 

builder = load_code("chemsh").get_builder()
builder.structure = SinglefileData(file="h2o_dimer.cjson")
builder.qm_theory = "nwchem" 
builder.QM_parameters = Dict({"method": "DFT", "basis": "6-31G"})
builder.MM_theory = "DL_POLY"
builder.MM_parameters = Dict({})
builder.forceFieldFile = SinglefileData(file="h2o_dimer.ff")
builder.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcyle": 100})

results, node = run.get_node(builder)

print("Final Energy = ", result.get("energy"))
```

This can either be run as a script or as directly within a verdi shell python environment. 