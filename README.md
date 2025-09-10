[![Release](https://img.shields.io/github/v/release/stfc/aiida-chemshell)](https://github.com/stfc/aiida-chemshell/releases)
[![PyPI](https://img.shields.io/pypi/v/aiida-chemshell)](https://pypi.org/project/aiida-chemshell/)

[![Docs status](https://github.com/stfc/aiida-chemshell/actions/workflows/ci-docs.yml/badge.svg?branch=main)](https://stfc.github.io/aiida-chemshell/)
[![Pipeline Status](https://github.com/stfc/aiida-chemshell/actions/workflows/ci-testing.yml/badge.svg?branch=main)](https://github.com/stfc/aiida-chemshell/actions)
[![Coverage Status]( https://coveralls.io/repos/github/stfc/aiida-chemshell/badge.svg?branch=main)](https://coveralls.io/github/stfc/aiida-chemshell?branch=main)

[![DOI](https://zenodo.org/badge/1050396974.svg)](https://doi.org/10.5281/zenodo.17055004)

# aiida-chemshell

An [AiiDA](https://www.aiida.net) plugin for the [ChemShell](https://chemshell.org/) multiscale 
computational chemistry software package. 

## Installation 

This plugin is available through PyPI and should be installed using [`pip`](https://pip.pypa.io/en/stable/) python package manager. 

```
pip install aiida-chemshell 
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
builder.structure = SinglefileData(file="absolute/path/to/water.cjson")
builder.qm_parameters = Dict({"theory": "NWChem", "method": "HF", "basis": "3-21G"})
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
builder.structure = SinglefileData(file="absolute/path/to/h2o_dimer.cjson")
builder.qm_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})
builder.mm_parameters = Dict({"theory": "DL_POLY"})
builder.force_field_file = SinglefileData(file="absolute/path/to/h2o_dimer.ff")
builder.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcyle": 100})

results, node = run.get_node(builder)

print("Final Energy = ", result.get("energy"))
```

This can either be run as a script or as directly within a verdi shell python environment. 