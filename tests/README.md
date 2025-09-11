# AiiDA-ChemShell Plugin Test Suite 

This plugin supports testing via pytest. The tests use AiiDA's pytest fixtures which requires aiida-core>=2.6, 
this can be installed by installing the 'dev' optional package requirements,

```bash
pip install .[dev] 
```

To run a simple test suite which doesn't require access the the ChemShell software run, 

```bash
pytest tests/test_inputs.py  
```

this will run checks for all the input script generators but will not run any compute jobs or test the parsers.

To be able to run the full test suite you will need to 
ensure the chemsh.x executable is available in the environment PATH and that it has been configured to run 
NWChem, DL_POLY and GULP either directly or as external codes. 