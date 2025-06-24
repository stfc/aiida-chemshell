from aiida_chemshell.calculations import ChemShellCalculation 

def test_defaults(generate_calcjob):

    tmpPth, calcInfo = generate_calcjob(ChemShellCalculation)

    assert calcInfo.retrieve_list == [ChemShellCalculation.FILE_STDOUT,]
    codeInfo = calcInfo.codes_info[0] 
    assert codeInfo.cmdline_params == [ChemShellCalculation.FILE_SCRIPT,]
    assert codeInfo.stdout_name == ChemShellCalculation.FILE_STDOUT

    scriptFile = tmpPth / ChemShellCalculation.FILE_SCRIPT 
    assert scriptFile.exists()

    scriptText = scriptFile.read_text() 
    assert "from chemsh import Fragment\n" in scriptText  
    assert "structure = Fragment(coords='water.cjson')\n" in scriptText 
    assert "from chemsh import NWChem\n" in scriptText 
    assert "qmtheory = NWChem(frag=structure)" in scriptText
    assert "from chemsh import SP\n" in scriptText 
    assert "SP(theory=qmtheory, gradients=False, hessian=False).run()\n" in scriptText 
    

def test_default_MM_SP(generate_calcjob, generate_inputs):

    inputs = generate_inputs(mm={"theory": "DL_POLY"}, structure_fname="butanol.cjson", ff_fname="butanol.ff", sp={"gradients": True})
    tmpPth, calcInfo = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calcInfo.local_copy_list) == 2 

    scriptFile = tmpPth / ChemShellCalculation.FILE_SCRIPT 
    assert scriptFile.exists()

    scriptText = scriptFile.read_text() 
    assert "from chemsh import Fragment\n" in scriptText  
    assert "structure = Fragment(coords='butanol.cjson')\n" in scriptText 
    assert "from chemsh import DL_POLY\n" in scriptText 
    assert "mmtheory = DL_POLY(frag=structure, ff='butanol.ff')\n" in scriptText 
    assert "from chemsh import SP\n" in scriptText 
    assert "SP(theory=mmtheory, gradients=True, hessian=False).run()\n" in scriptText 


def test_default_QMMM_SP(generate_calcjob, generate_inputs):

    inputs = generate_inputs(qm={"method": "HF"}, structure_fname="h2o_dimer.cjson", ff_fname="h2o_dimer.ff")
    tmpPth, calcInfo = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calcInfo.local_copy_list) == 2 

    scritpFile = tmpPth / ChemShellCalculation.FILE_SCRIPT
    assert scritpFile.exists()

    scriptText = scritpFile.read_text()
    assert "from chemsh import Fragment\n" in scriptText
    assert "structure = Fragment(coords='h2o_dimer.cjson')\n" in scriptText
    assert "from chemsh import NWChem\n" in scriptText
    assert "qmtheory = NWChem( method='HF')\n" in scriptText
    assert "from chemsh import DL_POLY\n" in scriptText
    assert "mmtheory = DL_POLY(ff='h2o_dimer.ff')\n" in scriptText
    assert "from chemsh import QMMM\n" in scriptText 
    assert "qmmm = QMMM(frag=structure, qm=qmtheory, mm=mmtheory, qm_region=[0, 1, 2])\n" in scriptText
    assert "from chemsh import SP\n" in scriptText 
    assert "SP(theory=qmmm, gradients=False, hessian=False).run()\n" in scriptText 


def test_default_QM_Opt(generate_calcjob, generate_inputs):

    inputs = generate_inputs(opt={"maxcycle": 100}, qm={"method": "dft", "charge": 0})
    tmpPth, calcInfo = generate_calcjob(ChemShellCalculation, inputs)

    scriptFile = tmpPth / ChemShellCalculation.FILE_SCRIPT 
    assert scriptFile.exists()

    scriptText = scriptFile.read_text() 
    assert "from chemsh import Fragment\n" in scriptText  
    assert "structure = Fragment(coords='water.cjson')\n" in scriptText 
    assert "from chemsh import NWChem\n" in scriptText 
    assert "qmtheory = NWChem(frag=structure, method='dft', charge=0)" in scriptText
    assert "from chemsh import Opt\n" in scriptText 
    assert "Opt(theory=qmtheory, maxcycle=100).run()\n" in scriptText

    assert calcInfo.retrieve_list == [ChemShellCalculation.FILE_STDOUT, ChemShellCalculation.FILE_DLFIND]


def test_expanded_MM_parameters(generate_calcjob, generate_inputs):

    inputs = generate_inputs(mm={"theory": "DL_POLY", "timestep": 0.0001, "rcut": 10.0}, structure_fname="butanol.cjson", ff_fname="butanol.ff", sp={"gradients": True})
    tmpPth, calcInfo = generate_calcjob(ChemShellCalculation, inputs)

    assert len(calcInfo.local_copy_list) == 2 

    scriptFile = tmpPth / ChemShellCalculation.FILE_SCRIPT 
    assert scriptFile.exists()

    scriptText = scriptFile.read_text() 
    assert "from chemsh import Fragment\n" in scriptText  
    assert "structure = Fragment(coords='butanol.cjson')\n" in scriptText 
    assert "from chemsh import DL_POLY\n" in scriptText 
    assert "mmtheory = DL_POLY(frag=structure, ff='butanol.ff', timestep=0.0001, rcut=10.0)\n" in scriptText 
    assert "from chemsh import SP\n" in scriptText 
    assert "SP(theory=mmtheory, gradients=True, hessian=False).run()\n" in scriptText 