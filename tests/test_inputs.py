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
    assert "qmtheory = NWChem(frag=structure, method='HF', basis='3-21G', charge=0, functional='B3LYP', mult=1, scftype='rhf')\n" in scriptText 
    assert "from chemsh import SP\n" in scriptText 
    assert "SP(theory=qmtheory, gradients=False, hessian=False).run()\n" in scriptText 
    

def test_default_MM_SP(generate_calcjob, generate_inputs):

    inputs = generate_inputs(mm={"theory": "dl_poly"}, structure_fname="butanol.cjson", ff_fname="butanol.ff", sp={"gradients": True})
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