"""Tests for carrying the pre-defined IsolatedAtoms workflows with aiida-chemshell."""

import pytest
from aiida.engine import run_get_node

from aiida_chemshell.workflows.isolated_atoms import IsolatedAtomicEnergiesWorkChain


@pytest.mark.xfail(reason="Will fail if NWChem not properly configured.")
def test_geometry_optimisation_workflow(chemsh_code, get_test_data_file):
    """Test a geometry optimisation workflow with vibrational analysis."""
    inputs = {
        "structure": get_test_data_file(),
        "code": chemsh_code(),
        "qm_parameters": {"theory": "NWChem", "method": "HF"},
    }
    results, node = run_get_node(IsolatedAtomicEnergiesWorkChain, **inputs)

    # print(results)
    assert node.is_finished_ok, (
        "WorkChain node failed for IsolatedAtomicEnergiesWorkChain"
    )
    assert len(list(results.get("atom_energies").get_dict().keys())) == 2
    h_energy_ref = -0.496198609381
    assert abs(results.get("atom_energies").get_dict()["H"] - h_energy_ref) < 1e-10
    o_energy_ref = -74.267449889229
    assert abs(results.get("atom_energies").get_dict()["O"] - o_energy_ref) < 1e-10
