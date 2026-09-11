"""Tests for carrying the pre-defined IsolatedAtoms workflows with aiida-chemshell."""

from unittest.mock import MagicMock

import pytest
from aiida.engine import run_get_node
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.orm import Dict

from aiida_chemshell.workflows.isolated_atoms import IsolatedAtomicEnergiesWorkChain


def test_default_input_tags_applied(chemsh_code, water_structure_object):
    """The ``structure`` input is tagged as it is never forwarded to a calculation."""
    inputs = {
        "structure": water_structure_object,
        "code": chemsh_code,
        "qm_parameters": Dict({"theory": "NWChem"}),
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, IsolatedAtomicEnergiesWorkChain, **inputs)
    process.apply_default_input_tags()

    structure = process.inputs.structure
    assert structure.label == "Input Chemical Structure"
    assert structure.description == (
        "The input structure to extract isolated atomic energies from."
    )

    # ``qm_parameters`` is forwarded to the sub-calculations, so it is tagged by
    # the base ChemShellCalculation auto-tagger rather than here.
    assert process.inputs.qm_parameters.label == ""


def test_metadata_options_forwarded_to_subcalculations(
    chemsh_code, water_structure_object
):
    """Metadata (e.g. MPI resources) is forwarded to every sub-calculation.

    The calculation ``metadata`` supplied to the WorkChain under its ``chemsh``
    namespace must reach each ChemShell single-point sub-calculation unchanged.
    This inspects the inputs the WorkChain would pass to each sub-calculation
    without executing them: ``submit`` is mocked so the expensive ChemShell
    calculations are skipped, keeping the test fast.
    """
    resources = {"num_machines": 1, "num_mpiprocs_per_machine": 2}
    inputs = {
        "structure": water_structure_object,
        "code": chemsh_code,
        "qm_parameters": Dict({"theory": "NWChem"}),
        "chemsh": {"metadata": {"options": {"resources": resources}}},
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, IsolatedAtomicEnergiesWorkChain, **inputs)

    # Skip the actual (slow) ChemShell calculations; only capture the submissions.
    process.submit = MagicMock(return_value=MagicMock())

    process.determine_unique_atom_types()
    process.atom_energies()

    # Water contains two unique atom types (O and H).
    assert process.submit.call_count == 2, "Incorrect number of sub processes created."
    for call in process.submit.call_args_list:
        options = call.kwargs["metadata"]["options"]
        assert dict(options["resources"]) == resources, (
            "Calculation metadata was not forwarded to the sub-calculation."
        )


@pytest.mark.xfail(reason="Will fail if NWChem not properly configured.")
def test_geometry_optimisation_workflow(chemsh_code, get_test_data_file):
    """Test a geometry optimisation workflow with vibrational analysis."""
    inputs = {
        "structure": get_test_data_file(),
        "code": chemsh_code,
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
