"""Tests for carrying out pre-defined workflows with aiida-chemshell."""

from aiida.engine import run_get_node

from aiida_chemshell.workflows.optimisation import GeometryOptimisationWorkChain


def test_geometry_optimisation_workflow(chemsh_code, get_test_data_file):
    """Test a geometry optimisation workflow with vibrational analysis."""
    inputs = {
        "code": chemsh_code(),
        "structure": get_test_data_file(),
        "qm_parameters": {"theory": "PySCF", "method": "HF"},
        "basis_quality": "fast",
        "vibrational_analysis": True,
    }
    results, node = run_get_node(GeometryOptimisationWorkChain, **inputs)

    assert node.is_finished_ok, f"WorkChain failed with exit status {node.exit_status}"

    assert len(node.called) > 0, "WorkChain did not launch any subprocesses"

    assert node.called[0].inputs.qm_parameters.get(
        "basis", ""
    ) == GeometryOptimisationWorkChain.get_basis_set_label("fast")

    assert abs(results.get("final_energy") - -75.585959742867) < 1e-10, (
        "Incorrect final energy for geometry optimisation workflow."
    )

    assert "Temperature:     300.00 Kelvin" in str(results.get("vibrational_analysis"))
    assert " Mode     Eigenvalue Frequency" in str(results.get("vibrational_analysis"))
    assert "total vibrational energy correction" in str(
        results.get("vibrational_analysis")
    )

    assert results.get("vibrational_energies").get("Temperature / Kelvin") == 300.0
    assert results.get("vibrational_energies").get("ZPE / J/mol") == 57173.49993
    assert results.get("vibrational_energies").get("Enthalpy / J/mol") == 3.84524
    assert results.get("vibrational_energies").get("Entropy / J/mol/K") == 0.01430

    # print(results.get("vibrational_analysis"))
    # assert False
