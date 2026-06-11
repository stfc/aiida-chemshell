"""Tests for carrying out pre-defined workflows with aiida-chemshell."""

from aiida.engine import run_get_node

from aiida_chemshell.workflows.optimisation import GeometryOptimisationWorkChain


def test_geometry_optimisation_workflow(chemsh_code, get_test_data_file):
    """Test a geometry optimisation workflow with vibrational analysis."""
    inputs = {
        "chemsh": {
            "code": chemsh_code(),
            "structure": get_test_data_file(),
            "qm_parameters": {"theory": "PySCF", "method": "HF"},
        },
        "basis_quality": "fast",
        "vibrational_analysis": True,
    }
    results, node = run_get_node(GeometryOptimisationWorkChain, **inputs)

    assert node.is_finished_ok, f"WorkChain failed with exit status {node.exit_status}"

    assert len(node.called) > 0, "WorkChain did not launch any subprocesses"

    assert node.called[0].inputs.qm_parameters.get(
        "basis", ""
    ) == GeometryOptimisationWorkChain.get_basis_set_label("fast")

    assert abs(results.get("final_energy") - -75.585959742867) < 1e-9, (
        "Incorrect final energy for geometry optimisation workflow."
    )

    assert results.get("vibrational_energies").get("Temperature / Kelvin") == 300.0
    assert results.get("vibrational_energies").get("ZPE / J/mol") == 57173.49993
    assert results.get("vibrational_energies").get("Enthalpy / J/mol") == 3.84524
    assert results.get("vibrational_energies").get("Entropy / J/mol/K") == 0.01430

    assert results.get("vibrational_modes").get_shape("Modes") == (3, 5)

    modes = results.get("vibrational_modes").get_array("Modes")
    assert (modes[0][0] - 1799.584) < 1e-10, "Incorrect frequency reported for mode 1"
    assert (modes[2][2] - 0.0089900284) < 1e-10, "Incorrect ZPE reported for mode 3"


def test_optimisation_workflow_mlip_training(
    chemsh_code, get_test_data_file, janus_code
):
    """Test a geometry optimisation workflow with vibrational analysis."""
    from aiida_mlip.helpers.help_load import load_model

    inputs = {
        "chemsh": {
            "code": chemsh_code(),
            "structure": get_test_data_file("butanol.cjson"),
            "qm_parameters": {
                "theory": "PySCF",
                "method": "hf",
                "functional": "blyp",
            },
        },
        "mlip_model": load_model(None, "mace_mp"),
        "mlip_code": janus_code,
        "basis_quality": "fast",
        "vibrational_analysis": False,
    }
    results, node = run_get_node(GeometryOptimisationWorkChain, **inputs)

    assert node.is_finished_ok, f"WorkChain failed with exit status {node.exit_status}"

    assert len(node.called) > 0, "WorkChain did not launch any subprocesses"

    # assert node.called[0].inputs.qm_parameters.get(
    #    "basis", ""
    # ) == GeometryOptimisationWorkChain.get_basis_set_label("fast")
    #
    # assert abs(results.get("final_energy") - -75.585959742867) < 1e-9, (
    #    "Incorrect final energy for geometry optimisation workflow."
    # )

    subs = node.called
    for sub in subs:
        # if "Generate MLIP training" in sub.label:
        #     print(sub.outputs.training_input.content)
        if "MLIP Fine-Tuning" in sub.label:
            print(sub.outputs.retrieved.list_object_names())
            print("ERROR")
            print(sub.outputs.retrieved.get_object_content("_scheduler-stderr.txt"))
            print("OUTPUT")
            print(sub.outputs.retrieved.get_object_content("_scheduler-stdout.txt"))
            print("AIIDA OUTPUT")
            print(sub.outputs.retrieved.get_object_content("aiida-stdout.txt"))
            print("results")
            print(sub.outputs.retrieved.list_object_names(path="results"))
            print("LOG")
            # print(sub.outputs.retrieved.get_object_content("logs/test_run-123.log"))

        assert sub.is_finished_ok, f"Node '{sub.label}' failed to finish correctly."
    # assert False
