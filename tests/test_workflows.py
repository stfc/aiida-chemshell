"""Tests for carrying out pre-defined workflows with aiida-chemshell."""

from aiida.engine import run_get_node
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_chemshell.workflows.optimisation import GeometryOptimisationWorkChain


def test_default_input_tags_applied(chemsh_code, get_test_data_file):
    """WorkChain specific inputs are tagged with default labels/descriptions."""
    inputs = {
        "chemsh": {
            "code": chemsh_code,
            "structure": get_test_data_file(),
            "qm_parameters": {"theory": "NWChem"},
        },
        "vibrational_analysis": True,
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, GeometryOptimisationWorkChain, **inputs)
    process.apply_default_input_tags()

    vib = process.inputs.vibrational_analysis
    assert vib.label == "Vibrational Analysis Flag"
    assert vib.description == (
        "Whether to calculate the vibrational modes of the optimised structure."
    )

    # The exposed ``chemsh`` structure is left untagged here; it is tagged by the
    # base ChemShellCalculation auto-tagger when forwarded to the sub-calculation.
    assert process.inputs.chemsh.structure.label == ""


def test_existing_input_tags_preserved(chemsh_code, get_test_data_file):
    """Pre-existing labels/descriptions on WorkChain inputs are not changed."""
    from aiida.orm import Bool

    vibrational_analysis = Bool(True)
    vibrational_analysis.label = "My custom flag"
    vibrational_analysis.description = "A custom description."

    inputs = {
        "chemsh": {
            "code": chemsh_code,
            "structure": get_test_data_file(),
            "qm_parameters": {"theory": "NWChem"},
        },
        "vibrational_analysis": vibrational_analysis,
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, GeometryOptimisationWorkChain, **inputs)
    process.apply_default_input_tags()

    vib = process.inputs.vibrational_analysis
    assert vib.label == "My custom flag"
    assert vib.description == "A custom description."


def test_geometry_optimisation_workflow(chemsh_code, get_test_data_file):
    """Test a geometry optimisation workflow with vibrational analysis."""
    inputs = {
        "chemsh": {
            "code": chemsh_code,
            "structure": get_test_data_file(),
            "qm_parameters": {"theory": "PySCF", "method": "HF", "basis": "3-21G"},
        },
        "vibrational_analysis": True,
    }
    results, node = run_get_node(GeometryOptimisationWorkChain, **inputs)

    assert node.is_finished_ok, f"WorkChain failed with exit status {node.exit_status}"

    assert len(node.called) > 0, "WorkChain did not launch any subprocesses"

    assert abs(results.get("final_energy") - -75.585959742867) < 1e-9, (
        "Incorrect final energy for geometry optimisation workflow."
    )

    # The optimised structure is now returned as a StructureData node, so the
    # geometry fed to the vibrational analysis step is round-tripped through
    # Bohr->Angstrom->Bohr. This shifts the thermochemistry by a physically
    # negligible amount relative to passing the raw '.cjson' file through.
    assert results.get("vibrational_energies").get("Temperature / Kelvin") == 300.0
    assert results.get("vibrational_energies").get("ZPE / J/mol") == 57173.46025
    assert results.get("vibrational_energies").get("Enthalpy / J/mol") == 3.84523
    assert results.get("vibrational_energies").get("Entropy / J/mol/K") == 0.01430

    assert results.get("vibrational_modes").get_shape("Modes") == (3, 5)

    modes = results.get("vibrational_modes").get_array("Modes")
    assert (modes[0][0] - 1799.584) < 1e-10, "Incorrect frequency reported for mode 1"
    assert (modes[2][2] - 0.0089900284) < 1e-10, "Incorrect ZPE reported for mode 3"
