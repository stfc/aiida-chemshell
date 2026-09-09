"""Tests for the BatchProcessWorkChain."""

from unittest.mock import MagicMock

import pytest
from aiida.engine import run_get_node
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.orm import Dict

from aiida_chemshell.workflows.batch_calculation import BatchProcessWorkChain


def test_batch_forwards_metadata_options(chemsh_code, water_trajectory_object):
    """Resources set on the WorkChain are forwarded to every sub-calculation.

    This inspects the inputs the WorkChain would pass to each sub-calculation
    without executing them: ``submit`` is mocked so the expensive ChemShell
    calculations are skipped, keeping the test fast.
    """
    resources = {"num_machines": 1, "num_mpiprocs_per_machine": 2}
    inputs = {
        "code": chemsh_code,
        "trajectory": water_trajectory_object,
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
        "calc": {"metadata": {"options": {"resources": resources}}},
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, BatchProcessWorkChain, **inputs)

    # Skip the actual (slow) ChemShell calculations; only capture the submissions.
    process.submit = MagicMock(return_value=MagicMock())

    assert process.validate_inputs() is None, "WorkChain input validation failed."
    process.extract_structures_from_files()
    process.submit_jobs()

    assert process.submit.call_count == 3, "Incorrect number of sub processes created."
    for call in process.submit.call_args_list:
        options = call.kwargs["metadata"]["options"]
        assert dict(options["resources"]) == resources


def test_batch_from_trajectorydata(chemsh_code, water_trajectory_object):
    """DFT based single point test."""
    inputs = {
        "code": chemsh_code,
        "trajectory": water_trajectory_object,
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
        "combine_results": True,
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 4, "Incorrect number of sub processes created."

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes[:-1]):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10


@pytest.mark.xfail_aiida_2_8
def test_batch_from_structuredata(chemsh_code, water_trajectory_object):
    """DFT based single point test."""
    trajectory = water_trajectory_object

    inputs = {
        "code": chemsh_code,
        "structures": {
            "structure_1": trajectory.get_step_structure(0),
            "structure_2": trajectory.get_step_structure(1),
            "structure_3": trajectory.get_step_structure(2),
        },
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
        "combine_results": True,
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 4, "Incorrect number of sub processes created."

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes[:-1]):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10
    assert sub_nodes[-1].is_finished_ok, "Combination process failed"


def test_batch_from_structuredata_and_trajectorydata(
    chemsh_code, water_trajectory_object, water_structure_object
):
    """A batch mixing a trajectory and structures spawns one job per structure."""
    inputs = {
        "code": chemsh_code,
        "trajectory": water_trajectory_object,
        "structures": {
            "Structure_1": water_structure_object,
        },
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
    }

    runner = get_manager().get_runner()
    process = instantiate_process(runner, BatchProcessWorkChain, **inputs)

    process.submit = MagicMock(return_value=MagicMock())

    assert process.validate_inputs() is None, "WorkChain input validation failed."
    process.extract_structures_from_files()
    process.submit_jobs()

    # 3 trajectory frames + 1 StructureData input.
    assert process.submit.call_count == 4, "Incorrect number of sub processes created."


def test_batch_from_file(chemsh_code, get_test_data_file):
    """DFT based single point test."""
    structure_file = get_test_data_file("trajectory.xyz")
    inputs = {
        "code": chemsh_code,
        "structure_files": {
            structure_file.filename.strip(".xyz").replace(" ", "_"): structure_file
        },
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
        "calculation_parameters": Dict({"gradients": True}),
        "combine_results": True,
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 6, "Incorrect number of sub processes created."

    final_energies = [
        -75.585287789025,
        -75.585594607649,
        -75.585959615566,
        -272.88756364993,
    ]

    for i, sub_node in enumerate(sub_nodes[1:-1]):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10, (
            "Incorrect energy value for sub process."
        )

    with structure_file.open() as f:
        expected_lines = f.readlines()
    with results["combined_results"].open() as f:
        results_lines = f.readlines()

    assert len(expected_lines) == len(results_lines), (
        "Results file is of the incorrect length"
    )

    assert "Energy=-75.585287789025" in results_lines[1], (
        "Missing energy tag in first frame on results file."
    )

    assert "Properties=species:S:1:pos:R:3:force:R:3" in results_lines[1], (
        "Missing correct properties description tag in results file."
    )

    assert "0.01449338" in results_lines[3], (
        "Invalid force value for first frame in reulst file"
    )


def test_calcfunction_trajectory_directly():
    """Unit test for the combine_into_extended_xyz calcfunction."""
    from aiida.orm import Float, SinglefileData, StructureData

    from aiida_chemshell.workflows.batch_calculation import combine_into_extended_xyz

    struct1 = StructureData(cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    struct1.append_atom(position=(0.0, 0.0, 0.0), symbols="O")
    struct2 = StructureData(cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    struct2.append_atom(position=(1.0, 0.0, 0.0), symbols="H")

    energy1 = Float(-75.312)
    energy2 = Float(-5.456)

    kwargs = {
        "structure_0000": struct1,
        "energy_0000": energy1,
        "structure_0001": struct2,
        "energy_0001": energy2,
    }

    res = combine_into_extended_xyz(**kwargs)

    assert isinstance(res, SinglefileData)
    with res.open(res.filename, "r") as f:
        content = f.readlines()

    assert "1" in content[0]
    assert "Energy=-75.312" in content[1]
    assert "O" in content[2]
    assert "1" in content[3]
    assert "Energy=-5.456" in content[4]
    assert "H" in content[5]
