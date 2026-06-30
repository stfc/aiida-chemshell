"""Tests for the BatchProcessWorkChain."""

import pytest
from aiida.engine import run_get_node
from aiida.orm import Dict

from aiida_chemshell.workflows.batch_calculation import BatchProcessWorkChain


def test_batch_from_trajectorydata(chemsh_code, water_trajectory_object):
    """DFT based single point test."""
    inputs = {
        "code": chemsh_code(),
        "trajectory": water_trajectory_object,
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 3, "Incorrect number of sub processes created."

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10


@pytest.mark.xfail_aiida_2_8
def test_batch_from_structuredata(chemsh_code, water_trajectory_object):
    """DFT based single point test."""
    trajectory = water_trajectory_object

    inputs = {
        "code": chemsh_code(),
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
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 3, "Incorrect number of sub processes created."

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10


def test_batch_from_structuredata_and_trajectorydata(
    chemsh_code, water_trajectory_object, water_structure_object
):
    """DFT based single point test."""
    inputs = {
        "code": chemsh_code(),
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
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 4, "Incorrect number of sub processes created."

    final_energies = [
        -75.565560193461,
        -75.585287771819,
        -75.426355430539,
        -75.585287771819,
    ]

    for i, sub_node in enumerate(sub_nodes):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10
