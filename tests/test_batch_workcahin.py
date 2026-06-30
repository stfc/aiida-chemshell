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

    sub_nodes = node.called

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes):
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

    sub_nodes = node.called

    final_energies = [-75.565560193461, -75.585287771819, -75.426355430539]

    for i, sub_node in enumerate(sub_nodes):
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10
