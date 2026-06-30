"""Tests for the BatchProcessWorkChain."""

from aiida import __version__ as aiida_core_version
from aiida.engine import run_get_node
from aiida.orm import Dict
from packaging.version import parse as parse_version

from aiida_chemshell.workflows.batch_calculation import BatchProcessWorkChain

AIIDA_LESS_THAN_2_8 = parse_version(aiida_core_version) < parse_version("2.8.0")


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
