"""Tests for the BatchProcessWorkChain."""

import pytest
from aiida.engine import run_get_node
from aiida.orm import Dict

from aiida_chemshell.workflows.batch_calculation import BatchProcessWorkChain


def test_batch_exposes_calculation_metadata_options():
    """The calculation metadata options must be exposed under the 'calc' namespace.

    This allows users to specify the resources (e.g. number of MPI processes) used
    by the underlying series of ChemShell calculations, without clashing with the
    WorkChain's own reserved 'metadata' namespace.
    """
    spec = BatchProcessWorkChain.spec()
    assert "options" not in spec.inputs["metadata"], (
        "Calculation options must not be merged into the WorkChain metadata."
    )
    options = spec.inputs["calc"]["metadata"]["options"]
    assert "resources" in options, "The calculation metadata options are not exposed."


def test_batch_forwards_metadata_options(chemsh_code, water_trajectory_object):
    """Resources set on the WorkChain are forwarded to every sub-calculation."""
    resources = {"num_machines": 1, "num_mpiprocs_per_machine": 2}
    inputs = {
        "code": chemsh_code(),
        "trajectory": water_trajectory_object,
        "qm_parameters": Dict(
            {
                "theory": "PySCF",
                "method": "hf",
            }
        ),
        "calc": {"metadata": {"options": {"resources": resources}}},
    }
    results, node = run_get_node(BatchProcessWorkChain, **inputs)

    assert node.is_finished_ok, "WorkChain Failed"

    sub_nodes = node.called
    assert len(sub_nodes) == 3, "Incorrect number of sub processes created."
    for sub_node in sub_nodes:
        assert sub_node.get_option("resources") == resources


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


def test_batch_from_file(chemsh_code, get_test_data_file):
    """DFT based single point test."""
    structure_file = get_test_data_file("trajectory.xyz")
    inputs = {
        "code": chemsh_code(),
        "structure_files": {
            structure_file.filename.strip(".xyz").replace(" ", "_"): structure_file
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
    assert len(sub_nodes) == 5, "Incorrect number of sub processes created."

    final_energies = [
        -75.585287789025,
        -75.585594607649,
        -75.585959615566,
        -272.88756364993,
    ]

    for i, sub_node in enumerate(sub_nodes[1:]):
        assert sub_node.is_finished_ok, "Sub Process Failed"
        assert abs(sub_node.outputs.energy - final_energies[i]) < 1e-10
