"""PyTest configurations."""

import os
import pathlib
import shutil
import subprocess

import numpy
import pytest
from aiida import __version__ as aiida_core_version
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.orm import (
    ContainerizedCode,
    Dict,
    InstalledCode,
    SinglefileData,
    StructureData,
    TrajectoryData,
)
from packaging.version import parse as parse_version

pytest_plugins = "aiida.tools.pytest_fixtures"


def _docker_available() -> bool:
    """Return whether a usable Docker daemon is reachable on this host.

    This is used to gate the container-backed tests so that the Docker
    dependency only applies to the tests that actually launch a containerised
    ChemShell code, rather than to the test suite as a whole.
    """
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


DOCKER_AVAILABLE = _docker_available()


def pytest_configure(config):
    """Dynamically register a custom xfail condition based on AiiDA version."""
    config.addinivalue_line(
        "markers",
        "xfail_aiida_2_8: mark test as expected failure if aiida-core is < 2.8",
    )


def pytest_runtest_setup(item):
    """Evaluate the custom marker before running the test."""
    marker = item.get_closest_marker("xfail_aiida_2_8")
    if marker:
        if parse_version(aiida_core_version) < parse_version("2.8.0"):
            # Dynamically apply the xfail to this specific test instance
            item.add_marker(
                pytest.mark.xfail(
                    reason="This test requires features present in aiida-core >= 2.8",
                    raises=ValueError,
                )
            )


@pytest.fixture
def get_data_filepath() -> pathlib.Path:
    """Return the path to the tests data folder."""
    return pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture
def get_test_data_file(get_data_filepath):
    """Return a SinglefileData object containing the an input chemical structure."""

    def factory(fname: str = "water.cjson") -> SinglefileData:
        return SinglefileData(file=str(get_data_filepath / fname))

    return factory


@pytest.fixture
def chemsh_code(
    aiida_code, aiida_code_installed, aiida_computer
) -> ContainerizedCode | InstalledCode:
    """Return  a ChemShell AiiDA code instance.

    By default this returns a :class:`~aiida.orm.ContainerizedCode` that launches
    ChemShell inside a Docker container. The image defaults to
    ``ghcr.io/stfc/aiidalab-chemshell/chemsh:latest`` (override with
    ``CHEMSHELL_IMAGE`` and ``CHEMSHELL_CONTAINER_BIN``).

    If no Docker daemon is available, it falls back to a host-installed ``chemsh``
    executable (``CHEMSHELL_BIN``, default ``chemsh``).

    The code instance is given a fixed label such that it is created once and
    then reused by all tests within the temporary testing profile.
    """
    if not DOCKER_AVAILABLE:
        # Fall back to a host-installed ChemShell executable.
        return aiida_code_installed(
            filepath_executable=os.environ.get("CHEMSHELL_BIN", "chemsh"),
            default_calc_job_plugin="chemshell",
            prepend_text=os.environ.get("CHEMSHELL_PREPEND_TEXT", ""),
            append_text=os.environ.get("CHEMSHELL_APPEND_TEXT", ""),
        )

    image_name = os.environ.get(
        "CHEMSHELL_IMAGE", "ghcr.io/stfc/aiidalab-chemshell/chemsh:latest"
    )
    filepath_executable = os.environ.get(
        "CHEMSHELL_CONTAINER_BIN", "/opt/chemsh-py/bin/intel/chemsh"
    )

    computer = aiida_computer(label="localhost-docker", transport_type="core.local")
    computer.set_use_double_quotes(True)
    computer.configure()

    # Clear any potential entrypoints so AiiDA can handle the executable
    # in the container. Run the container as the host user so that files
    # written into the bind-mounted working directory are owned by that user
    # and can be cleaned up by pytest (otherwise they are root-owned and the
    # temp-directory cleanup fails with noisy warnings).
    #
    # The host uid may not exist in the image's /etc/passwd (e.g. the GitHub
    # Actions runner is uid 1001, whereas the image only defines uid 1000), in
    # which case ``$HOME`` defaults to an unwritable ``/`` and username lookups
    # fail, breaking ChemShell/Intel-MPI. Point ``HOME`` at the (host-owned)
    # working directory and provide a ``USER`` so this works for any uid.
    engine_command = (
        f"docker run --rm --user {os.getuid()}:{os.getgid()} "
        "-e HOME=/workdir -e USER=aiida "
        "-v $PWD:/workdir:rw -w /workdir --entrypoint= {image_name}"
    )

    return aiida_code(
        "core.code.containerized",
        label="chemsh-containerized",
        default_calc_job_plugin="chemshell",
        computer=computer,
        filepath_executable=filepath_executable,
        engine_command=engine_command,
        image_name=image_name,
        with_mpi=False,  # MPI is handled by plugin when using ``chemsh`` executable
    )


@pytest.fixture(scope="function")
def janus_code(aiida_code_installed):
    """Return a Janus AiiDA code instance."""
    import os
    import shutil

    janus_path = shutil.which("janus") or os.environ.get("JANUS_PATH")

    return aiida_code_installed(
        label="janus",
        default_calc_job_plugin="mlip.sp",
        filepath_executable=janus_path,
    )


@pytest.fixture
def water_structure_object() -> StructureData:
    """Return a AiiDA StructureData object of a water molecule."""
    structure = StructureData()
    structure_str = """3

    O   0.000 0.000 0.000
    H  -0.754606402 0.590032355 0.0
    H   0.754606402 0.590032355 0.0
    """
    structure._parse_xyz(structure_str)
    return structure


@pytest.fixture
def water_trajectory_object() -> TrajectoryData:
    """Return a AiiDA StructureData object of a water molecule."""
    trajectory = TrajectoryData()
    symbols = ["O", "H", "H"]
    positions = numpy.array(
        [
            [[0.0, 0.0, 0.0], [-0.9, 0.590032355, 0.0], [0.9, 0.590032355, 0.0]],
            [
                [0.0, 0.0, 0.0],
                [-0.754606402, 0.590032355, 0.0],
                [0.754606402, 0.590032355, 0.0],
            ],
            [[0.0, 0.0, 0.0], [-0.5, 0.590032355, 0.0], [0.5, 0.590032355, 0.0]],
        ]
    )
    if parse_version(aiida_core_version) < parse_version("2.8.0"):
        trajectory.set_trajectory(symbols=symbols, positions=positions)
    else:
        trajectory.set_trajectory(
            symbols=symbols, positions=positions, pbc=[False, False, False]
        )
    return trajectory


@pytest.fixture
def generate_inputs(chemsh_code, get_test_data_file):
    """Return a dictionary of inputs for the ChemShellCalculation."""

    def factory(
        sp: dict | None = None,
        qm: dict | None = None,
        mm: dict | None = None,
        structure_fname: str | StructureData = "water.cjson",
        ff_fname: str | None = None,
        opt: dict | None = None,
    ) -> dict:
        if isinstance(structure_fname, str):
            structure = get_test_data_file(structure_fname)
        else:
            structure = structure_fname
        inputs = {"code": chemsh_code, "structure": structure}
        if sp:
            inputs["calculation_parameters"] = Dict(sp)
        if qm:
            inputs["qm_parameters"] = Dict(qm)
            if "theory" not in qm:
                inputs["qm_parameters"]["theory"] = "NWChem"
        if mm:
            inputs["mm_parameters"] = Dict(mm)
            if "theory" not in mm:
                inputs["mm_parameters"]["theory"] = "DL_POLY"
        if not qm and not mm and not ff_fname:
            inputs["qm_parameters"] = Dict({"theory": "NWChem"})

        if ff_fname:
            inputs["force_field_file"] = get_test_data_file(ff_fname)
            if "mm_parameters" not in inputs:
                inputs["mm_parameters"] = Dict({"theory": "DL_POLY"})

        if opt:
            inputs["optimisation_parameters"] = Dict(opt)

        if "mm_parameters" in inputs and "qm_parameters" in inputs:
            inputs["qmmm_parameters"] = Dict({"qm_region": range(3)})

        return inputs

    return factory


@pytest.fixture
def generate_calcjob(tmp_path, generate_inputs):
    """Return an initialised aiida-chemshell CalcJob instance."""

    def factory(process_class: CalcJob, inputs=generate_inputs(), return_process=False):
        manager = get_manager()
        runner = manager.get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        if return_process:
            return process

        calc_info = process.prepare_for_submission(Folder(tmp_path))
        return tmp_path, calc_info

    return factory
