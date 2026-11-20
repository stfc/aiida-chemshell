"""Tests for the aiida-mcp integration module, aiida_chemshell.mcp.

Requires the optional ``mcp`` extra (``pip install -e .[mcp]``), which installs
aiida-mcp and the ``mcp`` SDK; skipped entirely if that extra is not installed (e.g.
in CI jobs that only install the base package).
"""

import asyncio
import uuid
from unittest import mock

import pytest

pytest.importorskip("aiida_mcp")

from aiida import orm  # noqa: E402
from aiida_mcp.plugins import calcjob, workflow  # noqa: E402
from aiida_mcp.plugins.inputs import ControlInput, FileInput  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from aiida_chemshell import mcp as chemshell_mcp  # noqa: E402


def _tool_names(server: MCPServer) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


@pytest.fixture
def code(aiida_localhost):
    """Return a stored ChemShell code on the test computer."""
    return orm.InstalledCode(
        label=f"chemshell-test-{uuid.uuid4().hex[:8]}",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="chemshell",
    ).store()


@pytest.fixture
def fake_submit():
    """Replace the daemon submit with a stub that returns a process-like node."""
    node = orm.CalcJobNode()
    node.set_process_type("aiida.calculations:chemshell")
    with mock.patch.object(calcjob, "submit", return_value=node) as patched:
        node.store_all = lambda: None
        yield patched


@pytest.fixture
def stored_node_returned(fake_submit):
    """Make the fake submit return an already-stored CalcJobNode."""
    fake_submit.return_value = orm.CalcJobNode().store()
    return fake_submit


def test_is_available():
    """is_available reflects whether the chemshell calcjob entry point is installed."""
    assert chemshell_mcp.is_available() is True


def test_describe_chemshell_inputs():
    """describe_calcjob reports the ChemShell-specific inputs."""
    description = calcjob.describe_calcjob("chemshell")
    inputs = {port["name"]: port for port in description["inputs"]}

    assert inputs["structure"]["required"] is True
    assert inputs["structure"]["types"] == [
        "SinglefileData",
        "StructureData",
        "TrajectoryData",
    ]
    assert inputs["code"]["required"] is False


def test_describe_chemshell_outputs_exclude_base_calcjob_outputs():
    """describe_calcjob excludes outputs inherited from the base CalcJob."""
    description = calcjob.describe_calcjob("chemshell")
    outputs = {port["name"]: port for port in description["outputs"]}

    assert "energy" in outputs
    assert "optimised_structure" in outputs
    assert "retrieved" not in outputs


def test_describe_chemshell_exit_codes():
    """describe_calcjob reports the ChemShell-specific exit codes."""
    description = calcjob.describe_calcjob("chemshell")
    statuses = {code["status"]: code["name"] for code in description["exit_codes"]}

    assert statuses[300] == "ERROR_STDOUT_NOT_FOUND"


def test_describe_chemshell_option_defaults():
    """describe_calcjob reports the metadata option defaults the plugin changed."""
    defaults = calcjob.describe_calcjob("chemshell")["metadata_option_defaults"]

    assert defaults["parser_name"] == "chemshell"
    assert defaults["resources"] == {"num_machines": 1, "num_mpiprocs_per_machine": 4}


def test_register_adds_tool():
    """register() adds the ChemShell calcjob and workflow describe/submit tools."""
    server = MCPServer("test")
    chemshell_mcp.register(server)
    assert _tool_names(server) == {
        "describe_chemshell_calcjob",
        "submit_chemshell_job",
        "describe_chemshell_opt_workflow",
        "submit_chemshell_opt_workflow",
        "describe_chemshell_atomic_energies_workflow",
        "submit_chemshell_atomic_energies_workflow",
        "describe_chemshell_batch_workflow",
        "submit_chemshell_batch_workflow",
    }


def test_submit_job_builds_validated_inputs(code, stored_node_returned):
    """submit_job builds, stores and submits the required inputs."""
    result = chemshell_mcp.submit_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        num_machines=2,
        num_mpiprocs_per_machine=8,
    )

    stored_node_returned.assert_called_once()
    submitted = stored_node_returned.call_args.kwargs
    assert submitted["code"].pk == code.pk
    assert submitted["structure"].is_stored
    assert submitted["output_structure_as_file"] is False
    resources = submitted["metadata"]["options"]["resources"]
    assert resources == {"num_machines": 2, "num_mpiprocs_per_machine": 8}

    assert result["pk"] == stored_node_returned.return_value.pk
    assert set(result["inputs"]) == {"structure"}
    assert result["code"] == code.full_label


def test_submit_job_includes_optional_inputs_when_given(code, stored_node_returned):
    """submit_job includes optional inputs only when they are given."""
    chemshell_mcp.submit_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        structure_index=2,
        calculation_parameters=ControlInput(parameters={"gradients": True}),
        mm_parameters=ControlInput(parameters={"theory": "DL_POLY"}),
        force_field_file=FileInput(text="FF data"),
        output_structure_as_file=True,
    )

    submitted = stored_node_returned.call_args.kwargs
    assert isinstance(submitted["structure_index"], orm.Int)
    assert submitted["structure_index"].value == 2
    assert isinstance(submitted["calculation_parameters"], orm.Dict)
    assert submitted["calculation_parameters"].get_dict() == {"gradients": True}
    assert isinstance(submitted["force_field_file"], orm.SinglefileData)
    assert submitted["output_structure_as_file"] is True

    unset_names = (
        "structure2",
        "optimisation_parameters",
        "qm_parameters",
        "qmmm_parameters",
    )
    for name in unset_names:
        assert name not in submitted


def test_submit_job_sets_label_and_description(code, stored_node_returned):
    """submit_job passes label and description through to the calcjob metadata."""
    chemshell_mcp.submit_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        label="my run",
        description="a test run",
    )

    metadata = stored_node_returned.call_args.kwargs["metadata"]
    assert metadata["label"] == "my run"
    assert metadata["description"] == "a test run"


def test_submit_job_omits_label_and_description_by_default(code, stored_node_returned):
    """submit_job omits label and description from metadata when not given."""
    chemshell_mcp.submit_job(
        code=code.pk, structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n")
    )

    metadata = stored_node_returned.call_args.kwargs["metadata"]
    assert "label" not in metadata
    assert "description" not in metadata


def test_submit_job_unknown_code_creates_nothing(fake_submit):
    """An unknown code is a ToolError and creates no nodes."""
    before = orm.QueryBuilder().append(orm.Node).count()

    with pytest.raises(ToolError, match="No code found"):
        chemshell_mcp.submit_job(
            code="nope@nowhere", structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n")
        )

    fake_submit.assert_not_called()
    assert orm.QueryBuilder().append(orm.Node).count() == before


def test_submit_job_invalid_structure_pk_store_nothing(code, fake_submit):
    """A PK of the wrong node type for structure is a ToolError and creates nothing."""
    wrong = orm.Int(1).store()
    before = orm.QueryBuilder().append(orm.Node).count()
    after_setup = before

    expected = "expected SinglefileData, StructureData, TrajectoryData"
    with pytest.raises(ToolError, match=expected):
        chemshell_mcp.submit_job(code=code.pk, structure=FileInput(pk=wrong.pk))

    fake_submit.assert_not_called()
    assert orm.QueryBuilder().append(orm.Node).count() == after_setup


def test_submit_job_rejects_bad_resources(code, fake_submit):
    """A resources value below 1 is a ToolError."""
    with pytest.raises(ToolError, match="num_machines must be at least 1"):
        chemshell_mcp.submit_job(
            code=code.pk,
            structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
            num_machines=0,
        )
    fake_submit.assert_not_called()


def test_submit_calcjob_validates_before_storing(aiida_profile, fake_submit):
    """submit_calcjob validates required inputs before storing any of them."""
    unstored = orm.SinglefileData(content="x", filename="structure.xyz")

    with pytest.raises(ToolError, match="Invalid inputs for chemshell"):
        calcjob.submit_calcjob("chemshell", {"structure2": unstored})

    assert not unstored.is_stored
    fake_submit.assert_not_called()


def test_submit_tool_is_registered_with_input_schema():
    """submit_chemshell_job is registered with the expected input schema."""
    server = MCPServer("test")
    chemshell_mcp.register(server)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    schema = tools["submit_chemshell_job"].input_schema
    assert set(schema["required"]) == {"code", "structure"}
    optional = {
        "structure_index",
        "structure2",
        "calculation_parameters",
        "optimisation_parameters",
        "qm_parameters",
        "mm_parameters",
        "qmmm_parameters",
        "force_field_file",
        "output_structure_as_file",
        "label",
        "description",
    }
    assert optional <= set(schema["properties"])


# --- workflow submission ---


@pytest.fixture
def fake_workflow_submit():
    """Replace aiida.engine.submit, as imported into aiida_mcp.plugins.workflow."""
    node = orm.WorkChainNode()
    node.set_process_type("aiida.workflows:chemshell.opt")
    with mock.patch.object(workflow, "submit", return_value=node) as patched:
        node.store_all = lambda: None
        yield patched


@pytest.fixture
def stored_workflow_node_returned(fake_workflow_submit):
    """Make the fake submit return an already-stored WorkChainNode."""
    fake_workflow_submit.return_value = orm.WorkChainNode().store()
    return fake_workflow_submit


def test_submit_opt_job_builds_validated_inputs(code, stored_workflow_node_returned):
    """submit_opt_job nests the ChemShell calcjob inputs under 'chemsh'."""
    result = chemshell_mcp.submit_opt_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        vibrational_analysis=True,
        num_machines=2,
        num_mpiprocs_per_machine=8,
    )

    stored_workflow_node_returned.assert_called_once()
    submitted = stored_workflow_node_returned.call_args.kwargs
    chemsh = submitted["chemsh"]
    assert chemsh["code"].pk == code.pk
    assert chemsh["structure"].is_stored
    resources = chemsh["metadata"]["options"]["resources"]
    assert resources == {"num_machines": 2, "num_mpiprocs_per_machine": 8}
    assert "label" not in chemsh["metadata"]
    assert isinstance(submitted["vibrational_analysis"], orm.Bool)
    assert submitted["vibrational_analysis"].value is True

    assert result["pk"] == stored_workflow_node_returned.return_value.pk
    assert set(result["inputs"]) == {"structure"}
    assert result["code"] == code.full_label


def test_submit_opt_job_sets_label_on_workflow_node_not_calcjob_step(
    code, stored_workflow_node_returned
):
    """submit_opt_job's label/description set the workflow's own top-level metadata."""
    chemshell_mcp.submit_opt_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        label="my opt run",
        description="an optimisation",
    )

    submitted = stored_workflow_node_returned.call_args.kwargs
    assert submitted["metadata"]["label"] == "my opt run"
    assert submitted["metadata"]["description"] == "an optimisation"
    assert "label" not in submitted["chemsh"]["metadata"]
    assert "description" not in submitted["chemsh"]["metadata"]


def test_submit_opt_job_omits_metadata_by_default(code, stored_workflow_node_returned):
    """submit_opt_job omits top-level metadata when no label/description is given."""
    chemshell_mcp.submit_opt_job(
        code=code.pk, structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n")
    )

    submitted = stored_workflow_node_returned.call_args.kwargs
    assert "metadata" not in submitted


def test_submit_opt_job_rejects_bad_resources(code, fake_workflow_submit):
    """A resources value below 1 is a ToolError."""
    with pytest.raises(ToolError, match="num_machines must be at least 1"):
        chemshell_mcp.submit_opt_job(
            code=code.pk,
            structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
            num_machines=0,
        )
    fake_workflow_submit.assert_not_called()


def test_submit_atomic_energies_job_builds_top_level_inputs(
    code, stored_workflow_node_returned
):
    """submit_atomic_energies_job exposes code/structure/qm_parameters at top level."""
    result = chemshell_mcp.submit_atomic_energies_job(
        code=code.pk,
        structure=FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
        qm_parameters=ControlInput(parameters={"theory": "NWChem"}),
    )

    stored_workflow_node_returned.assert_called_once()
    submitted = stored_workflow_node_returned.call_args.kwargs
    assert submitted["code"].pk == code.pk
    assert submitted["structure"].is_stored
    assert isinstance(submitted["qm_parameters"], orm.Dict)
    assert "metadata" in submitted["chemsh"]
    assert "code" not in submitted["chemsh"]

    assert result["pk"] == stored_workflow_node_returned.return_value.pk
    assert set(result["inputs"]) == {"structure", "qm_parameters"}


def test_submit_batch_job_requires_a_structure_source(code, fake_workflow_submit):
    """Giving none of trajectory/structures/structure_files is a ToolError."""
    with pytest.raises(ToolError, match="Give at least one of"):
        chemshell_mcp.submit_batch_job(code=code.pk)
    fake_workflow_submit.assert_not_called()


def test_submit_batch_job_builds_structures_namespace(
    code, stored_workflow_node_returned
):
    """submit_batch_job builds the structures namespace from a dict of FileInputs."""
    result = chemshell_mcp.submit_batch_job(
        code=code.pk,
        structures={
            "water": FileInput(text="1\n\nH 0.0 0.0 0.0\n"),
            "methane": FileInput(text="1\n\nC 0.0 0.0 0.0\n"),
        },
    )

    stored_workflow_node_returned.assert_called_once()
    submitted = stored_workflow_node_returned.call_args.kwargs
    assert set(submitted["structures"]) == {"water", "methane"}
    assert all(node.is_stored for node in submitted["structures"].values())
    assert submitted["calc"]["metadata"]["options"]["resources"] == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 4,
    }

    assert set(result["inputs"]["structures"]) == {"water", "methane"}


def test_submit_batch_job_builds_structure_files_by_index(
    code, stored_workflow_node_returned
):
    """submit_batch_job keys the structure_files namespace by list index."""
    chemshell_mcp.submit_batch_job(
        code=code.pk,
        structure_files=[
            FileInput(text="structure one"),
            FileInput(text="structure two"),
        ],
    )

    submitted = stored_workflow_node_returned.call_args.kwargs
    assert set(submitted["structure_files"]) == {"0", "1"}


def test_submit_batch_job_builds_trajectory_input(code, stored_workflow_node_returned):
    """submit_batch_job loads an existing TrajectoryData node by PK."""
    trajectory = orm.TrajectoryData().store()

    chemshell_mcp.submit_batch_job(code=code.pk, trajectory=trajectory.pk)

    submitted = stored_workflow_node_returned.call_args.kwargs
    assert submitted["trajectory"].pk == trajectory.pk


def test_submit_batch_job_unknown_trajectory_pk(code, fake_workflow_submit):
    """A trajectory PK that doesn't exist is a ToolError."""
    with pytest.raises(ToolError, match="no node with pk 999999"):
        chemshell_mcp.submit_batch_job(code=code.pk, trajectory=999999)
    fake_workflow_submit.assert_not_called()


def test_submit_batch_job_trajectory_wrong_type(code, fake_workflow_submit):
    """A trajectory PK that isn't a TrajectoryData is a ToolError."""
    wrong = orm.Int(1).store()
    with pytest.raises(ToolError, match="expected TrajectoryData"):
        chemshell_mcp.submit_batch_job(code=code.pk, trajectory=wrong.pk)
    fake_workflow_submit.assert_not_called()
