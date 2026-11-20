"""ChemShell tools for aiida-mcp, registered via the aiida_mcp.plugins entry point."""

from typing import Any

from aiida import orm
from aiida.engine.daemon.client import get_daemon_client
from aiida.manage.configuration import get_profile
from aiida_mcp import connection
from aiida_mcp.plugins.calcjob import (
    calcjob_available,
    describe_calcjob,
    load_code,
    submit_calcjob,
)
from aiida_mcp.plugins.inputs import ControlInput, FileInput, build_node
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ENTRY_POINT = "chemshell"

STRUCTURE_TYPES = (orm.SinglefileData, orm.StructureData, orm.TrajectoryData)
PARAMETER_INPUTS = (
    "calculation_parameters",
    "optimisation_parameters",
    "qm_parameters",
    "mm_parameters",
    "qmmm_parameters",
)
DATA_INPUT_NAMES = (
    "structure",
    "structure_index",
    "structure2",
    "force_field_file",
    *PARAMETER_INPUTS,
)


def is_available() -> bool:
    """Return whether the ChemShell calcjob plugin is installed.

    Returns
    -------
    bool
        True if the ``chemshell`` entry point is installed.
    """
    return calcjob_available(ENTRY_POINT)


def submit_job(
    code: str | int,
    structure: FileInput,
    structure_index: int | None = None,
    structure2: FileInput | None = None,
    calculation_parameters: ControlInput | None = None,
    optimisation_parameters: ControlInput | None = None,
    qm_parameters: ControlInput | None = None,
    mm_parameters: ControlInput | None = None,
    qmmm_parameters: ControlInput | None = None,
    force_field_file: FileInput | None = None,
    output_structure_as_file: bool = False,
    num_machines: int = 1,
    num_mpiprocs_per_machine: int = 4,
    label: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build the inputs for a ChemShell calcjob and submit it to the daemon.

    Parameters
    ----------
    code : str or int
        Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
    structure : FileInput
        The input structure, as a path, text, or PK of a `SinglefileData`
        (``.xyz``, ``.pun`` or ``.cjson``), `StructureData`, or
        `TrajectoryData` node.
    structure_index : int or None
        Index to extract one structure from a multi-structure
        `TrajectoryData` input.
    structure2 : FileInput or None
        A second input structure, e.g. the final structure for a NEB
        optimisation. Same accepted forms as `structure`.
    calculation_parameters : ControlInput or None
        Dict of parameters for the ChemShell Task object.
    optimisation_parameters : ControlInput or None
        Dict of parameters for a geometry optimisation task. Providing this
        configures an optimisation task for the job.
    qm_parameters : ControlInput or None
        Dict of parameters for the QM Theory object.
    mm_parameters : ControlInput or None
        Dict of parameters for the MM interface.
    qmmm_parameters : ControlInput or None
        Dict of parameters for the QM/MM interface.
    force_field_file : FileInput or None
        File with force field parameters for the MM interface, as a path,
        text, or PK of a `SinglefileData` node.
    output_structure_as_file : bool
        If true, return the optimised structure as a `SinglefileData`
        ChemShell ``.cjson`` file instead of a `StructureData` node.
    num_machines : int
        Number of machines to run on.
    num_mpiprocs_per_machine : int
        MPI processes per machine.
    label : str or None
        Optional label to set on the calcjob node.
    description : str or None
        Optional description to set on the calcjob node.

    Returns
    -------
    dict[str, Any]
        The process PK, uuid, label, description, code, input node PKs, and
        daemon state.

    Raises
    ------
    ToolError
        If `num_machines` or `num_mpiprocs_per_machine` is below 1, `code`
        does not match a known code, or an input fails validation.
    """
    connection.ensure_profile()

    resources = (
        ("num_machines", num_machines),
        ("num_mpiprocs_per_machine", num_mpiprocs_per_machine),
    )
    for name, value in resources:
        if value < 1:
            raise ToolError(f"{name} must be at least 1, got {value}")

    metadata: dict[str, Any] = {
        "options": {
            "resources": {
                "num_machines": num_machines,
                "num_mpiprocs_per_machine": num_mpiprocs_per_machine,
            }
        }
    }
    if label is not None:
        metadata["label"] = label
    if description is not None:
        metadata["description"] = description

    inputs: dict[str, Any] = {
        "code": load_code(code),
        "structure": build_node(
            structure, "structure", "structure.xyz", STRUCTURE_TYPES
        ),
        "output_structure_as_file": output_structure_as_file,
        "metadata": metadata,
    }
    inputs.update(
        _optional_inputs(
            structure_index=structure_index,
            structure2=structure2,
            force_field_file=force_field_file,
            parameters={
                "calculation_parameters": calculation_parameters,
                "optimisation_parameters": optimisation_parameters,
                "qm_parameters": qm_parameters,
                "mm_parameters": mm_parameters,
                "qmmm_parameters": qmmm_parameters,
            },
        )
    )

    node = submit_calcjob(ENTRY_POINT, inputs)

    daemon_running = get_daemon_client(get_profile().name).is_daemon_running
    data_inputs = {name: inputs[name].pk for name in DATA_INPUT_NAMES if name in inputs}
    result: dict[str, Any] = {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.label,
        "description": node.description,
        "code": inputs["code"].full_label,
        "inputs": data_inputs,
        "daemon_running": daemon_running,
    }
    if not daemon_running:
        result["note"] = "The daemon is not running; the job will wait until it starts."
    return result


def _optional_inputs(
    structure_index: int | None,
    structure2: FileInput | None,
    force_field_file: FileInput | None,
    parameters: dict[str, ControlInput | None],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    if structure_index is not None:
        inputs["structure_index"] = orm.Int(structure_index)
    if structure2 is not None:
        inputs["structure2"] = build_node(
            structure2, "structure2", "structure2.xyz", STRUCTURE_TYPES
        )
    if force_field_file is not None:
        inputs["force_field_file"] = build_node(
            force_field_file, "force_field_file", "forcefield.ff", (orm.SinglefileData,)
        )
    for name, value in parameters.items():
        if value is not None:
            inputs[name] = build_node(value, name, f"{name}.json", (orm.Dict,))
    return inputs


def register(mcp: MCPServer) -> None:
    """Register the ChemShell tools on the given MCP server.

    Parameters
    ----------
    mcp : MCPServer
        The server to register the ChemShell tools on.
    """

    @mcp.tool()
    def describe_chemshell_calcjob() -> dict[str, Any]:
        """Describe the ChemShell calcjob: inputs, outputs, exit codes and defaults.

        Returns
        -------
        dict[str, Any]
            The calcjob's inputs, outputs, exit codes and changed metadata
            option defaults.
        """
        return describe_calcjob(ENTRY_POINT)

    @mcp.tool()
    def submit_chemshell_job(
        code: str | int,
        structure: FileInput,
        structure_index: int | None = None,
        structure2: FileInput | None = None,
        calculation_parameters: ControlInput | None = None,
        optimisation_parameters: ControlInput | None = None,
        qm_parameters: ControlInput | None = None,
        mm_parameters: ControlInput | None = None,
        qmmm_parameters: ControlInput | None = None,
        force_field_file: FileInput | None = None,
        output_structure_as_file: bool = False,
        num_machines: int = 1,
        num_mpiprocs_per_machine: int = 4,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Submit a ChemShell calculation to the daemon; creates nodes and starts a job.

        Give exactly one of path, text, pk (or parameters for the ``*_parameters``
        inputs) for each file-like input. Follow the job with `get_process` and
        `get_process_report`.

        Parameters
        ----------
        code : str or int
            Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
        structure : FileInput
            The input structure, as a path, text, or PK of a
            `SinglefileData` (``.xyz``, ``.pun`` or ``.cjson``),
            `StructureData`, or `TrajectoryData` node.
        structure_index : int or None
            Index to extract one structure from a multi-structure
            `TrajectoryData` input.
        structure2 : FileInput or None
            A second input structure, e.g. the final structure for a NEB
            optimisation. Same accepted forms as `structure`.
        calculation_parameters : ControlInput or None
            Dict of parameters for the ChemShell Task object.
        optimisation_parameters : ControlInput or None
            Dict of parameters for a geometry optimisation task. Providing
            this configures an optimisation task for the job.
        qm_parameters : ControlInput or None
            Dict of parameters for the QM Theory object.
        mm_parameters : ControlInput or None
            Dict of parameters for the MM interface.
        qmmm_parameters : ControlInput or None
            Dict of parameters for the QM/MM interface.
        force_field_file : FileInput or None
            File with force field parameters for the MM interface, as a
            path, text, or PK of a `SinglefileData` node.
        output_structure_as_file : bool
            If true, return the optimised structure as a `SinglefileData`
            ChemShell ``.cjson`` file instead of a `StructureData` node.
        num_machines : int
            Number of machines to run on.
        num_mpiprocs_per_machine : int
            MPI processes per machine.
        label : str or None
            Optional label to set on the calcjob node.
        description : str or None
            Optional description to set on the calcjob node.

        Returns
        -------
        dict[str, Any]
            The process PK, uuid, label, description, code, input node PKs,
            and daemon state.
        """
        return submit_job(
            code=code,
            structure=structure,
            structure_index=structure_index,
            structure2=structure2,
            calculation_parameters=calculation_parameters,
            optimisation_parameters=optimisation_parameters,
            qm_parameters=qm_parameters,
            mm_parameters=mm_parameters,
            qmmm_parameters=qmmm_parameters,
            force_field_file=force_field_file,
            output_structure_as_file=output_structure_as_file,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            label=label,
            description=description,
        )
