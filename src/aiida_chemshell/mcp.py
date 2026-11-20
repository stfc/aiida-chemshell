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
from aiida_mcp.plugins.inputs import (
    ControlInput,
    FileInput,
    build_node,
    load_existing_node,
)
from aiida_mcp.plugins.workflow import describe_workflow, submit_workflow
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ENTRY_POINT = "chemshell"
OPT_WORKFLOW_ENTRY_POINT = "chemshell.opt"
ATOMIC_ENERGIES_WORKFLOW_ENTRY_POINT = "chemshell.atomic_energies"
BATCH_WORKFLOW_ENTRY_POINT = "chemshell.batch"

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

    This is the low-level single-calculation entry point. For a geometry optimisation,
    prefer `submit_chemshell_opt_workflow`, which also runs isolated-atom energies and
    an optional vibrational analysis automatically; for several independent structures
    sharing the same calculation settings, prefer `submit_chemshell_batch_workflow`.

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


def _options_metadata(
    num_machines: int, num_mpiprocs_per_machine: int
) -> dict[str, Any]:
    """Build a calcjob-step ``metadata`` dict carrying only resource options.

    For a WorkChain's exposed-calcjob-step namespace (e.g. ``chemsh.metadata``) —
    label/description belong on the *workflow's own* top-level ``metadata`` instead
    (see `_workflow_metadata`), not on an inner calcjob step's.
    """
    for name, value in (
        ("num_machines", num_machines),
        ("num_mpiprocs_per_machine", num_mpiprocs_per_machine),
    ):
        if value < 1:
            raise ToolError(f"{name} must be at least 1, got {value}")

    return {
        "options": {
            "resources": {
                "num_machines": num_machines,
                "num_mpiprocs_per_machine": num_mpiprocs_per_machine,
            }
        }
    }


def _workflow_metadata(label: str | None, description: str | None) -> dict[str, Any]:
    """Build the top-level ``metadata`` dict (label/description) for a workflow node."""
    metadata: dict[str, Any] = {}
    if label is not None:
        metadata["label"] = label
    if description is not None:
        metadata["description"] = description
    return metadata


def submit_opt_job(
    code: str | int,
    structure: FileInput,
    structure2: FileInput | None = None,
    calculation_parameters: ControlInput | None = None,
    optimisation_parameters: ControlInput | None = None,
    qm_parameters: ControlInput | None = None,
    mm_parameters: ControlInput | None = None,
    qmmm_parameters: ControlInput | None = None,
    force_field_file: FileInput | None = None,
    output_structure_as_file: bool = False,
    vibrational_analysis: bool = False,
    num_machines: int = 1,
    num_mpiprocs_per_machine: int = 4,
    label: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build the inputs for a ChemShell geometry optimisation workflow and submit it.

    Runs a geometry optimisation, isolated-atom energies for every element in the
    structure, and (optionally) a vibrational analysis of the optimised geometry — all
    in one submission. Prefer this over `submit_chemshell_job` for optimisations.

    Parameters
    ----------
    code : str or int
        Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
    structure : FileInput
        The input structure, as a path, text, or PK of a `SinglefileData`
        (``.xyz``, ``.pun`` or ``.cjson``), `StructureData`, or
        `TrajectoryData` node.
    structure2 : FileInput or None
        A second input structure, e.g. the final structure for a NEB
        optimisation. Same accepted forms as `structure`.
    calculation_parameters : ControlInput or None
        Dict of parameters for the ChemShell Task object.
    optimisation_parameters : ControlInput or None
        Dict of parameters for the geometry optimisation task.
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
    vibrational_analysis : bool
        If true, also calculate the vibrational modes and thermochemical
        properties of the optimised structure.
    num_machines : int
        Number of machines to run on, for each calculation step.
    num_mpiprocs_per_machine : int
        MPI processes per machine, for each calculation step.
    label : str or None
        Optional label to set on the workflow node.
    description : str or None
        Optional description to set on the workflow node.

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

    chemsh_inputs: dict[str, Any] = {
        "code": load_code(code),
        "structure": build_node(
            structure, "structure", "structure.xyz", STRUCTURE_TYPES
        ),
        "output_structure_as_file": output_structure_as_file,
        "metadata": _options_metadata(num_machines, num_mpiprocs_per_machine),
    }
    chemsh_inputs.update(
        _optional_inputs(
            structure_index=None,
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

    inputs: dict[str, Any] = {
        "chemsh": chemsh_inputs,
        "vibrational_analysis": orm.Bool(vibrational_analysis),
    }
    workflow_metadata = _workflow_metadata(label, description)
    if workflow_metadata:
        inputs["metadata"] = workflow_metadata

    node = submit_workflow(OPT_WORKFLOW_ENTRY_POINT, inputs)

    daemon_running = get_daemon_client(get_profile().name).is_daemon_running
    data_inputs = {
        name: chemsh_inputs[name].pk
        for name in DATA_INPUT_NAMES
        if name in chemsh_inputs
    }
    result: dict[str, Any] = {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.label,
        "description": node.description,
        "code": chemsh_inputs["code"].full_label,
        "inputs": data_inputs,
        "daemon_running": daemon_running,
    }
    if not daemon_running:
        result["note"] = "The daemon is not running; the job will wait until it starts."
    return result


def submit_atomic_energies_job(
    code: str | int,
    structure: FileInput,
    qm_parameters: ControlInput,
    num_machines: int = 1,
    num_mpiprocs_per_machine: int = 4,
    label: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build the inputs for a ChemShell isolated-atom-energies workflow and submit it.

    Extracts every unique element from `structure`, builds an isolated single-atom
    structure for each, and runs an independent single-point QM calculation on it.

    Parameters
    ----------
    code : str or int
        Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
    structure : FileInput
        The structure to extract isolated atomic energies from, as a path,
        text, or PK of a `SinglefileData`, `StructureData`, or
        `TrajectoryData` node.
    qm_parameters : ControlInput
        Dict of parameters for the QM Theory object. The ``PySCF`` QM
        backend is not supported for isolated-atom calculations.
    num_machines : int
        Number of machines to run on, for each atom's calculation.
    num_mpiprocs_per_machine : int
        MPI processes per machine, for each atom's calculation.
    label : str or None
        Optional label to set on the workflow node.
    description : str or None
        Optional description to set on the workflow node.

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

    inputs: dict[str, Any] = {
        "code": load_code(code),
        "structure": build_node(
            structure, "structure", "structure.xyz", STRUCTURE_TYPES
        ),
        "qm_parameters": build_node(
            qm_parameters, "qm_parameters", "qm_parameters.json", (orm.Dict,)
        ),
        "chemsh": {
            "metadata": _options_metadata(num_machines, num_mpiprocs_per_machine)
        },
    }
    workflow_metadata = _workflow_metadata(label, description)
    if workflow_metadata:
        inputs["metadata"] = workflow_metadata

    node = submit_workflow(ATOMIC_ENERGIES_WORKFLOW_ENTRY_POINT, inputs)

    daemon_running = get_daemon_client(get_profile().name).is_daemon_running
    result: dict[str, Any] = {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.label,
        "description": node.description,
        "code": inputs["code"].full_label,
        "inputs": {
            "structure": inputs["structure"].pk,
            "qm_parameters": inputs["qm_parameters"].pk,
        },
        "daemon_running": daemon_running,
    }
    if not daemon_running:
        result["note"] = "The daemon is not running; the job will wait until it starts."
    return result


def submit_batch_job(
    code: str | int,
    trajectory: int | None = None,
    structures: dict[str, FileInput] | None = None,
    structure_files: list[FileInput] | None = None,
    calculation_parameters: ControlInput | None = None,
    optimisation_parameters: ControlInput | None = None,
    qm_parameters: ControlInput | None = None,
    mm_parameters: ControlInput | None = None,
    qmmm_parameters: ControlInput | None = None,
    force_field_file: FileInput | None = None,
    output_structure_as_file: bool = False,
    combine_results: bool = False,
    num_machines: int = 1,
    num_mpiprocs_per_machine: int = 4,
    label: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build the inputs for a ChemShell batch workflow and submit it to the daemon.

    Runs the same calculation settings independently over several structures. Give
    exactly one of `trajectory`, `structures`, or `structure_files`.

    Parameters
    ----------
    code : str or int
        Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code,
        shared by every structure in the batch.
    trajectory : int or None
        PK of an existing `TrajectoryData` node; every frame is processed.
    structures : dict[str, FileInput] or None
        A mapping of arbitrary keys to structures (path, text, or PK of a
        `StructureData` node) to batch process.
    structure_files : list of FileInput or None
        A list of structure files (path, text, or PK of a `SinglefileData`
        node) to batch process; each is parsed for the structures it
        contains (only ``.xyz`` files are currently supported).
    calculation_parameters : ControlInput or None
        Dict of parameters for the ChemShell Task object, shared by every structure.
    optimisation_parameters : ControlInput or None
        Dict of parameters for a geometry optimisation task, shared by every
        structure. Note: `combine_results` is not supported together with this.
    qm_parameters : ControlInput or None
        Dict of parameters for the QM Theory object, shared by every structure.
    mm_parameters : ControlInput or None
        Dict of parameters for the MM interface, shared by every structure.
    qmmm_parameters : ControlInput or None
        Dict of parameters for the QM/MM interface, shared by every structure.
    force_field_file : FileInput or None
        File with force field parameters for the MM interface, shared by
        every structure.
    output_structure_as_file : bool
        If true, return each optimised structure as a `SinglefileData`
        ChemShell ``.cjson`` file instead of a `StructureData` node.
    combine_results : bool
        If true, combine every structure's results into one extended XYZ
        file output. Not supported together with `optimisation_parameters`.
    num_machines : int
        Number of machines to run on, for each structure's calculation.
    num_mpiprocs_per_machine : int
        MPI processes per machine, for each structure's calculation.
    label : str or None
        Optional label to set on the workflow node.
    description : str or None
        Optional description to set on the workflow node.

    Returns
    -------
    dict[str, Any]
        The process PK, uuid, label, description, code, input node PKs, and
        daemon state.

    Raises
    ------
    ToolError
        If none of `trajectory`/`structures`/`structure_files` is given,
        `num_machines` or `num_mpiprocs_per_machine` is below 1, `code`
        does not match a known code, or an input fails validation.
    """
    connection.ensure_profile()

    if trajectory is None and not structures and not structure_files:
        raise ToolError(
            "Give at least one of trajectory, structures, or structure_files."
        )

    inputs: dict[str, Any] = {
        "code": load_code(code),
        "calc": {"metadata": _options_metadata(num_machines, num_mpiprocs_per_machine)},
        "output_structure_as_file": output_structure_as_file,
    }
    workflow_metadata = _workflow_metadata(label, description)
    if workflow_metadata:
        inputs["metadata"] = workflow_metadata
    inputs.update(
        _optional_inputs(
            structure_index=None,
            structure2=None,
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
    if trajectory is not None:
        inputs["trajectory"] = load_existing_node(
            trajectory, "trajectory", (orm.TrajectoryData,)
        )
    if structures:
        inputs["structures"] = {
            key: build_node(
                value, f"structures.{key}", f"{key}.xyz", (orm.StructureData,)
            )
            for key, value in structures.items()
        }
    if structure_files:
        inputs["structure_files"] = {
            str(index): build_node(
                value,
                f"structure_files[{index}]",
                f"structure_{index}.xyz",
                (orm.SinglefileData,),
            )
            for index, value in enumerate(structure_files)
        }
    if combine_results:
        inputs["combine_results"] = True

    node = submit_workflow(BATCH_WORKFLOW_ENTRY_POINT, inputs)

    daemon_running = get_daemon_client(get_profile().name).is_daemon_running
    reported_inputs: dict[str, Any] = {
        name: inputs[name].pk
        for name in ("force_field_file", *PARAMETER_INPUTS)
        if name in inputs
    }
    if "trajectory" in inputs:
        reported_inputs["trajectory"] = inputs["trajectory"].pk
    if "structures" in inputs:
        reported_inputs["structures"] = {
            key: structure.pk for key, structure in inputs["structures"].items()
        }
    if "structure_files" in inputs:
        reported_inputs["structure_files"] = {
            key: structure.pk for key, structure in inputs["structure_files"].items()
        }
    result: dict[str, Any] = {
        "pk": node.pk,
        "uuid": node.uuid,
        "label": node.label,
        "description": node.description,
        "code": inputs["code"].full_label,
        "inputs": reported_inputs,
        "daemon_running": daemon_running,
    }
    if not daemon_running:
        result["note"] = "The daemon is not running; the job will wait until it starts."
    return result


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

        This is the low-level single-calculation entry point. For a geometry
        optimisation, prefer `submit_chemshell_opt_workflow`, which also runs
        isolated-atom energies and an optional vibrational analysis automatically; for
        several independent structures sharing the same calculation settings, prefer
        `submit_chemshell_batch_workflow`.

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

    @mcp.tool()
    def describe_chemshell_opt_workflow() -> dict[str, Any]:
        """Describe the ChemShell geometry optimisation workflow: outputs, exit codes.

        See `submit_chemshell_opt_workflow`'s docstring for its actual arguments.

        Returns
        -------
        dict[str, Any]
            The workflow's outputs and exit codes.
        """
        return describe_workflow(OPT_WORKFLOW_ENTRY_POINT)

    @mcp.tool()
    def submit_chemshell_opt_workflow(
        code: str | int,
        structure: FileInput,
        structure2: FileInput | None = None,
        calculation_parameters: ControlInput | None = None,
        optimisation_parameters: ControlInput | None = None,
        qm_parameters: ControlInput | None = None,
        mm_parameters: ControlInput | None = None,
        qmmm_parameters: ControlInput | None = None,
        force_field_file: FileInput | None = None,
        output_structure_as_file: bool = False,
        vibrational_analysis: bool = False,
        num_machines: int = 1,
        num_mpiprocs_per_machine: int = 4,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Submit a ChemShell geometry optimisation workflow to the daemon.

        Creates nodes and starts a run. Runs a geometry optimisation, isolated-atom
        energies for every element in the structure, and (optionally) a vibrational
        analysis of the optimised geometry — all in one submission. Prefer this over
        `submit_chemshell_job` for optimisations. Follow the job with `get_process`
        and `get_process_report`.

        Parameters
        ----------
        code : str or int
            Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
        structure : FileInput
            The input structure, as a path, text, or PK of a
            `SinglefileData` (``.xyz``, ``.pun`` or ``.cjson``),
            `StructureData`, or `TrajectoryData` node.
        structure2 : FileInput or None
            A second input structure, e.g. the final structure for a NEB
            optimisation. Same accepted forms as `structure`.
        calculation_parameters : ControlInput or None
            Dict of parameters for the ChemShell Task object.
        optimisation_parameters : ControlInput or None
            Dict of parameters for the geometry optimisation task.
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
        vibrational_analysis : bool
            If true, also calculate the vibrational modes and
            thermochemical properties of the optimised structure.
        num_machines : int
            Number of machines to run on, for each calculation step.
        num_mpiprocs_per_machine : int
            MPI processes per machine, for each calculation step.
        label : str or None
            Optional label to set on the workflow node.
        description : str or None
            Optional description to set on the workflow node.

        Returns
        -------
        dict[str, Any]
            The process PK, uuid, label, description, code, input node PKs,
            and daemon state.
        """
        return submit_opt_job(
            code=code,
            structure=structure,
            structure2=structure2,
            calculation_parameters=calculation_parameters,
            optimisation_parameters=optimisation_parameters,
            qm_parameters=qm_parameters,
            mm_parameters=mm_parameters,
            qmmm_parameters=qmmm_parameters,
            force_field_file=force_field_file,
            output_structure_as_file=output_structure_as_file,
            vibrational_analysis=vibrational_analysis,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            label=label,
            description=description,
        )

    @mcp.tool()
    def describe_chemshell_atomic_energies_workflow() -> dict[str, Any]:
        """Describe the ChemShell isolated-atom-energies workflow: outputs, exit codes.

        See `submit_chemshell_atomic_energies_workflow`'s docstring for its actual
        arguments.

        Returns
        -------
        dict[str, Any]
            The workflow's outputs and exit codes.
        """
        return describe_workflow(ATOMIC_ENERGIES_WORKFLOW_ENTRY_POINT)

    @mcp.tool()
    def submit_chemshell_atomic_energies_workflow(
        code: str | int,
        structure: FileInput,
        qm_parameters: ControlInput,
        num_machines: int = 1,
        num_mpiprocs_per_machine: int = 4,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Submit a ChemShell isolated-atom-energies workflow to the daemon.

        Extracts every unique element from `structure`, builds an isolated single-atom
        structure for each, and runs an independent single-point QM calculation on it.
        Follow the job with `get_process` and `get_process_report`.

        Parameters
        ----------
        code : str or int
            Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
        structure : FileInput
            The structure to extract isolated atomic energies from, as a
            path, text, or PK of a `SinglefileData`, `StructureData`, or
            `TrajectoryData` node.
        qm_parameters : ControlInput
            Dict of parameters for the QM Theory object. The ``PySCF`` QM
            backend is not supported for isolated-atom calculations.
        num_machines : int
            Number of machines to run on, for each atom's calculation.
        num_mpiprocs_per_machine : int
            MPI processes per machine, for each atom's calculation.
        label : str or None
            Optional label to set on the workflow node.
        description : str or None
            Optional description to set on the workflow node.

        Returns
        -------
        dict[str, Any]
            The process PK, uuid, label, description, code, input node PKs,
            and daemon state.
        """
        return submit_atomic_energies_job(
            code=code,
            structure=structure,
            qm_parameters=qm_parameters,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            label=label,
            description=description,
        )

    @mcp.tool()
    def describe_chemshell_batch_workflow() -> dict[str, Any]:
        """Describe the ChemShell batch workflow: outputs and exit codes.

        See `submit_chemshell_batch_workflow`'s docstring for its actual arguments.

        Returns
        -------
        dict[str, Any]
            The workflow's outputs and exit codes.
        """
        return describe_workflow(BATCH_WORKFLOW_ENTRY_POINT)

    @mcp.tool()
    def submit_chemshell_batch_workflow(
        code: str | int,
        trajectory: int | None = None,
        structures: dict[str, FileInput] | None = None,
        structure_files: list[FileInput] | None = None,
        calculation_parameters: ControlInput | None = None,
        optimisation_parameters: ControlInput | None = None,
        qm_parameters: ControlInput | None = None,
        mm_parameters: ControlInput | None = None,
        qmmm_parameters: ControlInput | None = None,
        force_field_file: FileInput | None = None,
        output_structure_as_file: bool = False,
        combine_results: bool = False,
        num_machines: int = 1,
        num_mpiprocs_per_machine: int = 4,
        label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Submit a ChemShell batch workflow to the daemon.

        Runs the same settings over many structures. Give exactly one of
        `trajectory`, `structures`, or `structure_files`. Prefer this over calling
        `submit_chemshell_job` once per structure when several independent structures
        share the same calculation settings. Follow the job with `get_process` and
        `get_process_report`.

        Parameters
        ----------
        code : str or int
            Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code,
            shared by every structure in the batch.
        trajectory : int or None
            PK of an existing `TrajectoryData` node; every frame is processed.
        structures : dict[str, FileInput] or None
            A mapping of arbitrary keys to structures (path, text, or PK of
            a `StructureData` node) to batch process.
        structure_files : list of FileInput or None
            A list of structure files (path, text, or PK of a
            `SinglefileData` node) to batch process; each is parsed for the
            structures it contains (only ``.xyz`` files are currently
            supported).
        calculation_parameters : ControlInput or None
            Dict of parameters for the ChemShell Task object, shared by
            every structure.
        optimisation_parameters : ControlInput or None
            Dict of parameters for a geometry optimisation task, shared by
            every structure. Note: `combine_results` is not supported
            together with this.
        qm_parameters : ControlInput or None
            Dict of parameters for the QM Theory object, shared by every
            structure.
        mm_parameters : ControlInput or None
            Dict of parameters for the MM interface, shared by every
            structure.
        qmmm_parameters : ControlInput or None
            Dict of parameters for the QM/MM interface, shared by every
            structure.
        force_field_file : FileInput or None
            File with force field parameters for the MM interface, shared
            by every structure.
        output_structure_as_file : bool
            If true, return each optimised structure as a `SinglefileData`
            ChemShell ``.cjson`` file instead of a `StructureData` node.
        combine_results : bool
            If true, combine every structure's results into one extended
            XYZ file output. Not supported together with
            `optimisation_parameters`.
        num_machines : int
            Number of machines to run on, for each structure's calculation.
        num_mpiprocs_per_machine : int
            MPI processes per machine, for each structure's calculation.
        label : str or None
            Optional label to set on the workflow node.
        description : str or None
            Optional description to set on the workflow node.

        Returns
        -------
        dict[str, Any]
            The process PK, uuid, label, description, code, input node PKs,
            and daemon state.
        """
        return submit_batch_job(
            code=code,
            trajectory=trajectory,
            structures=structures,
            structure_files=structure_files,
            calculation_parameters=calculation_parameters,
            optimisation_parameters=optimisation_parameters,
            qm_parameters=qm_parameters,
            mm_parameters=mm_parameters,
            qmmm_parameters=qmmm_parameters,
            force_field_file=force_field_file,
            output_structure_as_file=output_structure_as_file,
            combine_results=combine_results,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            label=label,
            description=description,
        )
