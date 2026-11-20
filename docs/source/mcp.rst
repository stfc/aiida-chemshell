MCP integration
================

`aiida-mcp <https://github.com/stfc/aiida-mcp>`_ is a
`Model Context Protocol <https://modelcontextprotocol.io>`_ server that lets an MCP
client, such as an AI assistant, inspect an AiiDA database and submit calculations.
``aiida-chemshell`` ships its own integration module, ``aiida_chemshell.mcp``, which
exposes both the raw ChemShell calcjob and three higher-level workflows as MCP tools.

Installing the optional ``mcp`` extra:

.. code-block:: bash

   pip install "aiida-chemshell[mcp]"

pulls in aiida-mcp and registers ``aiida_chemshell.mcp`` under its ``aiida_mcp.plugins``
entry point group. Any aiida-mcp server running in the same environment then
automatically exposes all eight tools below — no changes needed in aiida-mcp itself. See
aiida-mcp's own documentation for how to run the server and its other tools.

Which tool should I use?
---------------------------

``submit_chemshell_job`` runs a single ChemShell calculation and nothing else. Prefer
one of the workflow tools instead where they fit your task, since each one wraps extra
steps ``submit_chemshell_job`` does not do on its own:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Tool
   * - One calculation; you'll handle everything else yourself
     - ``submit_chemshell_job``
   * - Geometry optimisation (plus isolated-atom energies, optionally vibrational
       analysis)
     - ``submit_chemshell_opt_workflow``
   * - Isolated-atom reference energies for every element in a structure
     - ``submit_chemshell_atomic_energies_workflow``
   * - The same calculation settings over several independent structures (a trajectory,
       a set of ``StructureData`` nodes, or a batch of structure files)
     - ``submit_chemshell_batch_workflow``

``submit_chemshell_job``
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``code``
     - Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
   * - ``structure``
     - The input structure, as a path, text, or PK of a ``SinglefileData`` (``.xyz``,
       ``.pun`` or ``.cjson``), ``StructureData``, or ``TrajectoryData`` node.
   * - ``structure_index``
     - Index to extract one structure from a multi-structure ``TrajectoryData`` input.
   * - ``structure2``
     - A second input structure, e.g. the final structure for a NEB optimisation. Same
       accepted forms as ``structure``.
   * - ``calculation_parameters``
     - Dict of parameters for the ChemShell Task object.
   * - ``optimisation_parameters``
     - Dict of parameters for a geometry optimisation task. Providing this configures
       an optimisation task for the job.
   * - ``qm_parameters``
     - Dict of parameters for the QM Theory object.
   * - ``mm_parameters``
     - Dict of parameters for the MM interface.
   * - ``qmmm_parameters``
     - Dict of parameters for the QM/MM interface.
   * - ``force_field_file``
     - File with force field parameters for the MM interface, as a path, text, or PK of
       a ``SinglefileData`` node.
   * - ``output_structure_as_file``
     - If true, return the optimised structure as a ``SinglefileData`` ChemShell
       ``.cjson`` file instead of a ``StructureData`` node.
   * - ``num_machines``
     - Machines to run on. Default 1.
   * - ``num_mpiprocs_per_machine``
     - MPI processes per machine. Default 4.
   * - ``label``
     - Optional label to set on the calcjob node.
   * - ``description``
     - Optional description to set on the calcjob node.

Give exactly one of ``path`` (a file on the machine running the server), ``text`` (the
file contents as a string), or ``pk`` (an existing node in the database) for each
file-like input, or ``parameters`` for the ``*_parameters`` inputs (stored as a
``Dict`` node).

Example arguments:

.. code-block:: json

   {
     "code": "chemshell@localhost",
     "structure": {"path": "/data/water.cjson"},
     "qm_parameters": {"parameters": {"theory": "NWChem", "method": "HF", "basis": "3-21G"}}
   }

The inputs are checked against the calcjob definition before anything is stored — a bad
input returns an error and creates no nodes. On success the tool returns the process PK
and the PKs of the input nodes; follow the job with aiida-mcp's ``get_process`` and
``get_process_report`` tools. If the daemon is stopped, the job is created and waits
until the daemon starts, and the result says so.

A typical session: call ``get_status``, call ``list_codes`` to find the code, call
``submit_chemshell_job``, then poll ``get_process`` until ``state`` is ``finished`` and
check ``exit_status``. An ``exit_status`` of 0 means success; other values are listed by
``describe_chemshell_calcjob``.

``submit_chemshell_opt_workflow``
------------------------------------

Runs a geometry optimisation, isolated-atom energies for every element in the
structure, and (optionally) a vibrational analysis of the optimised geometry — all in
one submission. Takes the same arguments as ``submit_chemshell_job`` (minus
``structure_index``, which doesn't apply to a single optimisation), plus:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``vibrational_analysis``
     - If true, also calculate the vibrational modes and thermochemical properties of
       the optimised structure. Default false.

Example arguments:

.. code-block:: json

   {
     "code": "chemshell@localhost",
     "structure": {"path": "/data/water.cjson"},
     "qm_parameters": {"parameters": {"theory": "NWChem", "method": "DFT", "basis": "6-31G"}},
     "vibrational_analysis": true
   }

Behaviour is otherwise the same as ``submit_chemshell_job``: inputs are validated
before anything is stored, and the tool returns the process PK and the PKs of the
``chemsh``-namespaced input nodes it created. Note that ``label``/``description`` here
set the *workflow* node's label/description, not any individual calculation step's.
Follow the job with ``get_process`` and ``get_process_report``, and check
``exit_status`` — a value of 420–424 means a step of the workflow itself failed (see
``describe_chemshell_opt_workflow``); otherwise check the failed step's own
``exit_status`` against ``describe_chemshell_calcjob``.

``submit_chemshell_atomic_energies_workflow``
-------------------------------------------------

Extracts every unique element from ``structure``, builds an isolated single-atom
structure for each, and runs an independent single-point QM calculation on it.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``code``
     - Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code.
   * - ``structure``
     - The structure to extract isolated atomic energies from, as a path, text, or PK
       of a ``SinglefileData``, ``StructureData``, or ``TrajectoryData`` node.
   * - ``qm_parameters``
     - Dict of parameters for the QM Theory object. The ``PySCF`` QM backend is not
       supported for isolated-atom calculations.
   * - ``num_machines``
     - Machines to run on, for each atom's calculation. Default 1.
   * - ``num_mpiprocs_per_machine``
     - MPI processes per machine, for each atom's calculation. Default 4.
   * - ``label``
     - Optional label to set on the workflow node.
   * - ``description``
     - Optional description to set on the workflow node.

The tool returns the process PK and the PKs of the ``structure``/``qm_parameters``
input nodes. Its ``atom_energies`` output (see ``describe_chemshell_atomic_energies_workflow``)
is a ``Dict`` mapping each element symbol to its isolated-atom energy.

``submit_chemshell_batch_workflow``
--------------------------------------

Runs the same calculation settings independently over several structures. Give
**exactly one** of ``trajectory``, ``structures``, or ``structure_files``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``code``
     - Label (e.g. ``chemshell@localhost``) or PK of the ChemShell code, shared by
       every structure in the batch.
   * - ``trajectory``
     - PK of an existing ``TrajectoryData`` node; every frame is processed.
   * - ``structures``
     - A JSON object mapping arbitrary keys to structures (``{"path": ...}``,
       ``{"text": ...}`` or ``{"pk": ...}`` of a ``StructureData`` node) to batch
       process.
   * - ``structure_files``
     - A JSON array of structure files (same ``path``/``text``/``pk`` shape, PK of a
       ``SinglefileData`` node) to batch process; each is parsed for the structures it
       contains (only ``.xyz`` files are currently supported).
   * - ``calculation_parameters``, ``optimisation_parameters``, ``qm_parameters``,
       ``mm_parameters``, ``qmmm_parameters``, ``force_field_file``
     - Shared by every structure in the batch. Same shapes as ``submit_chemshell_job``.
       Note: ``combine_results`` is not supported together with
       ``optimisation_parameters``.
   * - ``output_structure_as_file``
     - If true, return each optimised structure as a ``SinglefileData`` ChemShell
       ``.cjson`` file instead of a ``StructureData`` node.
   * - ``combine_results``
     - If true, combine every structure's results into one extended XYZ file output.
       Default false.
   * - ``num_machines``
     - Machines to run on, for each structure's calculation. Default 1.
   * - ``num_mpiprocs_per_machine``
     - MPI processes per machine, for each structure's calculation. Default 4.
   * - ``label``
     - Optional label to set on the workflow node.
   * - ``description``
     - Optional description to set on the workflow node.

Example arguments, processing two named structures:

.. code-block:: json

   {
     "code": "chemshell@localhost",
     "structures": {
       "water": {"path": "/data/water.cjson"},
       "methane": {"path": "/data/methane.cjson"}
     },
     "qm_parameters": {"parameters": {"theory": "NWChem", "method": "HF", "basis": "3-21G"}}
   }

The tool returns the process PK and the PKs of every input node it created, including
``structures``/``structure_files`` as a mapping from each given key (or list index, as
a string, for ``structure_files``) to its node's PK. If no structure source is given at
all, the call fails with a ``ToolError`` before anything is created.
