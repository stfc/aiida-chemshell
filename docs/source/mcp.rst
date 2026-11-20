MCP integration
================

`aiida-mcp <https://github.com/stfc/aiida-mcp>`_ is a
`Model Context Protocol <https://modelcontextprotocol.io>`_ server that lets an MCP
client, such as an AI assistant, inspect an AiiDA database and submit calculations.
``aiida-chemshell`` ships its own integration module, ``aiida_chemshell.mcp``, which
exposes ChemShell as a pair of MCP tools.

Installing the optional ``mcp`` extra:

.. code-block:: bash

   pip install "aiida-chemshell[mcp]"

pulls in aiida-mcp and registers ``aiida_chemshell.mcp`` under its ``aiida_mcp.plugins``
entry point group. Any aiida-mcp server running in the same environment then
automatically exposes ``describe_chemshell_calcjob`` and ``submit_chemshell_job`` tools —
no changes needed in aiida-mcp itself. See aiida-mcp's own documentation for how to run
the server and its other tools.

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
