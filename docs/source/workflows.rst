Workflows
=========

In addition to the low-level ``chemshell`` calculation described in :doc:`inputs`, the plugin provides a set
of AiiDA WorkChains that orchestrate several calculations into higher level tasks. Registered WorkChains can
be loaded through the AiiDA :func:`~aiida.plugins.factories.WorkflowFactory` using their entry point name:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Entry point
     - WorkChain
     - Purpose
   * - ``chemshell.opt``
     - :class:`~aiida_chemshell.workflows.optimisation.GeometryOptimisationWorkChain`
     - Geometry optimisation with optional vibrational analysis.
   * - ``chemshell.atomic_energies``
     - :class:`~aiida_chemshell.workflows.isolated_atoms.IsolatedAtomicEnergiesWorkChain`
     - Isolated single-atom reference energies for each species in a structure.
   * - ``chemshell.batch``
     - :class:`~aiida_chemshell.workflows.batch_calculation.BatchProcessWorkChain`
     - Apply the same calculation inputs to a series of structures.

Geometry Optimisation WorkChain
-------------------------------

``chemshell.opt`` wraps the ``chemshell`` calculation to run a geometry optimisation and, optionally, a
follow-up vibrational frequency analysis on the optimised structure. The full set of ``chemshell`` calculation
inputs is exposed under the ``chemsh`` namespace, so any input documented in :doc:`inputs` can be supplied
there.

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Input
     - Type
     - Description
   * - ``chemsh``
     - namespace
     - All inputs of the ``chemshell`` calculation (``structure``, ``qm_parameters``, ...).
   * - ``vibrational_analysis``
     - ``Bool``
     - If ``True``, run a vibrational frequency analysis on the optimised structure.

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Output
     - Type
     - Description
   * - ``final_energy``
     - ``Float``
     - The final energy of the optimised structure.
   * - ``optimised_structure``
     - ``SinglefileData``
     - The optimised geometry.
   * - ``vibrational_energies``
     - ``Dict``
     - Thermochemical properties (only when ``vibrational_analysis`` is ``True``).
   * - ``vibrational_modes``
     - ``ArrayData``
     - Vibrational modes (only when ``vibrational_analysis`` is ``True``).

If ``qm_parameters`` are not supplied in the ``chemsh`` namespace, the WorkChain falls back to a default
NWChem DFT setup (``B3LYP`` / ``cc-pvdz``).

.. code-block:: python

    from aiida.engine import run
    from aiida.orm import load_code, SinglefileData, Dict, Bool
    from aiida.plugins import WorkflowFactory
    from aiida import load_profile

    load_profile("user_profile")

    GeometryOptimisation = WorkflowFactory("chemshell.opt")

    builder = GeometryOptimisation.get_builder()
    builder.chemsh.code = load_code("chemsh")
    builder.chemsh.structure = SinglefileData(file="/absolute/path/to/water.cjson")
    builder.chemsh.qm_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})
    builder.chemsh.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcycle": 100})
    builder.vibrational_analysis = Bool(True)

    results, node = run.get_node(builder)

    print("Final energy      = ", results["final_energy"].value)
    print("Thermochemistry   = ", results["vibrational_energies"].get_dict())

Isolated Atomic Energies WorkChain
----------------------------------

``chemshell.atomic_energies`` computes the single point energy of each unique atomic species in a structure
as an isolated atom.

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Input
     - Type
     - Description
   * - ``structure``
     - ``SinglefileData`` | ``StructureData``
     - The structure whose unique species should be enumerated.
   * - ``qm_parameters``
     - ``Dict``
     - QM theory parameters used for the single-atom calculations.
   * - ``code``
     - ``Code``
     - The ChemShell code instance.

The single output ``atom_energies`` is a ``Dict`` mapping each element symbol to its isolated-atom energy.

.. code-block:: python

    from aiida.engine import run
    from aiida.orm import load_code, SinglefileData, Dict
    from aiida.plugins import WorkflowFactory
    from aiida import load_profile

    load_profile("user_profile")

    AtomicEnergies = WorkflowFactory("chemshell.atomic_energies")

    builder = AtomicEnergies.get_builder()
    builder.code = load_code("chemsh")
    builder.structure = SinglefileData(file="/absolute/path/to/water.cjson")
    builder.qm_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})

    results, node = run.get_node(builder)
    print(results["atom_energies"].get_dict())

.. note::

   Isolated atom calculations are not supported with the ``PySCF`` QM backend.

Batch processing
----------------

``chemshell.batch`` (:class:`~aiida_chemshell.workflows.batch_calculation.BatchProcessWorkChain`) applies the
same set of ``chemshell`` inputs to a series of structures.

The structures to process can be supplied in any of three ways (at least one is required):

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Input
     - Type
     - Description
   * - ``trajectory``
     - ``TrajectoryData``
     - A trajectory whose frames are each processed as a separate calculation.
   * - ``structures``
     - namespace of ``StructureData``
     - A dictionary of ``StructureData`` nodes to process.
   * - ``structure_files``
     - namespace of ``SinglefileData``
     - A dictionary of structure files. Currently only ``.xyz`` trajectory files are parsed; others are skipped.

All of the ``chemshell`` calculation inputs are exposed (excluding ``structure``, which is provided through
the batch-specific inputs above), so the QM/MM parameters, optimisation parameters, etc., are shared across
every structure in the batch. The calculation ``metadata`` is exposed under a dedicated ``calc`` namespace:
any options set via ``calc.metadata.options`` (for example the computational resources and number of MPI
processes) are forwarded to every job in the batch. If none of the three structure inputs are provided the
WorkChain returns exit code ``350`` (``ERROR_NO_INPUTS``).

.. code-block:: python

    from aiida.engine import run
    from aiida.orm import load_code, StructureData, Dict
    from aiida.plugins import WorkflowFactory
    from aiida import load_profile

    load_profile("user_profile")

    BatchProcess = WorkflowFactory("chemshell.batch")

    builder = BatchProcess.get_builder()
    builder.code = load_code("chemsh")
    builder.qm_parameters = Dict({"theory": "NWChem", "method": "HF", "basis": "3-21G"})
    builder.structures = {
        "molecule_a": StructureData(...),
        "molecule_b": StructureData(...),
    }
    # Resources set here are applied to every calculation in the batch
    builder.calc.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 8,
    }

    results, node = run.get_node(builder)
