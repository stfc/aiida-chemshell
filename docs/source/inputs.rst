Inputs, Parameters and Outputs
==============================

This page is the reference for the ``chemshell`` calculation
(:class:`~aiida_chemshell.calculations.base.ChemShellCalculation`). It describes every input port, the valid
keys accepted by each dictionary-style input, the outputs that are produced, and the exit codes that may be
returned. For complete runnable scripts see :doc:`examples`, and for the higher level WorkChains see
:doc:`workflows`.

How the calculation type is selected
------------------------------------

There is a single ``chemshell`` calculation plugin. The type of job it performs is inferred from which
inputs are provided, rather than from a dedicated "task" input:

* **Single point energy** — the default when no ``optimisation_parameters`` input is given.
* **Geometry optimisation** — configured whenever the ``optimisation_parameters`` input is provided.
* **Vibrational frequencies / thermochemistry** — a geometry optimisation with ``thermal: True`` in the
  ``optimisation_parameters``.
* **Nudged Elastic Band (NEB)** — a geometry optimisation with ``neb`` set to ``free``, ``frozen`` or
  ``perpendicular`` (an end-point ``structure2`` is also required).

The level of theory is selected independently:

* **QM** — provide ``qm_parameters``.
* **MM** — provide ``mm_parameters`` together with a ``force_field_file``.
* **QM/MM** — provide ``qm_parameters``, ``mm_parameters`` (with a ``force_field_file``) and
  ``qmmm_parameters``.

Input ports
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 22 12 44

   * - Port
     - Type
     - Required
     - Description
   * - ``structure``
     - ``SinglefileData`` | ``StructureData`` | ``TrajectoryData``
     - yes
     - The input structure. A ``SinglefileData`` must contain an ``.xyz``, ``.pun`` or ``.cjson`` file.
   * - ``structure_index``
     - ``Int``
     - no
     - Index of the structure to extract when ``structure`` is a ``TrajectoryData`` containing multiple frames.
   * - ``structure2``
     - ``SinglefileData`` | ``StructureData``
     - no
     - An additional structure, used as the final (end-point) structure for NEB optimisations.
   * - ``calculation_parameters``
     - ``Dict``
     - no
     - Parameters for the single point Task object. See `calculation_parameters`_.
   * - ``optimisation_parameters``
     - ``Dict``
     - no
     - Parameters for a geometry optimisation Task. Providing this input configures a geometry optimisation. See `optimisation_parameters`_.
   * - ``qm_parameters``
     - ``Dict``
     - no
     - Parameters for the QM Theory object. See `qm_parameters`_.
   * - ``mm_parameters``
     - ``Dict``
     - no
     - Parameters for the MM Theory interface. See `mm_parameters`_.
   * - ``force_field_file``
     - ``SinglefileData``
     - no
     - Force field parameter file for the MM interface. Required whenever ``mm_parameters`` is used.
   * - ``qmmm_parameters``
     - ``Dict``
     - no
     - Parameters for the QM/MM interface. See `qmmm_parameters`_.

.. note::

   The following validation rules are enforced across the input namespace:

   * ``mm_parameters`` requires a ``force_field_file`` (and vice versa).
   * ``qmmm_parameters`` requires both ``qm_parameters`` and ``mm_parameters``.

Metadata / resources
~~~~~~~~~~~~~~~~~~~~~

The number of MPI processes is derived from ``metadata.options.resources``, which defaults to:

.. code-block:: python

    builder.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 4,
    }

.. _calculation_parameters:

``calculation_parameters``
--------------------------

Controls a single point energy calculation.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``gradients``
     - ``bool``
     - Compute the first derivatives (gradients) of the energy. Returned in the ``gradients`` output.
   * - ``hessian``
     - ``bool``
     - Compute the second derivatives (hessian). Stored alongside the gradients in the ``gradients`` output.

.. _optimisation_parameters:

``optimisation_parameters``
---------------------------

Providing this input configures the job as a geometry optimisation (driven by DL-FIND). All keys are
optional; ChemShell defaults apply to any that are omitted.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``algorithm``
     - ``str``
     - Optimisation algorithm, e.g. ``lbfgs``.
   * - ``coordinates``
     - ``str``
     - Coordinate system used for the optimisation.
   * - ``maxcycle``
     - ``int``
     - Maximum number of optimisation cycles.
   * - ``maxene``
     - ``int``
     - Maximum number of energy evaluations.
   * - ``trust_radius``
     - ``float``
     - Trust radius for the step control.
   * - ``maxstep``
     - ``float``
     - Maximum step size.
   * - ``tolerance``
     - ``float``
     - Convergence tolerance.
   * - ``thermal``
     - ``bool``
     - Perform a vibrational frequency / thermochemical analysis on the structure instead of an optimisation.
   * - ``save_path``
     - ``bool``
     - Retrieve the optimisation path (per-step geometries and forces) as trajectory outputs.
   * - ``neb``
     - ``str``
     - Enable an NEB calculation; one of ``free``, ``frozen`` or ``perpendicular``. Requires ``structure2``.
   * - ``nimages``
     - ``int``
     - Number of images along the NEB path.
   * - ``nebk``
     - ``float``
     - NEB spring force constant.
   * - ``dimer``
     - ``bool``
     - Enable a dimer transition-state search.
   * - ``delta``
     - ``float``
     - Dimer separation.
   * - ``tsrelative``
     - ``bool``
     - Transition-state search option.

.. _qm_parameters:

``qm_parameters``
-----------------

Configures the QM Theory object. The ``theory`` key is required and selects the QM interface; the remaining
keys are passed through to the corresponding ChemShell Theory object.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``theory``
     - ``str``
     - **Required.** The QM interface to use (see `Supported QM theories`_).
   * - ``method``
     - ``str``
     - Electronic structure method, one of ``HF`` or ``DFT``.
   * - ``basis``
     - ``str``
     - Basis set name.
   * - ``functional``
     - ``str``
     - Exchange-correlation functional (for DFT).
   * - ``charge``
     - ``int`` | ``float``
     - Total system charge.
   * - ``mult``
     - ``int`` | ``float``
     - Spin multiplicity.
   * - ``scftype``
     - ``str``
     - SCF type, one of ``RHF``, ``UHF``, ``ROHF``, ``RKS``, ``UKS`` or ``ROKS``.
   * - ``maxiter``
     - ``int``
     - Maximum number of SCF iterations.
   * - ``scf``
     - ``float``
     - SCF convergence threshold.
   * - ``damping``
     - ``bool``
     - Enable SCF damping.
   * - ``diis``
     - ``bool``
     - Enable DIIS convergence acceleration.
   * - ``direct``
     - ``bool``
     - Use a direct SCF.
   * - ``restart``
     - ``bool``
     - Restart from a previous calculation.
   * - ``pseudopotential``
     - ``str`` | ``dict``
     - Pseudopotential specification.
   * - ``path``
     - ``str``
     - Path option passed to the interface.

Supported QM theories
~~~~~~~~~~~~~~~~~~~~~~~

The following values are accepted for the ``theory`` key (case-insensitive). They correspond to the members
of :class:`~aiida_chemshell.utils.ChemShellQMTheory`:

``CASTEP``, ``CP2K``, ``DFTBP``, ``FHI_AIMS``, ``GAMESS_UK``, ``GAUSSIAN``, ``LSDALTON``, ``MNDO``,
``MOLPRO``, ``NWCHEM``, ``ORCA``, ``PYSCF`` and ``TURBOMOLE``.

.. _mm_parameters:

``mm_parameters``
-----------------

Configures the MM Theory interface. The ``theory`` key is required and selects the MM backend. A
``force_field_file`` input must always accompany ``mm_parameters``.

All backends accept the common keys below; each backend additionally accepts a set of backend-specific keys.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``theory``
     - ``str``
     - **Required.** The MM interface, one of ``DL_POLY``, ``GULP`` or ``NAMD``.


Backend-specific keys include, for example, ``rcut``, ``rvdw``, ``ewald``, ``timestep`` and ``steps`` for
``DL_POLY``; ``molecule`` and ``conjugate`` for ``GULP``; and the extensive NAMD option set (``cutoff``,
``pme``, ``switching`` etc.). See
:meth:`~aiida_chemshell.calculations.base.ChemShellCalculation.get_valid_mm_paramater_keys` for the complete,
authoritative list of keys accepted for each backend.

Supported MM theories correspond to the members of :class:`~aiida_chemshell.utils.ChemShellMMTheory`:
``DL_POLY``, ``GULP`` and ``NAMD``.

.. _qmmm_parameters:

``qmmm_parameters``
-------------------

Configures the QM/MM coupling. Required whenever both ``qm_parameters`` and ``mm_parameters`` are supplied.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``qm_region``
     - ``list[int]``
     - The atom indices that make up the QM region of the QM/MM partition.

Outputs
-------

Which outputs are produced depends on the calculation type and the parameters requested.

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Output
     - Type
     - Produced when / description
   * - ``energy``
     - ``Float``
     - Always. The total energy of the system.
   * - ``gradients``
     - ``ArrayData``
     - When ``gradients`` and/or ``hessian`` are requested. Holds the ``gradients`` and/or ``hessian`` arrays.
   * - ``optimised_structure``
     - ``SinglefileData``
     - After a successful geometry optimisation. A ChemShell structure file containing the optimised geometry.
   * - ``optimisation_path``
     - ``ArrayData``
     - After a geometry optimisation. Per-step values (e.g. energies) along the optimisation.
   * - ``trajectory_path``
     - ``TrajectoryData``
     - When ``save_path: True``. The geometries visited during the optimisation.
   * - ``trajectory_force``
     - ``SinglefileData``
     - When ``save_path: True``. An XYZ-style file of forces at each optimisation step.
   * - ``vibrational_energies``
     - ``Dict``
     - When ``thermal: True``. Thermochemical properties (ZPE, enthalpy, entropy, ...).
   * - ``vibrational_modes``
     - ``ArrayData``
     - When ``thermal: True``. The calculated vibrational modes.
   * - ``neb_path``
     - ``TrajectoryData``
     - After an NEB calculation. The determined reaction pathway.
   * - ``neb_info``
     - ``ArrayData``
     - After an NEB calculation. Per-image data (``path_length``, ``energy``, ``work``, ``effective_mass``).

Exit codes
----------

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Code
     - Label
     - Meaning
   * - ``300``
     - ``ERROR_STDOUT_NOT_FOUND``
     - The ``output.log`` ChemShell output file could not be accessed.
   * - ``301``
     - ``ERROR_MISSING_FINAL_ENERGY``
     - ChemShell did not compute a final energy for the requested task.
   * - ``302``
     - ``ERROR_MISSING_OPTIMISED_STRUCTURE_FILE``
     - The expected optimised structure file was not produced.
   * - ``303``
     - ``ERROR_RESULTS_FILE_NOT_FOUND``
     - The expected ``result.json`` results file was not produced.
   * - ``304``
     - ``ERROR_MISSING_GRADIENTS``
     - The requested gradients or hessian were not computed.
