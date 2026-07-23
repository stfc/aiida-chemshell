Example Calculations
====================

The examples below use the ``chemsh`` code configured in the :doc:`introduction`. The associated structure
and force field files referenced can be found in the ``tests/data/`` folder of the source repository. For a
full description of every available input and its parameters, see the :doc:`inputs` reference.

Each script can be run either as a standalone python file or directly within a ``verdi shell`` python
environment. When running inside a ``verdi shell`` the call to :func:`~aiida.manage.configuration.load_profile`
is not required.

QM Based Single Point Energy
----------------------------

The following script runs a Quantum Mechanics (QM) based single point energy calculation using the NWChem
ChemShell interface.

.. code-block:: python

    from aiida.engine import run
    from aiida.orm import load_code, SinglefileData, Dict
    from aiida import load_profile

    load_profile("user_profile")  # Not required within a verdi shell environment

    builder = load_code("chemsh").get_builder()
    builder.structure = SinglefileData(file="/absolute/path/to/water.cjson")
    builder.qm_parameters = Dict({"theory": "NWChem", "method": "HF", "basis": "3-21G"})
    builder.calculation_parameters = Dict({"gradients": False, "hessian": False})

    results, node = run.get_node(builder)

    print("Final Energy = ", results["energy"].value)

To also retrieve the analytic gradients, set ``gradients`` to ``True`` in the ``calculation_parameters``.
The gradients are returned as an ``ArrayData`` node under the ``gradients`` output (see :doc:`inputs`).

QM/MM Based Geometry Optimisation
---------------------------------

The following script runs a combined QM/MM based geometry optimisation using the NWChem and DL_POLY
ChemShell interfaces. Providing the ``optimisation_parameters`` input is what configures the job as a
geometry optimisation rather than a single point energy calculation.

.. code-block:: python

    from aiida.engine import run
    from aiida.orm import load_code, SinglefileData, Dict
    from aiida import load_profile

    load_profile("user_profile")  # Not required within a verdi shell environment

    builder = load_code("chemsh").get_builder()
    builder.structure = SinglefileData(file="/absolute/path/to/h2o_dimer.cjson")
    builder.qm_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})
    builder.mm_parameters = Dict({"theory": "DL_POLY"})
    builder.force_field_file = SinglefileData(file="/absolute/path/to/h2o_dimer.ff")
    builder.qmmm_parameters = Dict({"qm_region": [0, 1, 2]})
    builder.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcycle": 100})

    results, node = run.get_node(builder)

    print("Final Energy = ", results["energy"].value)
    optimised_structure = results["optimised_structure"]  # SinglefileData (.cjson)

.. note::

   Molecular mechanics requires both an ``mm_parameters`` input **and** a ``force_field_file``. A QM/MM
   calculation additionally requires ``qm_parameters``; the ``qmmm_parameters`` dictionary defines the QM
   region via its ``qm_region`` key. See :doc:`inputs` for the full set of validation rules.
