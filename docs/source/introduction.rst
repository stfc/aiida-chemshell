Introduction
============

Welcome to the `AiiDA <https://www.aiida.net>`_ ChemShell plugin documentation. This plugin provides an AiiDA workflow plugin for the
`ChemShell <https://chemshell.org/>`_ multiscale computational chemistry package developed at STFC.

Quick Start
-----------

This plugin should be installed alongside the ``aiida-core`` package, typically using the `pip <https://pip.pypa.io/en/stable/>`_ python package manager.
The latest release is available from PyPI:

.. code-block:: bash

    pip install aiida-chemshell

Alternatively, to install the current development version from a local checkout of the source:

.. code-block:: bash

    cd aiida-chemshell
    pip install .

Requirements
~~~~~~~~~~~~~

To use this plugin a configured AiiDA profile and computer are required, as well as a compiled ChemShell
executable. For more information on how to set up AiiDA refer to the `AiiDA documentation <https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html>`_.

Setup
~~~~~

To configure the ChemShell plugin you will need to create a new AiiDA code object. The following is an example 
of a basis YAML configuration file for a ChemShell code:

.. code-block:: yaml 

    label: chemshell 
    description: ChemShell 
    computer: localhost 
    filepath_executable: chemsh.x 
    default_calc_job_plugin: chemshell 
    use_double_quotes: false
    with_mpi: true 
    prepend_text: ""
    append_text: "" 

Write this to a file named ``chemshell.yml`` ensuring the value for ``computer`` matches the label of your
computer configured in the previous step. The code can then be configured by running:

.. code-block:: bash

    verdi code create core.code.installed --config chemshell.yml -n

If successful, this will create a new AiiDA code with the label ``chemshell`` which can then be used to run
ChemShell calculations within the AiiDA framework.

ChemShell with MPI
~~~~~~~~~~~~~~~~~~~

The configuration of an MPI enabled ChemShell code differs slightly depending on whether the code instance
uses the ``chemsh`` wrapper script or the ``chemsh.x`` executable directly:

* If you have a parallel (MPI enabled) build of ChemShell but are using the ``chemsh`` wrapper executable,
  set ``with_mpi: false``. The underlying plugin will handle how to invoke the MPI runtime.
* If you are calling the ``chemsh.x`` executable directly, then ``with_mpi: true`` is required to run
  parallel jobs.

The number of MPI processes is taken from the ``metadata.options.resources`` of the calculation, defaulting
to one machine with four processes per machine (see :doc:`inputs`).

.. seealso::

   :doc:`inputs` for the full list of calculation inputs, parameters and outputs, and :doc:`examples`
   for complete runnable scripts.

