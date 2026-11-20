Agent Skill
===========

`aiida-chemshell` ships an `Agent Skill <https://www.anthropic.com/news/skills>`_ — a
``SKILL.md`` file describing how to set up, submit and troubleshoot ChemShell
calculations with this plugin, for AI assistants (such as Claude Code) that support the
format. It covers configuring a ChemShell code, the two submission paths (a plain AiiDA
script, or the :doc:`mcp` tools — including which of the eight tools to use for a single
calculation versus a geometry optimisation, isolated-atom energies, or a batch run),
building the QM/MM/optimisation parameter dicts, the calcjob's outputs and exit codes,
and the available WorkChains.

It is installed alongside the package at
``aiida_chemshell/skills/aiida-chemshell/SKILL.md``. The ``aiida-chemshell`` command
installs it into your assistant's skills directory:

.. code-block:: bash

   aiida-chemshell skills install

This copies the skill into ``~/.claude/skills/`` by default. Pass a directory to install
somewhere else, e.g. a project-local ``.claude/skills/``:

.. code-block:: bash

   aiida-chemshell skills install .claude/skills
