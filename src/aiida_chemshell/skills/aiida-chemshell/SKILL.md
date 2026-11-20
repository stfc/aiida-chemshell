---
name: aiida-chemshell
description: >-
  Set up, submit, and troubleshoot ChemShell multiscale computational chemistry
  calculations through AiiDA using the aiida-chemshell plugin. Use when the user wants to
  run a ChemShell QM, MM or QM/MM single point energy, geometry optimisation, NEB,
  isolated-atom-energies, batch, or vibrational frequency job from AiiDA, build a
  ChemShell calcjob's parameter inputs, interpret aiida-chemshell's calcjob or workflow
  outputs or exit codes, choose one of its WorkChains (`chemshell.opt`,
  `chemshell.atomic_energies`, `chemshell.batch`), or asks about the aiida_chemshell.mcp
  tools (describe_chemshell_calcjob, submit_chemshell_job,
  describe_chemshell_opt_workflow, submit_chemshell_opt_workflow,
  describe_chemshell_atomic_energies_workflow,
  submit_chemshell_atomic_energies_workflow, describe_chemshell_batch_workflow,
  submit_chemshell_batch_workflow).
---

# aiida-chemshell

`aiida-chemshell` wraps the [ChemShell](https://chemshell.org/) multiscale computational
chemistry package as an AiiDA `CalcJob` (entry point `chemshell`), so ChemShell runs get
full AiiDA data provenance.

## Prerequisites

An AiiDA profile, a configured `Computer`, and a `Code` with
`default_calc_job_plugin: chemshell` are required before anything can be submitted. If
none exists yet, configure one with a YAML file and `verdi code create
core.code.installed`:

```yaml
label: chemsh
description: ChemShell
computer: localhost
filepath_executable: /absolute/path/to/chemsh
default_calc_job_plugin: chemshell
use_double_quotes: false
with_mpi: false
prepend_text: ''
append_text: ''
```

```bash
verdi code create core.code.installed --config chemshell.yml -n
```

The code's `with_mpi` setting depends on the executable: use `with_mpi: false` for the
`chemsh` wrapper (the plugin handles MPI invocation itself), or `with_mpi: true` only
when the code points directly at `chemsh.x`.

## Two ways to submit a job

**Prefer the MCP tools if connected to an `aiida-mcp` server.** There are four
describe/submit tool pairs, one for the raw calcjob and one per WorkChain (see
"WorkChains" below) — pick the narrowest one that fits the task instead of always
reaching for the raw calcjob:

| Task | Tools |
|---|---|
| One calculation; you'll handle everything else yourself | `describe_chemshell_calcjob` / `submit_chemshell_job` |
| Geometry optimisation (plus isolated-atom energies, optionally vibrational analysis) | `describe_chemshell_opt_workflow` / `submit_chemshell_opt_workflow` |
| Isolated-atom reference energies for every element in a structure | `describe_chemshell_atomic_energies_workflow` / `submit_chemshell_atomic_energies_workflow` |
| The same calculation settings over several independent structures (a trajectory, a set of `StructureData` nodes, or a batch of structure files) | `describe_chemshell_batch_workflow` / `submit_chemshell_batch_workflow` |

Call the matching `describe_*` tool first to confirm current inputs/outputs/exit codes,
then the matching `submit_*` tool, then follow up with aiida-mcp's own `get_process` /
`get_process_report` to poll status and see output. `structure`, `structure2`, and
`force_field_file` each take exactly one of `path` (file on the server's machine), `text`
(raw file contents), or `pk` (existing node); the `*_parameters` inputs
(`calculation_parameters`, `optimisation_parameters`, `qm_parameters`, `mm_parameters`,
`qmmm_parameters`) take `parameters` (a dict) instead. The three workflow `submit_*`
tools take the same shaped arguments as `submit_chemshell_job` (see each tool's own
schema for the exact set — e.g. the batch tool takes `trajectory`/`structures`/
`structure_files` instead of a single `structure`) and `label`/`description` there set
the *workflow* node's label/description, not any individual calculation step's.

**Otherwise, write an AiiDA script** using the builder API. A QM single point energy
calculation:

```python
from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, SinglefileData, load_code

load_profile("username")

builder = load_code("chemsh").get_builder()
builder.structure = SinglefileData(file="/absolute/path/to/water.cjson")
builder.qm_parameters = Dict({"theory": "NWChem", "method": "HF", "basis": "3-21G"})
builder.calculation_parameters = Dict({"gradients": False, "hessian": False})

results, node = run.get_node(builder)
```

A combined QM/MM geometry optimisation additionally needs `mm_parameters`,
`force_field_file` (required together — one without the other is a validation error),
and `optimisation_parameters` (which, if supplied, switches the job from a single point
energy to a geometry optimisation):

```python
from aiida import load_profile
from aiida.engine import run
from aiida.orm import Dict, SinglefileData, load_code

load_profile("username")

builder = load_code("chemsh").get_builder()
builder.structure = SinglefileData(file="/absolute/path/to/h2o_dimer.cjson")
builder.qm_parameters = Dict({"theory": "NWChem", "method": "DFT", "basis": "6-31G"})
builder.mm_parameters = Dict({"theory": "DL_POLY"})
builder.force_field_file = SinglefileData(file="/absolute/path/to/h2o_dimer.ff")
builder.optimisation_parameters = Dict({"algorithm": "lbfgs", "maxcycle": 100})

results, node = run.get_node(builder)
```

`structure` also accepts a `StructureData` or `TrajectoryData` node instead of a
pre-formatted file (`.xyz`, `.pun` or `.cjson`); when it is a `TrajectoryData`, also set
`structure_index` to select which frame to use. `structure2` (same accepted types as
`structure`, minus `TrajectoryData`) supplies a second structure, e.g. the final image
for a NEB optimisation.

## Building the parameter dicts

- `calculation_parameters` (Task object): `gradients`, `hessian` — both `bool`.
- `optimisation_parameters` (geometry optimisation task): `maxcycle`, `maxene`,
  `coordinates`, `algorithm`, `trust_radius`, `maxstep`, `tolerance`, `neb` (one of
  `"no"`, `"free"`, `"frozen"`, `"perpendicular"` — set alongside `structure2` to run a
  NEB calculation), `nimages`, `nebk`, `dimer`, `delta`, `tsrelative`, `thermal` (set
  `True` to run a vibrational frequency analysis instead of saving an optimised
  geometry), `save_path` (set `True` to also retrieve a per-step trajectory and forces).
- `qm_parameters` (QM Theory object): requires `theory` (one of `CASTEP`, `CP2K`,
  `DFTBplus`/`DFTBP`, `FHIaims`/`FHI_AIMS`, `GAMESS_UK`, `Gaussian`, `LSDalton`, `MNDO`,
  `Molpro`, `NWChem`, `ORCA`, `PySCF`, `TURBOMOLE` — case-insensitive), plus `method`
  (`"HF"` or `"DFT"`), `basis`, `charge`, `functional`, `mult`, `scftype` (one of `RHF`,
  `UHF`, `ROHF`, `RKS`, `UKS`, `ROKS`), `damping`, `diis`, `direct`, `maxiter`, `path`,
  `pseudopotential`, `restart`, `scf`.
- `mm_parameters` (MM interface): requires `theory` (`DL_POLY`, `GULP`, or `NAMD`); the
  remaining valid keys depend on which theory is chosen (e.g. DL_POLY accepts `rcut`,
  `rvdw`, `ewald`, `timestep`, `steps`, ...). Using `mm_parameters` requires
  `force_field_file` to also be set, and vice versa.
- `qmmm_parameters` (QM/MM interface): requires both `qm_parameters` and `mm_parameters`
  to also be set. Supports `qm_region` (list of atom indices in the QM region).

Invalid keys, missing companion inputs (MM without a force field, QM/MM without both QM
and MM), or wrong value types are rejected by the calcjob's input validators before
anything is submitted — read the validation error message, it names the offending key(s)
and the valid set.

## Outputs

Always produced: `energy` (`Float`).

Conditionally produced, depending on which inputs were given:

- `gradients` (`ArrayData`, key `"gradients"` and/or `"hessian"`) — if
  `calculation_parameters.gradients` / `.hessian` is `True`.
- `optimised_structure` (`StructureData`, or `SinglefileData` containing a ChemShell
  `.cjson` file if `output_structure_as_file` is `True`) and `optimisation_path`
  (`ArrayData`, key `"energies"`) — for a geometry optimisation (`optimisation_parameters`
  set, `thermal` falsy, `neb` not one of `"free"`/`"frozen"`/`"perpendicular"`).
- `trajectory_path` (`TrajectoryData`) and `trajectory_force` (`ArrayData`, one
  `(natoms, 3)` array per frame named `Frame_{i}`) — if `optimisation_parameters.save_path`
  is `True`.
- `vibrational_energies` (`Dict`) and `vibrational_modes` (`ArrayData`, key `"Modes"`,
  negative values mean imaginary frequencies) — if `optimisation_parameters.thermal` is
  `True`.
- `neb_path` (`TrajectoryData`) and `neb_info` (`ArrayData`, keys `path_length`,
  `energy`, `work`, `effective_mass`) — if `optimisation_parameters.neb` is one of
  `"free"`, `"frozen"`, `"perpendicular"`.

## Exit codes

| Code | Name | Meaning |
|---|---|---|
| 300 | `ERROR_STDOUT_NOT_FOUND` | The `output.log` ChemShell log file wasn't retrieved — check the job actually ran. |
| 301 | `ERROR_MISSING_FINAL_ENERGY` | The results file didn't contain a final energy; read `output.log` for the reason. |
| 302 | `ERROR_MISSING_OPTIMISED_STRUCTURE_FILE` | A geometry optimisation didn't produce the expected optimised structure / path file. |
| 303 | `ERROR_RESULTS_FILE_NOT_FOUND` | The internal JSON results file wasn't retrieved. |
| 304 | `ERROR_MISSING_GRADIENTS` | `gradients`/`hessian` was requested but not found in the results. |

When a job fails, there is no `output` node for a failed parse step (`output.log` is
only retrieved, not automatically exposed) — use `verdi calcjob outputcat <pk>
output.log` or `verdi calcjob gotocomputer <pk>` to inspect the raw ChemShell log before
guessing further.

The WorkChains below have their own exit codes for workflow-level failures (e.g. a step
submitting with bad inputs, or — for `chemshell.batch` — no structures given at all);
call the matching `describe_chemshell_*_workflow` MCP tool, or inspect
`WorkflowFactory("<entry point>").spec().exit_codes`, for the current list rather than
assuming the numbers above apply. If a workflow step itself (a `chemshell` calcjob it
submitted) fails, check that step's own node and `exit_status` against the calcjob exit
codes above.

## WorkChains

In addition to the low-level `chemshell` calcjob, three WorkChains orchestrate it into
higher-level tasks (load via `WorkflowFactory("<entry point>")`, or submit through the
matching MCP `submit_chemshell_*_workflow` tool if connected to an `aiida-mcp` server —
see "Two ways to submit a job" above):

- `chemshell.opt` (`GeometryOptimisationWorkChain`; MCP: `submit_chemshell_opt_workflow`)
  — runs a geometry optimisation and optionally a follow-up vibrational analysis in one
  WorkChain. Exposes all `chemshell` inputs under a `chemsh` namespace, plus a top-level
  `vibrational_analysis` (`Bool`). Outputs: `final_energy`, `optimised_structure`, and
  (if requested) `vibrational_energies`/`vibrational_modes`. If `qm_parameters` isn't
  given, it defaults to NWChem DFT B3LYP/cc-pvdz.
- `chemshell.atomic_energies` (`IsolatedAtomicEnergiesWorkChain`; MCP:
  `submit_chemshell_atomic_energies_workflow`) — computes the single point energy of
  each unique atomic species in a structure in isolation. Inputs: `structure`,
  `qm_parameters`, `code`. Output: `atom_energies` (`Dict` mapping element symbol to
  energy). Not supported with the `PySCF` QM backend.
- `chemshell.batch` (`BatchProcessWorkChain`; MCP: `submit_chemshell_batch_workflow`) —
  applies the same `chemshell` inputs to a series of structures, given as one of
  `trajectory` (`TrajectoryData`), `structures` (namespace of `StructureData`), or
  `structure_files` (namespace of `SinglefileData`, `.xyz` only). Shared calcjob
  metadata/resources go under a `calc` namespace. Returns exit code 350
  (`ERROR_NO_INPUTS`) if none of the three structure inputs are given; `combine_results`
  (combine every structure's results into one extended XYZ output) isn't supported
  together with `optimisation_parameters`.
