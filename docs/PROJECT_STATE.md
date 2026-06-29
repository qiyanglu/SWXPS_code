# PROJECT_STATE

## Current state

SWANX uses `swanx` as the only supported Python namespace. The early `swxps`
namespace was removed before public release and is not an active compatibility
surface.

The primary human-editable project workflow is:

```text
swanx init my_project
        -> edit my_project/project.yaml
        -> python my_project/run_project.py
        -> my_project/runs/<project_name>_<timestamp>/ report folder
```

For custom Python workflows, the maintained object flow is:

```text
OPC + IMFP + optional experimental curves
        -> swanx.io
        -> SimulationStack / CoreLevelRequest / ReflectivityData / RockingCurveData
        -> simulation + fitting + diagnostics
```

Starter data are packaged with `swanx.project` for `swanx init` and are also
mirrored in the repository under `data/OPC/`, `data/IMFP/`, and `data/curves/`.
Maintained examples use the synthetic C/LaNiO3/SrTiO3 (C/LNO/STO) benchmark CSV
when they need reflectivity and rocking-curve data.

## Implemented workflow

- `swanx.project` validates and runs YAML ProjectSpec files.
- `swanx init my_project` creates `project.yaml`, `run_project.py`, a project
  README, a project-local `synthetic_residual_factory.py`, and by default a
  local `data/` copy of packaged C/LaNiO3/SrTiO3 starter data. The default
  project runs a JAX least-squares fit against the packaged synthetic
  reflectivity and four rocking-curve datasets.
- `swanx init --template minimal`, `--template multilayer`, and
  `--template fit-demo` generate beginner starters for the default fitting
  workflow, a simulation-only repeated multilayer, and an explicit fitting
  starter alias.
- `--copy-example-data` creates a self-contained copy from a chosen data root;
  `--data-root` points at another tutorial data root and writes relative paths
  when possible.
- `swanx inspect ...`, `swanx validate ...`, and `swanx run ...` are thin CLI
  wrappers for review, validation, and automation.
- PyYAML is optional via `python -m pip install -e ".[project]"`.
- `templates/project_minimal.yaml` and `templates/run_project.py` remain a
  repository-local JAX least-squares fitting starter.
- `docs/projectspec_reference.md` is the detailed YAML ProjectSpec reference,
  and `examples/01_quickstart_projectspec/` contains copy-pasteable ProjectSpec
  examples.
- `examples/` is organized as a user learning path: ProjectSpec quickstarts,
  experimental-data loading, compact Python API scripts, fitting examples, and
  advanced low-level visualizations. All maintained examples share the
  synthetic C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 case used by the benchmark
  folder.
- `swanx.io` reads OPC, IMFP, reflectivity, and rocking-curve files and builds
  `SimulationStack` and `CoreLevelRequest` objects from material tables.
- `swanx.preprocessing` owns rocking-curve normalization algorithms.
- `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`; maintained fitting backends live under `swanx.fitting.bo`, `swanx.fitting.jax_gradient`, and `swanx.fitting.jax_least_squares`.
- `swanx.io.__all__` is narrow and explicit; it does not export preprocessing
  functions or legacy flat helpers.

## YAML ProjectSpec notes

The current YAML ProjectSpec supports sections for `project`, `settings`, `materials`,
`parameters`, `stack`, `core_levels`, `datasets`, and `report`. The required
sections are `project`, `settings`, `materials`, `stack`, and `core_levels`;
`parameters`, `datasets`, and `report` default to empty mappings.

Supported YAML workflow features include:

- stable concrete stack layer IDs;
- layer tags and explicit core-level selection by `layer_ids`, `tags`, or
  `all: true`;
- compact repeat blocks for multilayers;
- inline parameter references and AST-whitelisted arithmetic expressions;
- polarization strings `"s"`, `"p"`, and `"unpolarized"`;
- project-local default output folders and a simple Markdown `report.md`;
- opt-in progress messages for `run_project(..., progress=True)`, enabled by
  `swanx run` and generated beginner scripts;
- per-plot skipped-output notes and experimental-overlay notes in `report.md`;
- compound reflectivity-plus-rocking-curve overview plots with incident-angle
  labels and no default residual PNG;
- method-aware plot filenames: fitting runs write `fit_overview.png`,
  `reflectivity_fit.png`, and `rocking_curves_fit.png`; `simulate_only` runs
  write `simulation_overview.png`, `reflectivity_simulation.png`, and
  `rocking_curves_simulation.png`;
- stack schematic plots for all run methods;
- least-squares convergence, parameter-range, and correlation plot images when
  diagnostics are available;
- Bayesian-optimization convergence and surrogate-slice plots when diagnostics
  are available;
- optional dataset weights/log floors, off-peak RC masks, and fixed-grid slicing
  settings that pass through to the existing `FittingProblem` APIs;
- complete `simulate_only` report output without best-fit parameter tables;
- method-specific report writers for least-squares, gradient, and BO result
  objects.

All thickness, roughness, depth, and IMFP values are in Angstrom. In YAML,
`roughness_A` on layer j means roughness/interdiffusion at the upper interface
of layer j, i.e. the interface between layer j-1 and layer j. `repeat_index`
is 1-based inside repeat blocks.

## API notes

- Beginner project runs should start with `swanx init my_project` followed by
  `python my_project/run_project.py`; advanced scripts can call
  `from swanx.project import run_project`.
- Custom simulations can start with `import swanx as sx`.
- OPC files are interpolated at photon energy.
- IMFP files are interpolated at `E_kin = h nu - E_B`.
- `RockingCurveRequest` does not read files directly.
- Unified slicing is the default high-level simulation path.
- `ReflectivityRequest`, `RockingCurveRequest`, and `FittingProblem` support
  `polarization="s"` by default, `polarization="p"`, and mixed dictionaries
  such as `{"s": 0.7, "p": 0.3}`.
- JAX least-squares/autodiff is the recommended fitting path for differentiable
  fixed-shape workflows; BO remains an optional global black-box baseline.
- YAML ProjectSpec fitting still requires user-provided factories for
  `jax_least_squares` and `jax_gradient`; no automatic no-code JAX residual
  builder is implemented. The packaged init starter includes an explicit
  factory for the synthetic C/LaNiO3/SrTiO3 case.
- ProjectSpec v1.3 package layout cleanup moves maintained backend
  implementations under `swanx.fitting` and report implementations under
  `swanx.project.reporting`; root backend modules and `swanx.project.reports`
  remain compatibility shims.

## Repository policy

- `src/swanx/` is the maintained package and only supported Python namespace.
- `tests/` contains regression tests.
- `examples/` contains compact tutorials built around the synthetic
  C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 benchmark case.
- `templates/` contains editable ProjectSpec starter files.
- `case_studies/` is local/private experimental input and runner space ignored
  by Git.
- `benchmarks/` contains synthetic fitting and performance benchmarks.
- `runs/` and `archive/` are local generated/superseded outputs ignored by Git.
- `docs/history/` contains archived historical handoffs and may intentionally
  mention old paths or retired namespaces.

## Latest validation

Run these before handing off substantial changes:

```bash
python -m pytest tests/test_project_workflow.py -q
python -m pytest -q
```

ProjectSpec smoke checks:

```bash
swanx init runs/projectspec_smoke
python runs/projectspec_smoke/run_project.py
swanx inspect runs/projectspec_smoke/project.yaml
swanx validate runs/projectspec_smoke/project.yaml
```

Default init JAX-fit smoke validation completed on 2026-06-29: a fresh
`swanx init` project loaded `synthetic_residual_factory.py`, ran
`jax_least_squares`, wrote `fit/best_parameters.csv`, and produced fit-named
plots. A simulation-only ProjectSpec smoke wrote `simulation_overview.png`,
`reflectivity_simulation.png`, and `rocking_curves_simulation.png`. Focused
ProjectSpec workflow tests and the full suite passed afterward; the full suite
kept its expected xfail and one existing diagnostics warning.

ProjectSpec rocking-curve normalization fix completed on 2026-06-29: configured
`rocking_curve_offpeak_mask` now normalizes experimental rocking-curve datasets
with the same off-peak denominator used for simulated curves, and the packaged
C/LaNiO3/SrTiO3 JAX starter residual uses `problem.offpeak_mask` instead of a
hard-coded peak window. Re-running `myproject/run_project.py` reduced the final
objective from `0.0029710860635918292` in `myproject_20260629_194603` to
`8.906614807793117e-08` in `myproject_20260629_195454`; full validation passed
with `250 passed, 1 xfailed`.
