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

Tutorial starter data are packaged with `swanx.project` for `swanx init` and are
also mirrored in the repository under `data/OPC/`, `data/IMFP/`, and
`data/curves/` for examples.

## Implemented workflow

- `swanx.project` validates and runs YAML ProjectSpec v1.2 files.
- `swanx init my_project` creates `project.yaml`, `run_project.py`, a project
  README, and by default a local `data/` copy of packaged minimal tutorial data.
- `swanx init --template minimal`, `--template multilayer`, and
  `--template fit-demo` generate beginner starters for simulation-only,
  repeated multilayers, and dataset/fitting-placeholder workflows.
- `--copy-example-data` creates a self-contained copy from a chosen data root;
  `--data-root` points at another tutorial data root and writes relative paths
  when possible.
- `swanx inspect ...`, `swanx validate ...`, and `swanx run ...` are thin CLI
  wrappers for review, validation, and automation.
- PyYAML is optional via `python -m pip install -e ".[project]"`.
- `templates/project_minimal.yaml` and `templates/run_project.py` remain a
  repository-local simulation-only starter.
- `swanx.io` reads OPC, IMFP, reflectivity, and rocking-curve files and builds
  `SimulationStack` and `CoreLevelRequest` objects from material tables.
- `swanx.preprocessing` owns rocking-curve normalization algorithms.
- `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`.
- `swanx.io.__all__` is narrow and explicit; it does not export preprocessing
  functions or legacy flat helpers.

## ProjectSpec v1.2 notes

ProjectSpec v1.2 supports sections for `project`, `settings`, `materials`,
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
- per-plot skipped-output notes and experimental-overlay notes in `report.md`;
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
- ProjectSpec v1.2 still requires user-provided factories for
  `jax_least_squares` and `jax_gradient`; no automatic no-code JAX residual
  builder is implemented.

## Repository policy

- `src/swanx/` is the maintained package and only supported Python namespace.
- `tests/` contains regression tests.
- `examples/` contains compact tutorials.
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

ProjectSpec v1.2 validation completed on 2026-06-27 with the focused workflow
tests and full pytest suite passing; the full suite keeps its expected xfail.
Exact counts are intentionally not pinned here because they become stale quickly.
