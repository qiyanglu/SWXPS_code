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
data/OPC + data/IMFP + data/curves
        -> swanx.io
        -> SimulationStack / CoreLevelRequest / ReflectivityData / RockingCurveData
        -> simulation + fitting + diagnostics
```

Tutorial data live at:

- `data/OPC/`
- `data/IMFP/`
- `data/curves/`

## Implemented workflow

- `swanx.project` validates and runs YAML ProjectSpec v1 files.
- `swanx init my_project` creates `project.yaml`, `run_project.py`, and a
  project README for the beginner workflow. `--copy-example-data` creates a
  self-contained starter; `--data-root` points at another tutorial data root.
- `templates/project_minimal.yaml` and `templates/run_project.py` provide a
  repository-local simulation-only starter.
- `swanx validate ...` and `swanx run ...` are thin CLI wrappers for automation.
- PyYAML is optional via `python -m pip install -e ".[project]"`.
- `swanx.io` reads OPC, IMFP, reflectivity, and rocking-curve files.
- `swanx.io` builds `SimulationStack` and `CoreLevelRequest` objects from
  material tables.
- `swanx.preprocessing` owns rocking-curve normalization algorithms.
- `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`.
- `swanx.io.__all__` is narrow and explicit; it does not export preprocessing
  functions or legacy flat helpers.

## ProjectSpec v1 notes

ProjectSpec v1 supports sections for `project`, `settings`, `materials`,
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
- JAX least-squares/autodiff is the recommended fitting path for fixed-shape
  workflows; BO remains an optional global black-box baseline/robustness check.

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

```bash
python -m pytest tests/test_project_workflow.py -q
# 20 passed

python -m pytest -q
# 240 passed, 1 xfailed

swanx init runs/v11_smoke_default
python "C:/Users/luqy0/OneDrive - ????/SWXPS_code/runs/v11_smoke_default/run_project.py"
# SWANX results written to: .../runs/v11_smoke_default/runs/v11_smoke_default_20260627_101343
# report.md exists under that output folder

swanx init runs/v11_smoke_copied --copy-example-data
swanx validate runs/v11_smoke_copied/project.yaml
# Validated runs11_smoke_copied\project.yaml
```
