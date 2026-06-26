# PROJECT_STATE

## Current state

SWANX uses `swanx` as the only supported Python namespace. The early `swxps`
namespace was removed before public release and is not an active compatibility
surface.

The primary human-editable project workflow is:

```text
project.yaml
        -> swanx.project.validate_project / run_project
        -> runs/<project_name>_<timestamp>/ report folder
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
- `templates/project_minimal.yaml` and `templates/run_project.py` provide the
  minimal editable project entry.
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
`parameters`, `stack`, `core_levels`, `datasets`, and `report`.

Supported YAML workflow features include:

- stable concrete stack layer IDs;
- layer tags and core-level selection by `layer_ids` or `tags`;
- compact repeat blocks for multilayers;
- inline parameter references and AST-whitelisted arithmetic expressions;
- polarization strings `"s"`, `"p"`, and `"unpolarized"`;
- complete `simulate_only` report output;
- method-specific report writers for least-squares, gradient, and BO result
  objects.

All thickness, roughness, depth, and IMFP values are in Angstrom. In YAML,
`roughness_A` on layer j means roughness/interdiffusion at the upper interface
of layer j, i.e. the interface between layer j-1 and layer j.

## API notes

- User project runs should start with `from swanx.project import run_project`.
- Custom simulations can start with `import swanx as sx`.
- OPC files are interpolated at photon energy.
- IMFP files are interpolated at `E_kin = h nu - E_B`.
- `RockingCurveRequest` does not read files directly.
- Unified slicing is the default high-level simulation path.
- `ReflectivityRequest`, `RockingCurveRequest`, and `FittingProblem` support
  `polarization="s"` by default, `polarization="p"`, and mixed dictionaries
  such as `{"s": 0.7, "p": 0.3}`.
- JAX least-squares/autodiff is the recommended fitting path for fixed-shape
  workflows; BO remains a baseline.

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
python templates/run_project.py
# completed and wrote a minimal_yaml_project run folder

python -m swanx.cli validate templates/project_minimal.yaml
# Validated templates\project_minimal.yaml

python -m swanx.cli run templates/project_minimal.yaml
# Wrote runs\minimal_yaml_project_<timestamp>

python -m pytest tests/test_project_workflow.py -q --basetemp runs/pytest_project_workflow
# 9 passed

python -m pytest -q --basetemp runs/pytest_project_full
# 229 passed, 1 xfailed
```
