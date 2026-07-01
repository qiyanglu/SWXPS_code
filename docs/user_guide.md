# User Guide

SWANX (**S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray
spectroscopy) is imported as `swanx`. The most beginner-friendly workflow is a
YAML ProjectSpec: edit one `project.yaml`, then run a small script or CLI
command.

## Overview

A SWANX project gathers the inputs needed for multilayer SW-XPS work: material
optical constants, electron attenuation data, a stack model, core levels, and
optional experimental curves. SWANX turns those into simulations or fits and
writes a project-local report folder.

```text
OPC + IMFP + stack + core levels + optional datasets
  -> simulation / fitting
  -> report.md + CSVs + plots
```

Use YAML when you want a reproducible project file. Use the Python API when you
need custom model code, custom fixed-shape JAX residuals, or a larger scripted
workflow.

## Core Concepts

- **Material**: a label such as `LNO` for LaNiO3 or `STO` for SrTiO3 that
  connects stack layers to OPC and IMFP files.
- **Stack layer**: one physical layer in order from vacuum to substrate. Each
  concrete layer has a stable `id`.
- **Optical constants (OPC)**: photon-energy-dependent `delta` and `beta` values
  for the refractive index convention `n = 1 - delta + i beta`.
- **IMFP**: inelastic mean free path, interpolated at photoelectron kinetic
  energy.
- **Core level**: an emitted photoelectron line such as `La 4d` with a binding
  energy and emitting-layer selector.
- **Emitting layer selector**: `emit_from.layer_ids`, `emit_from.tags`, or
  `emit_from.all: true`; material-only emission selection is not used.
- **Reflectivity curve**: reflected X-ray intensity versus incident angle.
- **Rocking curve**: core-level photoemission intensity versus incident angle,
  written conceptually as $$I_\mathrm{core}(\theta)$$.
- **Fit parameter**: a named scalar with `initial`, `lower`, and `upper` bounds
  when varied, or a constant `value` when fixed.
- **Report folder**: the project-local output directory containing `report.md`,
  CSVs, optional plots, and optimizer outputs.

## Quickstart: Fitting Starter Project

Create a self-contained starter project:

```bash
python -m pip install -e ".[project,least-squares,plot]"
swanx init my_project
python my_project/run_project.py
```

The generated project contains:

```text
my_project/
  project.yaml
  run_project.py
  README.md
  data/
```

The default generated YAML uses `run.mode: "jax_least_squares"`. It builds the
C/LaNiO3/SrTiO3 stack, loads the packaged synthetic reflectivity and rocking
curves, fits selected stack parameters, simulates the best-fit curves, and
writes a report.

Rocking curves use edge-polynomial normalization by default:

```yaml
settings:
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
```

This fits a second-order background to the first and last 10 percent of each
rocking curve and applies the same normalization to experimental data,
simulation-only curves, BO/generic fitting, and auto fixed-grid JAX
least-squares.

Typical first edits in `project.yaml` are:

- change `project.name`;
- edit `settings.photon_energy_ev` and the angle grid;
- update `materials` paths;
- edit `stack` thicknesses and roughnesses;
- choose which layers emit each `core_level`;
- set `run.outputs.plots: true` when matplotlib is installed.

Stack `thickness_A` and `roughness_A` fields can be numbers, parameter
references such as `$lno_thickness`, or safe expressions. Inside repeat blocks,
`repeat_index` is 1-based and `repeat_index0` is available for zero-based
formulas. Expressions may use arithmetic plus `min`, `max`, `sqrt`, `erf`,
`linear_map`, and `transition_erf`; arbitrary Python calls are rejected.

For a forward-modeling-only starter, use:

```bash
swanx init my_project --template simulate
```

`--template fit` is the preferred explicit fitting starter name. `minimal` and
`fit-demo` remain fitting starter aliases, and `multilayer` remains an alias for
the simulation-only starter.

More complete syntax is documented in
[projectspec_reference.md](projectspec_reference.md).
Copy-pasteable starter YAML files are in
[../examples/01_quickstart_projectspec](../examples/01_quickstart_projectspec).
The runnable example in
[../examples/04_fitting/projectspec_jax_least_squares](../examples/04_fitting/projectspec_jax_least_squares)
shows the same fitting scope as the default init project without generating a
new project folder first.

## Simulate Only And Overlay Data

To compare simulations with data while still avoiding fitting, keep:

```yaml
run:
  mode: "simulate_only"
```

Legacy `settings.fit_method: "simulate_only"` is still accepted.

and add datasets:

```yaml
datasets:
  reflectivity:
    path: "../../benchmarks/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv"
    name: "Reflectivity"
    angle_column: "angle_deg"
    intensity_column: "reflectivity"
    weight: 1.0
    log_floor: 1.0e-12

  rocking_curves:
    - path: "../../benchmarks/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv"
      name: "La 4d"
      angle_column: "angle_deg"
      intensity_column: "la4d_rc"
      weight: 1.0
```

Dataset paths are resolved relative to `project.yaml`, not the process current
working directory. The maintained examples use the synthetic C/LaNiO3/SrTiO3
(C/LNO/STO) benchmark CSV as a stand-in for measured data; replace the path and
column names with your own measurements when needed. Rocking-curve names should
match core-level names when you want overlays and residuals. If a rocking-curve
dataset omits `normalization`, it uses `settings.normalization`; setting a
dataset normalization to empty/null leaves that experimental curve as read.
`settings.rocking_curve_offpeak_mask` remains available for mean-normalized
workflows, but it is not needed for the default edge-polynomial normalization.

With datasets present, `simulate_only` reports can include experimental data
CSVs, residuals, and plot overlays, but they still do not write
`fit/best_parameters.csv` because no fitting was performed.

## Fit Workflow

JAX least-squares is the recommended fitting path for differentiable fixed-shape
workflows. For YAML stacks with fixed topology, use the internal fixed-grid
residual builder:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 100
    estimate_covariance: true
  outputs:
    plots: true
    identifiability: true
```

This path reads the stack, expressions, parameters, datasets, core levels, and
`settings.slicing.mode: "fixed_grid"` directly from `project.yaml`; no
project-local factory script is needed. For unusual fixed-shape residuals that
cannot be expressed by the ProjectSpec stack, advanced users can still set
`run.optimizer.residual_function_factory: "module:function"`. SWANX does not
fall back to Bayesian optimization when a JAX configuration is invalid.

Advanced fitting projects can set separate
`settings.reflectivity_angle_offset_parameter` and
`settings.rocking_curve_angle_offset_parameter` when reflectivity and
rocking-curve scans need independent angular offsets. Edge-polynomial
rocking-curve normalization is the ProjectSpec default and can be controlled with
`settings.normalization_edge_fraction` and
`settings.normalization_polynomial_order`.

Bayesian optimization is available as an optional global black-box baseline or
robustness check:

```yaml
run:
  mode: "bayesian_optimization"
  optimizer:
    n_calls: 40
    n_initial_points: 10
    random_state: 0
```

BO is not the default and not a fallback. It requires the optional fitting extra
and reports BO-specific evaluations and best-so-far CSVs; it does not write
least-squares covariance or correlation outputs.

## How To Inspect And Validate

Use inspection before running a new project:

```bash
swanx inspect project.yaml
```

`inspect` prints the project name, output preview, material paths, layer count,
core levels, datasets, varying parameters, optional dependency status, fitting
callback status, and a Doctor section. The Doctor section checks material and
dataset file status, plotting consequences if matplotlib is missing,
least-squares and BO optional dependencies, and auto-fixed-grid readiness. It
does not run simulations or fitting.

Validate the YAML and referenced files:

```bash
swanx validate project.yaml
```

Common validation errors include:

- missing OPC file for a non-vacuum stack material;
- missing IMFP file for a material that emits a core level;
- duplicate layer `id`;
- unknown layer tag or layer id in `emit_from`;
- missing `emit_from`;
- unknown parameter name in an expression;
- unsafe or unknown function in an expression;
- dataset path not found;
- missing fixed-grid slicing for the auto JAX residual.

## How To Read Outputs

A default run folder looks like:

```text
runs/<project_name>_<timestamp>/
  report.md
  input/
  resolved/
  simulation/
  data/
  fit/
  plots/
  optimizer/
```

Important files:

- `report.md`: human-readable summary, output list, plot notes, and skipped
  optional outputs.
- `resolved/stack_resolved.csv`: expanded stack after parameter expressions and
  repeat blocks are resolved.
- `resolved/materials_resolved.csv`: material labels and whether OPC/IMFP tables
  were loaded.
- `resolved/core_levels_resolved.csv`: core-level settings and emitting layer
  indices.
- `simulation/reflectivity_simulated.csv`: simulated reflectivity versus
  incident angle.
- `simulation/rocking_curves_simulated.csv`: simulated rocking curves for each
  core level.
- `data/reflectivity_experimental.csv` and `data/rocking_curves_experimental.csv`:
  copied experimental data, only when datasets exist.
- `fit/residuals.csv`: residuals when experimental data exist.
- `fit/best_parameters.csv`: fitted parameter summary for fitting methods, not
  for `simulate_only`.
- `optimizer/least_squares/`: status, residual vector, Jacobian, covariance,
  correlation, active bounds, convergence history, and parameter uncertainty
  when available.
- `optimizer/gradient/`: status, objective history, parameter history, gradient
  norms, and final gradient when available.
- `optimizer/bayesian/`: evaluations, best-so-far, parameter samples, and stage
  summary.
- `plots/`: overview, reflectivity, rocking curves, stack schematic, and
  backend-specific diagnostics when available. Fitting runs use
  `fit_overview.png`, `reflectivity_fit.png`, and `rocking_curves_fit.png`;
  `simulate_only` runs use `simulation_overview.png`,
  `reflectivity_simulation.png`, and `rocking_curves_simulation.png`.

## Advanced Python API

For custom scripts, use:

```python
import swanx as sx
```

File IO, preprocessing, fitting, and diagnostics live in focused namespaces:

```python
from swanx.io import load_material_tables, stack_from_layer_specs
from swanx.fitting import FittingProblem, FitParameter
from swanx.diagnostics import plot_correlation_matrix
```

A realistic custom file-based fitting setup is:

1. Read OPC and IMFP files with `load_material_tables(...)`.
2. Build a `SimulationStack` with `stack_from_layer_specs(...)`.
3. Build core-level requests with `core_level_from_tables(...)`.
4. Load experimental reflectivity with `read_reflectivity_data(...)`.
5. Load experimental rocking curves with `read_rocking_curve_data(...)`.
6. Pass those objects into `swanx.fitting.FittingProblem` or a fitting backend.

OPC and IMFP files are read outside JAX-traced residual functions. JAX fitting
receives fixed numerical arrays or fixed-shape model inputs.

## Troubleshooting

**Missing data file**

Paths in `project.yaml` are resolved relative to the YAML file. If a path works
from your shell but not from SWANX, rewrite it relative to `project.yaml` or use
an absolute path.

**Missing PyYAML**

Install the project extra:

```bash
python -m pip install -e ".[project]"
```

**Missing matplotlib**

Plots are optional. Install plotting support with:

```bash
python -m pip install -e ".[plot]"
```

If matplotlib is unavailable, SWANX skips plots and records the reason in
`report.md`.

**Missing JAX or SciPy**

For JAX least-squares workflows, install:

```bash
python -m pip install -e ".[least-squares]"
```

The default YAML path uses `run.optimizer.residual: "auto_fixed_grid"` and
requires `settings.slicing.mode: "fixed_grid"`. Install JAX/SciPy with the
least-squares extra before running it.

**Unknown layer tag**

Check `stack` layer `tags` and `core_levels.emit_from.tags`. Tags are strings
and must match exactly.

**Duplicate layer id**

Every concrete layer needs a unique `id`. In repeat blocks, include
`{repeat_index}` in repeated IDs.

**Missing emit_from**

Every core level must explicitly say where it emits from with `layer_ids`,
`tags`, or `all: true`.

**jax_least_squares without fixed-grid slicing**

Use `run.optimizer.residual: "auto_fixed_grid"` with
`settings.slicing.mode: "fixed_grid"`, or provide
`run.optimizer.residual_function_factory: "module:function"` for a custom JAX
residual. The legacy `settings.optimizer` and `settings.fit_method` fields
remain supported. SWANX does not fall back to BO.

**BO not installed**

Install the fitting extra:

```bash
python -m pip install -e ".[fit]"
```

BO remains an optional baseline and can require many evaluations in broad
parameter spaces.
