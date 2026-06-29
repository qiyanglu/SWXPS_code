<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy.

SWANX is a Python package for multilayer standing-wave XPS projects. It helps users build a sample model, simulate x-ray reflectivity and core-level rocking curves, compare with experimental data, fit selected stack parameters, and save reproducible reports.

Typical SWANX inputs are:

```text
optical constants + IMFP tables + stack model + core levels + optional datasets
```

Typical SWANX outputs are:

```text
simulated curves + fitted parameters + residuals + plots + report.md
```

SWANX is designed for users working with layered samples such as thin films, oxide heterostructures, superlattices, and multilayer standing-wave mirrors.

## What problem does SWANX solve?

Standing-wave XPS can provide depth sensitivity because the x-ray electric field inside a multilayer changes with incidence angle. A reflectivity scan gives

$$
R(\theta)
$$

and a core-level rocking curve gives

$$
I_\mathrm{core}(\theta)
$$

where the curve shape depends on where that core-level signal originates in the stack.

In practice, analyzing these curves requires keeping many things consistent: layer thicknesses, roughnesses, optical constants, IMFPs, emitting layers, experimental datasets, fitting parameters, and output diagnostics. SWANX packages these steps into a reproducible workflow.

## What can I do with SWANX?

You can use SWANX to:

- build multilayer stacks with stable layer IDs, tags, and repeat blocks;
- load optical-constant and IMFP tables;
- simulate multilayer x-ray reflectivity;
- simulate standing-wave XPS rocking curves for selected core levels;
- overlay simulated curves with experimental reflectivity and rocking-curve data;
- fit thickness and roughness parameters;
- generate project-local reports, plots, and CSV outputs;
- use JAX least-squares for differentiable fixed-shape fitting workflows;
- use Bayesian optimization as an optional global baseline or robustness check.

SWANX is inspired by standing-wave x-ray optics and SW-PES analysis workflows such as YXRO and SWOPT, but it has a narrower goal: a modern Python workflow for multilayer SW-XPS simulation, fitting, and diagnostics.

## Quickstart

Install the project workflow and plotting extras:

```bash
python -m pip install -e ".[project,plot]"
```

Create a starter project:

```bash
swanx init my_project
```

Run it:

```bash
python my_project/run_project.py
```

The generated project is self-contained. By default, `swanx init` copies packaged tutorial OPC, IMFP, and curve files into:

```text
my_project/data/
```

Each run writes results under:

```text
my_project/runs/<project_name>_<timestamp>/
```

The main human-readable output is:

```text
report.md
```

Useful CLI commands are:

```bash
swanx inspect my_project/project.yaml
swanx validate my_project/project.yaml
swanx run my_project/project.yaml
```

## Starter templates

```bash
swanx init my_project --template minimal
swanx init my_project --template multilayer
swanx init my_project --template fit-demo
```

Use:

- `minimal` for a small simulation-only project;
- `multilayer` for a repeat-block multilayer example;
- `fit-demo` for a project with example datasets and fitting placeholders.

To use your own data folder:

```bash
swanx init my_project --data-root /path/to/data
```

To copy data into the project:

```bash
swanx init my_project --copy-example-data --data-root /path/to/data
```

## Examples and benchmarks

The user-facing examples are organized as a learning path in
[`examples/`](examples/). Most of them use a compact LNO/STO tutorial system:
LaNiO3 layers on SrTiO3 with small OPC, IMFP, reflectivity, and rocking-curve
inputs under `data/`.

The larger C/LNO/STO synthetic case appears often in [`benchmarks/`](benchmarks/)
because it is a useful repeatable target for fitting, slicing, JAX
least-squares, and Bayesian-optimization comparisons.

## ProjectSpec overview

A SWANX project is controlled by `project.yaml`.

The top-level YAML sections are:

```yaml
project:
settings:
materials:
parameters:
stack:
core_levels:
datasets:
report:
```

A minimal stack looks like:

```yaml
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - id: "lno_1"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "$lno_thickness"
    roughness_A: "$interface_roughness"

  - id: "sto_substrate"
    material: "STO"
    thickness_A: 0.0
    roughness_A: 0.0
```

A core level explicitly selects its emitting layers:

```yaml
core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["lno_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0
```

Common conventions:

- `thickness_A` and `roughness_A` are in Angstrom.
- `roughness_A` is the upper-interface roughness of that layer.
- `repeat_index` is 1-based inside repeat blocks.
- Only parameters with `vary: true` are fitted.
- Dataset paths are resolved relative to `project.yaml`.
- Core levels should use `emit_from.layer_ids`, `emit_from.tags`, or `emit_from.all: true`.

For details, see:

- [`docs/projectspec_reference.md`](docs/projectspec_reference.md)
- [`examples/01_quickstart_projectspec/README.md`](examples/01_quickstart_projectspec/README.md)

## Outputs

A typical run folder contains:

```text
report.md
input/project_original.yaml
input/project_resolved.yaml
resolved/stack_resolved.csv
resolved/materials_resolved.csv
resolved/core_levels_resolved.csv
resolved/parameters_resolved.csv
simulation/reflectivity_simulated.csv
simulation/rocking_curves_simulated.csv
fit/fit_summary.json
```

If experimental datasets are provided, SWANX also writes experimental-data and residual files.

If fitting is performed, SWANX writes best-fit parameters and optimizer-specific outputs.

If plotting is enabled with

```yaml
report:
  save_plots: true
```

SWANX writes plots such as:

```text
plots/fit_overview.png
plots/reflectivity_fit.png
plots/rocking_curves_fit.png
plots/stack_schematic.png
```

Skipped optional outputs are recorded in `report.md`.

## Fitting

Simulation-only projects use:

```yaml
settings:
  fit_method: "simulate_only"
```

The recommended fitting path is JAX least-squares for differentiable fixed-shape workflows:

```yaml
settings:
  fit_method: "jax_least_squares"
  optimizer:
    residual_function_factory: "fit_factory:build_residual"
    max_nfev: 100
    estimate_covariance: true
```

Current ProjectSpec fitting requires an explicit user-provided factory callback. SWANX does not automatically generate no-code JAX residual functions.

Bayesian optimization is available as an optional global black-box baseline:

```yaml
settings:
  fit_method: "bayesian_optimization"
  optimizer:
    n_calls: 40
    n_initial_points: 10
    random_state: 0
```

BO is not the default fitting method and is not used as a fallback.

## Installation options

Core package:

```bash
python -m pip install -e .
```

Project workflow and plots:

```bash
python -m pip install -e ".[project,plot]"
```

Project workflow with JAX least-squares:

```bash
python -m pip install -e ".[project,least-squares,plot]"
```

Project workflow with Bayesian optimization:

```bash
python -m pip install -e ".[project,fit,plot]"
```

## Python API

For custom scripts:

```python
import swanx as sx
```

Focused namespaces are also available:

```python
from swanx.io import load_material_tables, stack_from_layer_specs
from swanx.io import read_reflectivity_data, read_rocking_curve_data
from swanx.project import init_project, inspect_project, validate_project, run_project
from swanx.fitting import optimize_with_jax_least_squares
```

Maintained fitting backends live under:

```text
swanx.fitting.bo
swanx.fitting.jax_gradient
swanx.fitting.jax_least_squares
```

Root modules such as `swanx.bo`, `swanx.jax_gradient`, `swanx.jax_least_squares`, `swanx.reflectivity`, and `swanx._fitting` are compatibility shims.

## Documentation

Start here:

- [`docs/user_guide.md`](docs/user_guide.md) — practical walkthrough
- [`docs/projectspec_reference.md`](docs/projectspec_reference.md) — YAML reference
- [`examples/README.md`](examples/README.md) — user learning path and example map
- [`examples/01_quickstart_projectspec/README.md`](examples/01_quickstart_projectspec/README.md) — copy-pasteable ProjectSpec examples
- [`docs/architecture.md`](docs/architecture.md) — package layout and design notes

## Background

SWANX builds on ideas from standing-wave x-ray optics and SW-XPS analysis while keeping a narrower Python-first scope focused on multilayer reflectivity, rocking curves, fitting, and diagnostics.

Related background:

- S.-H. Yang et al., “Making use of x-ray optical effects in photoelectron-, Auger electron-, and x-ray emission spectroscopies,” *Journal of Applied Physics* 113, 073513 (2013).
- O. Karslıoğlu et al., “An Efficient Algorithm for Automatic Structure Optimization in X-ray Standing-Wave Experiments,” *Journal of Electron Spectroscopy and Related Phenomena* 230, 10–20 (2019).

## Development

Run tests with:

```bash
python -m pytest
```

