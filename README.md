# SWANX

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale
**X**-ray spectroscopy.

SWANX is a Python workflow tool for multilayer standing-wave XPS projects. It
helps users build a stack, simulate x-ray reflectivity and core-level rocking
curves, fit selected parameters, and write reproducible reports. Its current
scope is multilayer SW-XPS reflectivity, rocking curves, fitting, and
diagnostics rather than a general spectroscopy or crystallography platform.

## Why SWANX?

Standing-wave XPS analysis is useful because the x-ray electric field inside a
multilayer changes with incidence angle. A reflectivity scan gives

$$R(\theta)$$

and a core-level rocking curve gives

$$I_\mathrm{core}(\theta)$$

In practice, the hard part is keeping optical constants, IMFP tables, the stack
model, emitting layers, experimental curves, fitting parameters, diagnostics,
and reports consistent. SWANX packages those moving parts into one editable
project workflow.

## What SWANX Currently Supports

- ProjectSpec YAML projects with `swanx init`, `swanx inspect`,
  `swanx validate`, and `swanx run`.
- OPC, IMFP, reflectivity, and rocking-curve data readers.
- Multilayer reflectivity and SW-XPS rocking-curve simulation.
- JAX least-squares fitting with the ProjectSpec `auto_fixed_grid` residual
  path for fixed-topology stacks.
- Bayesian optimization as an optional global baseline or robustness check.
- Markdown reports, CSV outputs, plots, parameter diagnostics, and optional
  least-squares identifiability reports.

## Quickstart

Install the project workflow, JAX least-squares, and plotting extras:

```bash
python -m pip install -e ".[project,least-squares,plot]"
```

Create and run a starter project:

```bash
swanx init my_project
python my_project/run_project.py
```

The default init project is self-contained. It copies packaged C/LaNiO3/SrTiO3 starter OPC, IMFP, and curve files into `my_project/data/`, then runs a JAX least-squares fit using an internal fixed-grid residual built from `project.yaml` with:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
```

Results are written under:

```text
my_project/runs/<project_name>_<timestamp>/
```

## Choose A Starter

```bash
swanx init my_project --template fit
swanx init my_project --template simulate
swanx init my_project --template minimal
swanx init my_project --template fit-demo
swanx init my_project --template multilayer
```

- `--template fit`: preferred C/LaNiO3/SrTiO3 fitting starter.
- `--template minimal`: legacy alias for the default fitting starter.
- `--template fit-demo`: explicit fitting starter alias.
- `--template multilayer` / `--template simulate`: simulation-only repeated
  multilayer starter.

Despite the name, `minimal` is not simulation-only; it is the default runnable
fitting project.

## ProjectSpec In One Minute

A SWANX project is controlled by `project.yaml`. The top-level sections are:

```yaml
project:
run:
settings:
materials:
parameters:
stack:
core_levels:
datasets:
report:
```

Use `run:` for mode, optimizer settings, and output switches. Detailed YAML
syntax belongs in the reference:

- [`docs/projectspec_reference.md`](docs/projectspec_reference.md)
- [`examples/01_quickstart_projectspec/`](examples/01_quickstart_projectspec/)

## Outputs

Every run writes a project-local folder such as:

```text
runs/<project_name>_<timestamp>/
```

Typical contents include `report.md`, resolved input CSVs, simulated curve
CSVs, experimental-data and residual CSVs when datasets are present, plot
files when plotting is enabled, method-specific files under `optimizer/`, and
optional `identifiability_analysis/` outputs for JAX least-squares runs.

For `simulate_only`, reports clearly state that no fitting was performed. For
fitting runs, reports include the final objective, best parameters, output
files, and concise diagnostic notes when available.

## Fitting

JAX least-squares is the recommended fitting path for differentiable
fixed-shape ProjectSpec workflows:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
  outputs:
    identifiability: true
```

`auto_fixed_grid` is the default YAML residual path for fixed-topology
ProjectSpec fits. It builds the residual internally from the YAML stack,
parameters, datasets, core levels, and fixed-grid slicing settings. Advanced
users can still provide `residual_function_factory: "module:function"` when a
custom residual cannot be described by the ProjectSpec stack.

Bayesian optimization is available as an optional global black-box baseline or
robustness check. BO is not the default fitting method and is not used as a fallback for JAX methods.

## Docs And Examples

- [`docs/user_guide.md`](docs/user_guide.md) - practical workflow guide.
- [`docs/projectspec_reference.md`](docs/projectspec_reference.md) - detailed
  YAML reference.
- [`examples/README.md`](examples/README.md) - example map and learning path.
- [`examples/04_fitting/projectspec_jax_least_squares/`](examples/04_fitting/projectspec_jax_least_squares/)
  - runnable ProjectSpec JAX least-squares example.

## Installation Options

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

## Background

SWANX is inspired by standing-wave x-ray optics and SW-PES workflows such as
YXRO and SWOPT, but it has a narrower Python-first scope focused on multilayer
SW-XPS reflectivity, rocking curves, fitting, and diagnostics.

Related background:

- S.-H. Yang et al., "Making use of x-ray optical effects in photoelectron-,
  Auger electron-, and x-ray emission spectroscopies," *Journal of Applied
  Physics* 113, 073513 (2013).
- O. Karslioglu et al., "An Efficient Algorithm for Automatic Structure
  Optimization in X-ray Standing-Wave Experiments," *Journal of Electron
  Spectroscopy and Related Phenomena* 230, 10-20 (2019).

## Development

Run tests with:

```bash
python -m pytest
```
