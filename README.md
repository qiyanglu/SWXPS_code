# SWANX

<p align="center">
  <img src="swanx_logo.png" alt="SWANX logo" height="320">
</p>

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale
**X**-ray spectroscopy.

SWANX is a Python workflow tool for multilayer standing-wave XPS projects. It
helps users build a stack, simulate X-ray reflectivity and core-level rocking
curves, fit selected parameters, and write reproducible reports. Its current
scope is multilayer SW-XPS reflectivity, rocking curves, fitting, and
diagnostics rather than a general spectroscopy or crystallography platform.

## Why SWANX?

Standing-wave XPS analysis is useful because the X-ray electric field inside a
multilayer changes with incidence angle, making reflectivity and core-level
rocking curves sensitive to layer thicknesses, roughness, composition profiles,
and emission depth. In practice, the hard part is keeping optical constants,
IMFP tables, the stack model, emitting layers, experimental curves, fitting
parameters, diagnostics, and reports consistent.

SWANX packages those moving parts into one editable project workflow. Its
JAX-based fixed-grid path gives differentiable forward simulations for
least-squares fitting, so users can move beyond black-box parameter searches
when the model topology is fixed. The accompanying diagnostics help identify
which parameters are actually constrained, which parameters are strongly
correlated, and where a reduced or reparameterized model may be more reliable.

## Features

- ProjectSpec YAML projects with `swanx init`, `swanx inspect`,
  `swanx validate`, and `swanx run`.
- OPC, IMFP, reflectivity, and rocking-curve data readers.
- Multilayer reflectivity and SW-XPS rocking-curve simulation.
- JAX least-squares fitting with the ProjectSpec `auto_fixed_grid` residual
  path for fixed-topology stacks.
- Bayesian optimization as an optional global baseline or robustness check.
- Markdown reports, CSV outputs, plots, parameter diagnostics, and
  least-squares identifiability reports with sensitivity, correlation,
  weak-mode, and dataset-contribution summaries.

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

The default init project is self-contained. It copies packaged OPC, IMFP, and
synthetic curve files for an example stack with a C capping layer on a
20-repeat LaNiO3/SrTiO3 superlattice mirror, meaning 40 oxide layers total,
grown on a SrTiO3 substrate. It then runs a JAX least-squares fit using an
internal fixed-grid residual built from `project.yaml` with:

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

The starter templates are small project folders meant to get different users to
their first successful run quickly. Use the fitting starter when you want to see
the full data-to-report workflow, including optimization and diagnostics. Use a
simulation starter when you want to learn the stack, materials, and plotting
syntax before introducing experimental datasets or fitted parameters.

```bash
swanx init my_project --template fit
swanx init my_project --template simulate
swanx init my_project --template minimal
swanx init my_project --template fit-demo
swanx init my_project --template multilayer
```

- `--template fit`: preferred fitting starter for the C-capped
  [LaNiO3/SrTiO3]x20/SrTiO3 synthetic benchmark geometry.
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
optional `identifiability_analysis/` and `next_project/` outputs for
JAX least-squares runs. The maintained fitting example enables
`next_project/project_best_start.yaml` and `next_project/project_reduced.yaml`
so users can restart from best-fit values or review a low-sensitivity reduced
model.

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
    next_project:
      best_start: true
      reduced: true
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

The numbered example folders are intended to be read as a path: quick ProjectSpec
YAMLs, experimental-data loading, compact Python API scripts, fitting examples,
and then advanced visualizations. Start with the examples map unless you already
know which layer of the workflow you want to inspect.

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

SWANX is inspired by standing-wave X-ray optics and SW-PES workflows such as
YXRO and SWOPT, but it has a narrower Python-first scope focused on multilayer
SW-XPS reflectivity, rocking curves, fitting, and diagnostics.

Related background:

- S.-H. Yang et al., "Making use of X-ray optical effects in photoelectron-,
  Auger electron-, and X-ray emission spectroscopies," *Journal of Applied
  Physics* 113, 073513 (2013).
- O. Karslioglu et al., "An Efficient Algorithm for Automatic Structure
  Optimization in X-ray Standing-Wave Experiments," *Journal of Electron
  Spectroscopy and Related Phenomena* 230, 10-20 (2019).

## Development

Run tests with:

```bash
python -m pytest
```
