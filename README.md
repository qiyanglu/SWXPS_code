<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

SWANX is a Python workflow tool for multilayer x-ray reflectivity and
standing-wave XPS: it helps you describe a layer stack, simulate reflectivity
and rocking curves, fit models to data, and write project-local reports with
CSVs, Markdown, plots, and diagnostics.

## What Is SWANX?

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale
**X**-ray spectroscopy. It is a modern Python workflow tool for multilayer
SW-XPS reflectivity, standing-wave field and rocking-curve simulation, fitting,
reports, and diagnostics.

SWANX is inspired by x-ray-optics and SW-PES workflows such as YXRO and SWOPT,
but it is not a full YXRO/SWOPT clone. The focus is a transparent, scriptable,
and testable Python package with a human-editable YAML project workflow.

## Why Standing-Wave XPS?

When an x-ray beam reflects from a multilayer, the incident and reflected waves
form a depth-dependent electric-field intensity inside the stack. As the
incident angle changes, the field antinodes move through buried layers and
interfaces. A core-level rocking curve measures photoemission intensity versus
incident angle:

$$I_\mathrm{core}(\theta)$$

Different core levels and different emitting layers can have different rocking
curves because their atoms sit at different depths and experience different
standing-wave fields and attenuation paths. This makes SW-XPS useful for buried
layers, interfaces, capping layers, and multilayer superlattices that are hard
to isolate with conventional XPS alone.

## What SWANX Can Do

| Capability | Current support |
| --- | --- |
| Multilayer x-ray reflectivity simulation | Parratt / transfer-matrix workflows for explicit layer stacks. |
| Standing-wave field and SW-XPS rocking-curve simulation | Core-level requests with emitting-layer selection and IMFP attenuation. |
| OPC, IMFP, reflectivity, and rocking-curve readers | `swanx.io` loads tutorial and user-provided files. |
| YAML ProjectSpec workflow | `swanx init`, `inspect`, `validate`, and `run` around one editable `project.yaml`. |
| Project-local reports | `report.md`, resolved CSVs, simulation/data/fit CSVs, and optional plots. |
| Stack, material, and core-level builders | Reusable builders for custom Python workflows. |
| JAX least-squares fitting | Recommended differentiable fixed-shape path with user-provided factory callbacks. |
| Bayesian optimization | Optional global black-box baseline / robustness check, not the default. |
| Diagnostics | Residuals, best parameters, and least-squares covariance/correlation when available. |

## What SWANX Does Not Try To Do

- SWANX is not a full YXRO replacement.
- SWANX does not implement Auger, XES, XMCD, circular polarization, or
  single-crystal diffraction.
- SWANX does not provide a GUI, Excel frontend, HTML report, or JSON ProjectSpec
  input.
- The current ProjectSpec workflow does not build automatic no-code JAX residual
  functions; JAX fitting still needs explicit callback factories.
- Bayesian optimization is not the default fitting path and is not used as a
  fallback when JAX callbacks are missing.

## Quickstart

Install the beginner project workflow and plotting extras, create a project, and
run it:

```bash
python -m pip install -e ".[project,plot]"
swanx init my_project
python my_project/run_project.py
```

Use the CLI for inspection and automation:

```bash
swanx inspect my_project/project.yaml
swanx validate my_project/project.yaml
swanx run my_project/project.yaml
```

The generated `run_project.py` prints progress messages and then prints the
output directory.

## Installation

Core simulation package:

```bash
python -m pip install -e .
```

Project workflow plus plots:

```bash
python -m pip install -e ".[project,plot]"
```

Project workflow plus recommended JAX least-squares tools:

```bash
python -m pip install -e ".[project,least-squares,plot]"
```

Project workflow plus Bayesian optimization baseline support:

```bash
python -m pip install -e ".[project,fit,plot]"
```

## ProjectSpec At A Glance

A YAML ProjectSpec has these top-level sections:

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

The full section-by-section reference is in
[docs/projectspec_reference.md](docs/projectspec_reference.md).

## Outputs

If `project.output_dir` is not set, runs are written next to `project.yaml`:

```text
runs/<project_name>_<timestamp>/
```

Every run writes `report.md`, input snapshots, resolved CSVs, simulation CSVs,
and `fit/fit_summary.json`. Runs with experimental datasets also write
`data/*_experimental.csv` and residuals. Fitting runs write best-parameter and
fit-contribution CSVs. When `report.save_plots: true` and matplotlib is
available, SWANX writes overview, reflectivity, rocking-curve, and stack
schematic plots; fitting backends may add optimizer-specific plots and CSVs.

## Next Steps

- Read the practical tutorial: [docs/user_guide.md](docs/user_guide.md).
- Use the YAML reference: [docs/projectspec_reference.md](docs/projectspec_reference.md).
- Browse copy-pasteable examples: [examples/projectspec](examples/projectspec).
- Start from the repository templates: [templates/project_minimal.yaml](templates/project_minimal.yaml)
  and [templates/run_project.py](templates/run_project.py).

## Advanced Python API

Custom scripts can import the compact API with:

```python
import swanx as sx
```

File IO, preprocessing, fitting, and diagnostics are available through focused
namespaces such as `swanx.io`, `swanx.preprocessing`, `swanx.fitting`, and
`swanx.diagnostics`. Maintained fitting backends live under `swanx.fitting`:
`swanx.fitting.bo`, `swanx.fitting.jax_gradient`, and
`swanx.fitting.jax_least_squares`. Root modules such as `swanx.bo`,
`swanx.jax_gradient`, `swanx.jax_least_squares`, `swanx.reflectivity`, and
`swanx._fitting` are compatibility shims only.
