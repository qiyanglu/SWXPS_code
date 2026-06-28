<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy.

SWANX is a Python package for building, simulating, fitting, and documenting
multilayer standing-wave XPS projects. It is designed for workflows where a user
has a multilayer stack, optical constants, core-level signals, and possibly
experimental reflectivity or rocking-curve data, and wants a reproducible way to
go from model assumptions to simulated curves, fits, plots, and reports.

SWANX focuses on a practical question:

> Given a multilayer sample, where do different core-level signals come from,
> and what stack parameters best explain the measured reflectivity and
> standing-wave XPS rocking curves?

## Why SWANX?

Standing-wave XPS is powerful because the x-ray field inside a multilayer sample
depends strongly on incidence angle. Near a multilayer Bragg condition, the
incident and reflected x-rays interfere, creating a depth-dependent excitation
field. As the incidence angle is scanned, atoms at different depths can be
highlighted differently.

A typical SW-XPS dataset therefore contains curves such as

$$
R(\theta)
$$

for x-ray reflectivity, and

$$
I_\mathrm{core}(\theta)
$$

for a core-level photoemission intensity measured as a function of incidence
angle. These core-level “rocking curves” can be sensitive to layer thickness,
interface roughness/interdiffusion, emitting-layer depth, optical constants, and
photoelectron attenuation.

In practice, SW-XPS analysis can be difficult because many pieces must stay
consistent:

- optical constants for each material;
- electron attenuation / IMFP tables;
- a multilayer stack model;
- roughness or interdiffusion assumptions;
- core levels and their emitting layers;
- reflectivity and rocking-curve datasets;
- fitting parameters and bounds;
- diagnostic plots and output files.

SWANX turns these pieces into a reproducible project workflow.

## What SWANX helps you do

SWANX is intended for multilayer SW-XPS projects such as oxide heterostructures,
superlattices, thin films on standing-wave mirrors, and similar layered samples.

Typical tasks include:

| Task | What SWANX provides |
|---|---|
| Build a multilayer model | YAML or Python stack definitions with stable layer IDs and tags |
| Load optical data | OPC and IMFP readers for material tables |
| Simulate reflectivity | Multilayer x-ray reflectivity vs incidence angle |
| Simulate SW-XPS rocking curves | Core-level intensity vs incidence angle for selected emitting layers |
| Compare to experiment | Overlay experimental reflectivity and rocking-curve points |
| Fit model parameters | Fit thickness / roughness parameters with bounded optimizers |
| Diagnose results | Residuals, best parameters, covariance/correlation for least-squares when available |
| Record outputs | Project-local `report.md`, CSV files, resolved inputs, and plots |

The recommended fitting path is **JAX least-squares** for differentiable,
fixed-shape workflows. Bayesian optimization is also available as an optional
global black-box baseline or robustness check; it is not the default and is not
used as a fallback.

SWANX is inspired by established x-ray-optics and standing-wave photoemission
workflows, including YXRO and SWOPT, but it is not intended to be a full clone of
either. The emphasis here is a modern, scriptable Python workflow for multilayer
reflectivity and SW-XPS analysis.

## Quickstart

From the repository root, install the project workflow and plotting extras:

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

The generated project is self-contained: `swanx init` copies packaged tutorial
OPC, IMFP, and curve files into `my_project/data/`.

The output folder will look like:

```text
my_project/runs/<project_name>_<timestamp>/
```

Each run writes a Markdown report:

```text
my_project/runs/<project_name>_<timestamp>/report.md
```

You can also use the CLI directly:

```bash
swanx inspect my_project/project.yaml
swanx validate my_project/project.yaml
swanx run my_project/project.yaml
```

## Starter Templates

`init` can generate a few different starting points:

```bash
swanx init my_project --template minimal
swanx init my_project --template multilayer
swanx init my_project --template fit-demo
```

Use:

- `minimal` for a small simulation-only project;
- `multilayer` for a repeat-block stack example;
- `fit-demo` for a project that includes experimental datasets and fitting placeholders.

You can also point SWANX to your own data folder:

```bash
swanx init my_project --data-root /path/to/data
```

or copy from that folder into the project:

```bash
swanx init my_project --copy-example-data --data-root /path/to/data
```

## Project File Overview

A SWANX project is controlled by `project.yaml`.

The top-level structure is:

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

A minimal stack might look like:

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

Core levels explicitly choose which layers emit:

```yaml
core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["lno_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0
```

A reflectivity dataset can be added with:

```yaml
datasets:
  reflectivity:
    path: "data/curves/lno_sto_reflectivity.csv"
    name: "Reflectivity"
    angle_column: "angle_deg"
    intensity_column: "reflectivity"
```

A rocking-curve dataset can be added with:

```yaml
datasets:
  rocking_curves:
    - path: "data/curves/la4d_rocking_curve.csv"
      name: "La 4d"
      angle_column: "angle_deg"
      intensity_column: "intensity"
      normalization: "mean"
```

For the full YAML reference, see:

- `docs/projectspec_reference.md`

For copy-pasteable project examples, see:

- `docs/projectspec_examples.md`
- `examples/projectspec/`

## Important YAML Conventions

Lengths are in Angstrom.

```yaml
thickness_A: 40.0
roughness_A: 3.0
```

`roughness_A` is the upper-interface roughness of a layer. For layer `j`, it
describes the interface between layer `j-1` and layer `j`.

Only parameters with `vary: true` are fitted:

```yaml
parameters:
  lno_thickness:
    initial: 40.0
    lower: 30.0
    upper: 50.0
    vary: true

  repeat_center:
    value: 20.0
    vary: false
```

Inline parameter references are supported:

```yaml
thickness_A: "$lno_thickness"
```

Safe arithmetic expressions are supported:

```yaml
thickness_A: "A_LNO + B_LNO * repeat_index"
```

In repeat blocks, `repeat_index` is 1-based:

```yaml
stack:
  - repeat:
      times: 4
      layers:
        - id: "lno_{repeat_index}"
          material: "LNO"
          tags: ["lno_layers"]
          thickness_A: "$lno_thickness"
          roughness_A: "$interface_roughness"
```

## Outputs

A run folder contains both human-readable and machine-readable outputs.

Common files include:

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

If experimental datasets are provided, SWANX also writes experimental-data and
residual files:

```text
data/reflectivity_experimental.csv
data/rocking_curves_experimental.csv
fit/residuals.csv
```

If fitting is performed, SWANX writes:

```text
fit/best_parameters.csv
fit/fit_contributions.csv
optimizer/
```

If `report.save_plots: true` and matplotlib is available, SWANX writes plots such
as:

```text
plots/fit_overview.png
plots/reflectivity_fit.png
plots/rocking_curves_fit.png
plots/stack_schematic.png
```

Least-squares runs may also include convergence, uncertainty, and correlation
plots when the required diagnostics are available.

Skipped optional outputs are recorded in `report.md`.

## Fitting Overview

Simulation-only projects use:

```yaml
settings:
  fit_method: "simulate_only"
```

For JAX least-squares fitting:

```yaml
settings:
  fit_method: "jax_least_squares"
  optimizer:
    residual_function_factory: "fit_factory:build_residual"
    max_nfev: 100
    estimate_covariance: true
```

The factory module can live next to `project.yaml`. SWANX adds the project folder
to the import path when loading project-local factories.

Current ProjectSpec fitting does not automatically synthesize a JAX residual
function. This is intentional: fixed-shape JAX residuals should be explicit and
traceable.

Bayesian optimization can be used as an optional baseline:

```yaml
settings:
  fit_method: "bayesian_optimization"
  optimizer:
    n_calls: 40
    n_initial_points: 10
    random_state: 0
```

Use BO when you want a global black-box check or a robustness comparison. Do not
treat it as the default fitting route for differentiable SWANX workflows.

## Installation Options

Core simulation package:

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

For custom scripts, use the compact simulation API:

```python
import swanx as sx
```

Focused subpackages are available for explicit workflows:

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

Root modules such as `swanx.bo`, `swanx.jax_gradient`,
`swanx.jax_least_squares`, `swanx.reflectivity`, and `swanx._fitting` are
compatibility shims.

## Documentation

Start here:

- `docs/user_guide.md` - practical walkthrough
- `docs/projectspec_reference.md` - full YAML reference
- `docs/projectspec_examples.md` - example project files
- `docs/architecture.md` - package layout and design notes

## Development

Run tests with:

```bash
python -m pytest
```

The maintained Python package namespace is:

```python
import swanx
```

The old `swxps` namespace is retired.

Generated run outputs should stay in local `runs/` folders and should not be
committed unless intentionally added as documentation examples.

## Background References

SWANX builds on ideas from standing-wave x-ray optics and SW-XPS analysis, while
keeping a narrower, Python-first scope.

Useful background papers include:

- S.-H. Yang et al., ?Making use of x-ray optical effects in photoelectron-,
  Auger electron-, and x-ray emission spectroscopies: Total reflection,
  standing-wave excitation, and resonant effects,? Journal of Applied Physics
  113, 073513 (2013). doi:10.1063/1.4790171
- O. Karsl?o?lu et al., ?An Efficient Algorithm for Automatic Structure
  Optimization in X-ray Standing-Wave Experiments,? Journal of Electron
  Spectroscopy and Related Phenomena 230, 10-20 (2019).
  doi:10.1016/j.elspec.2018.10.006
