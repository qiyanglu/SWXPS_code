<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale
**X**-ray spectroscopy: Python tools for multilayer X-ray reflectivity,
standing-wave XPS simulation, fitting, and diagnostics.

## YAML ProjectSpec Workflow

The YAML `ProjectSpec` is the main human-editable workflow. For a beginner
starter project, install the optional YAML support and run:

```bash
python -m pip install -e ".[project]"
swanx init my_project
python my_project/run_project.py
```

Default `swanx init` copies packaged minimal tutorial OPC, IMFP, and curve files
into `my_project/data/`, so the generated project is runnable from any process
current working directory. Useful variants are:

```bash
swanx init my_project --template multilayer
swanx init my_project --template fit-demo
swanx init my_project --copy-example-data --data-root /path/to/data
swanx init my_project --data-root /path/to/data
```

Use the CLI for automation and inspection:

```bash
swanx inspect my_project/project.yaml
swanx validate my_project/project.yaml
swanx run my_project/project.yaml
```

The generated `run_project.py` prints the output directory. See the
[full user guide](docs/user_guide.md) for ProjectSpec fields, templates,
reports, and fitting notes.

## Installation

Core simulation package:

```bash
python -m pip install -e .
```

Recommended beginner/project environment:

```bash
python -m pip install -e ".[project,plot]"
```

Recommended differentiable fitting environment:

```bash
python -m pip install -e ".[project,least-squares,plot]"
```

Bayesian optimization remains an optional global black-box baseline:

```bash
python -m pip install -e ".[project,fit,plot]"
```

## What To Edit

Edit `my_project/project.yaml`: `materials`, `parameters`, `stack`,
`core_levels`, optional `datasets`, and `report`. Thickness, roughness, depth,
and IMFP values are in Angstrom. In YAML, `roughness_A` means the upper-interface
roughness/interdiffusion of that layer. In repeat blocks, `repeat_index` is
1-based.

Core levels must explicitly select emitting layers with `emit_from.layer_ids`,
`emit_from.tags`, or `emit_from.all: true`.

## Outputs

If `project.output_dir` is not set, outputs are written under the project folder:

```text
my_project/runs/<project_name>_<timestamp>/
```

Every run writes `report.md`, input snapshots, resolved CSVs, simulation CSVs,
and `fit/fit_summary.json`. Simulation-only runs do not write
`fit/best_parameters.csv`; residuals are written only when experimental datasets
exist. When `report.save_plots: true` and matplotlib is available, reports also
include reflectivity, rocking-curve, and residual plots with experimental
overlays when data are present. Skipped plot reasons are recorded per plot in
`report.md`.

## Advanced Python API

Custom scripts can import the compact API with:

```python
import swanx as sx
```

File IO, preprocessing, fitting, and diagnostics are also available through
focused namespaces such as `swanx.io`, `swanx.preprocessing`,
`swanx.fitting`, and `swanx.diagnostics`. Use these when a custom script needs
more control than a YAML `ProjectSpec`.

## Fitting And Optimization

JAX least-squares is the recommended fitting path for differentiable fixed-shape
workflows. ProjectSpec v1.2 still requires user-provided JAX callback factories
for `jax_least_squares` and `jax_gradient`; SWANX does not generate automatic
no-code JAX residuals in this pass. Bayesian optimization is available as an
optional global black-box baseline and is not the default or a fallback.

## Examples And Development Notes

Repository-local starter files remain available under `templates/`. The
maintained package namespace is `swanx`; old `swxps` paths are retired. Excel,
GUI, JSON input, HTML reports, Auger, XES, XMCD, and single-crystal diffraction
frontends are out of scope for the current ProjectSpec workflow.
