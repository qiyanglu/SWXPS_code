<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** means **S**tanding-**W**ave **A**nalysis for **N**anoscale
**X**-ray spectroscopy: Python tools for multilayer X-ray reflectivity,
standing-wave XPS simulation, fitting, and diagnostics.

## What problem does SWANX solve?

SWANX has two supported user paths:

```text
project.yaml
        -> swanx.project.run_project(...)
        -> runs/<project_name>_<timestamp>/ report folder
```

For custom Python workflows, SWANX turns local data files into explicit
simulation and fitting inputs:

```text
data/OPC + data/IMFP + data/curves
        -> swanx.io
        -> SimulationStack / CoreLevelRequest / ReflectivityData / RockingCurveData
        -> simulation + fitting + diagnostics
```

The compact simulation API is still available for scripts:

```python
import swanx as sx

result = sx.simulate_reflectivity(
    sx.ReflectivityRequest(...)
)
```

## Installation

```bash
python -m pip install -e .
```

Recommended JAX least-squares and plotting environment:

```bash
python -m pip install -e ".[least-squares,plot]"
```

Bayesian-optimization baseline:

```bash
python -m pip install -e ".[fit]"
```

## YAML project workflow

The YAML `ProjectSpec` is the new human-editable project input for running a
SW-XPS project without writing a full custom fitting script. YAML support is
optional and installed with:

```bash
python -m pip install -e ".[project]"
```

The easiest entry is to copy or edit `templates/project_minimal.yaml` as your
`project.yaml`, then run a tiny Python script:

```bash
python templates/run_project.py
```

The same workflow is available through the thin CLI for automation:

```bash
swanx validate templates/project_minimal.yaml
swanx run templates/project_minimal.yaml
```

Excel and GUI frontends are not implemented here; they remain deferred frontend ideas.

Run tests with:

```bash
python -m pytest
```

## Start here: OPC + IMFP -> reflectivity + SW-XPS rocking curves

```python
import numpy as np
import swanx as sx
from swanx.io import (
    core_level_from_tables,
    load_material_tables,
    stack_from_layer_specs,
)

energy_ev = 900.0
angles = np.linspace(5.0, 15.0, 201)

tables = load_material_tables(
    opc_files={
        "LNO": "data/OPC/LaNiO3.dat",
        "STO": "data/OPC/SrTiO3.dat",
    },
    imfp_files={
        "LNO": "data/IMFP/LNO.ANG",
        "STO": "data/IMFP/STO.ANG",
    },
)

stack = stack_from_layer_specs(
    [
        {"material": "vacuum", "thickness": 0.0},
        {"material": "LNO", "thickness": 40.0, "roughness": 3.0},
        {"material": "STO", "thickness": 0.0},
    ],
    optical_constants=tables.optical_constants,
    energy_ev=energy_ev,
)

reflectivity = sx.simulate_reflectivity(
    sx.ReflectivityRequest(angles=angles, energy_ev=energy_ev, stack=stack)
)

la4d = core_level_from_tables(
    name="La 4d",
    binding_energy_ev=105.0,
    photon_energy_ev=energy_ev,
    concentration_by_material={"LNO": 1.0},
    imfp_tables=tables.imfp,
)

rocking_curves = sx.simulate_rocking_curves(
    sx.RockingCurveRequest(
        angles=angles,
        photon_energy_ev=energy_ev,
        stack=stack,
        core_levels=(la4d,),
    )
)
```

OPC tables are interpolated at photon energy. IMFP tables are interpolated at
photoelectron kinetic energy, `E_kin = h nu - E_B`. `RockingCurveRequest` does
not read files directly; `swanx.io` prepares file inputs before request
creation.

## Load experimental curves for fitting

```python
from swanx.io import read_reflectivity_data, read_rocking_curve_data

reflectivity_exp = read_reflectivity_data(
    "data/curves/lno_sto_reflectivity.csv",
)

la4d_exp = read_rocking_curve_data(
    "data/curves/la4d_rocking_curve.csv",
    normalization_mode="mean",
)
```

The loaders return `ReflectivityData` and `RockingCurveData` objects consumed by
`swanx.fitting`. IO reads files; `swanx.preprocessing` owns rocking-curve
normalization algorithms.

## Input file formats

OPC files use CXRO-style columns:

```text
Energy(eV), Delta, Beta
```

IMFP files may be TPP-style `.ANG`, CSV, or whitespace tables with energy and
IMFP columns.

Reflectivity and rocking-curve files may be CSV or whitespace tables:

```text
angle_deg,reflectivity
5.0,0.010
```

```text
angle_deg,intensity
5.0,1.00
```

Headerless curve files are supported when explicit column indices are supplied.

## Public API map

- `import swanx as sx`: recommended simulation entry point.
- `swanx.project`: YAML ProjectSpec validation, execution, and report output.
- `swanx.io`: OPC, IMFP, material-table, stack/core-level, and experimental
  curve readers.
- `swanx.preprocessing`: rocking-curve normalization.
- `swanx.fitting`: fitting data classes, objectives, BO baseline, and JAX
  optimizers.
- `swanx.diagnostics`: uncertainty, correlation, plotting, and reports.

High-level simulations use unified layer slicing by default. Set `slicing=None`
only when reproducing the legacy fixed-step path.

Reflectivity and rocking-curve requests default to s polarization. Set
`polarization="p"` for p polarization or pass a mixed dictionary such as
`{"s": 0.5, "p": 0.5}` to combine raw s/p reflectivity or SW-XPS intensity
before any rocking-curve normalization.

## Fitting and optimization

JAX-based automatic differentiation is the primary fitting method for
fixed-shape differentiable workflows. It provides autodiff Jacobians, JIT
compatibility, and fast least-squares convergence.

Bayesian optimization is retained as a global black-box baseline and robustness
comparison. It is generally slower for the current differentiable SWANX
workflows.

File IO is outside JAX-traced residual functions; fitting receives fixed arrays
or fixed-shape model inputs. The YAML ProjectSpec workflow is the preferred
human-editable project entry; direct fitting APIs remain available for custom
fixed-shape JAX or BO workflows.

## Examples

```bash
python examples/io/opc_imfp_rocking_curve_quickstart.py
python examples/io/experimental_curve_loading.py
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/fitting/jax_least_squares_reflectivity_fit.py
```

Tutorial data live under `data/`. These files are examples, not a built-in
materials database.

## Development notes

- `src/swanx/`: maintained package and only supported Python namespace.
- `tests/`: regression tests.
- `examples/`: compact tutorials.
- local `case_studies/`: private experimental inputs/runners kept on the working machine and ignored by Git.
- `benchmarks/`: synthetic fitting and performance benchmarks.
- `runs/`: local generated outputs, ignored by Git.
- `archive/`: local superseded experiments, ignored by Git.
- `docs/`: architecture, roadmap, plans, project state, TODO, and history.

Generated attempts belong in local `runs/`, not in tracked `examples/` or `benchmarks/`.

The early development namespace `swxps` was retired before public release. New
code should use `swanx`.
