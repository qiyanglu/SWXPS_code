<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** stands for **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy.

The Python package is imported as `swanx`.

SWANX provides transparent Python tools for multilayer X-ray reflectivity,
standing-wave XPS simulation, fitting, and diagnostics. The package prepares
optical constants and IMFP values from local files, then passes explicit
numerical requests into the simulation and fitting kernels.

The early development namespace `swxps` was retired before public release. New
code should use `swanx`.

## Start here: OPC + IMFP workflow

Beginner users should start with `import swanx as sx`. Advanced users can use
subpackages such as `swanx.stack`, `swanx.optics`, `swanx.xps`,
`swanx.fitting`, `swanx.diagnostics`, and `swanx.io`.

```python
import numpy as np
import swanx as sx
from swanx.io import (
    load_material_tables,
    stack_from_layer_specs,
    core_level_from_tables,
)

energy_ev = 900.0
angles = np.linspace(5.0, 15.0, 201)

tables = load_material_tables(
    opc_files={
        "LNO": "examples/data/OPC/LaNiO3.dat",
        "STO": "examples/data/OPC/SrTiO3.dat",
    },
    imfp_files={
        "LNO": "examples/data/IMFP/LNO.ANG",
        "STO": "examples/data/IMFP/STO.ANG",
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
    sx.ReflectivityRequest(
        angles=angles,
        energy_ev=energy_ev,
        stack=stack,
    )
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

OPC files are interpolated at photon energy. IMFP files are interpolated at
photoelectron kinetic energy, `E_kin = hν - E_B`. `RockingCurveRequest` does not
read files; `swanx.io` resolves files before request creation. The stack already
contains optical constants at `photon_energy_ev`, and each `CoreLevelRequest`
already contains IMFP values at its own kinetic energy.

High-level simulations use unified layer slicing by default. Set `slicing=None`
only to select the legacy fixed-step path.

## Public API shape

The compact beginner-facing objects are:

- `sx.SimulationStack`
- `sx.StackLayer`
- `sx.ReflectivityRequest`
- `sx.RockingCurveRequest`
- `sx.CoreLevelRequest`
- `sx.simulate_reflectivity(...)`
- `sx.simulate_rocking_curves(...)`

Function/action names use lowercase `snake_case`. Classes and dataclasses use
`PascalCase`. Advanced code should import from focused subpackages rather than
old flat implementation modules.

## Advanced fitting API

Fitting utilities intentionally live under `swanx.fitting`, rather than all
being exposed at top level:

```python
from swanx.fitting import (
    FitParameter,
    ReflectivityData,
    RockingCurveData,
    build_jax_residual_function,
    optimize_with_jax_least_squares,
)
```

`FittingProblem` uses unified slicing by default, matching the high-level
simulation requests. Pass `slicing=None` explicitly only when reproducing the
legacy separate `field_step` and `roughness_step` calculations.

## Optimization

SWANX supports two fitting strategies.

### JAX-based automatic differentiation (recommended)

- differentiable fixed-shape reflectivity and SW-XPS models;
- exact Jacobians of the implemented forward model via autodiff;
- JIT-compatible residual and Jacobian evaluation;
- much faster convergence than BO in the tested real-data workflows.

### Bayesian optimization (baseline)

- black-box global search;
- useful for robustness checks and comparison;
- generally slower for the current differentiable SWANX workflows.

Generic Python/NumPy grid materialization is not JAX-traceable. End-to-end JAX
fitting uses the maintained fixed-shape JAX path with topology/grid structure
prepared outside the traced function. OPC and IMFP files are read outside the
JAX-traced residual function; JAX fitting receives fixed numerical arrays or
fixed-shape model inputs.

## Unified layer slicing

For each finite layer, the default high-level grid uses

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2.0` Angstrom by default. The same
default applies to simulation and `FittingProblem`.

## Install and test

Minimal editable install:

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

Run the regression suite with:

```bash
python -m pytest
```

## Examples

```bash
python examples/io/opc_imfp_rocking_curve_quickstart.py
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/fitting/jax_gradient_reflectivity_fit.py
python examples/fitting/jax_least_squares_reflectivity_fit.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

The repository examples use tutorial data under `examples/data/OPC` and
`examples/data/IMFP`. For real analyses, users should provide their own OPC and
IMFP files and pass explicit paths through `load_material_tables`. These example
files are not a built-in materials database.

The BO benchmark is retained as an explicit legacy/baseline workflow and selects
`slicing=None` where it uses fixed steps.

## Repository layout

- `src/swanx/`: primary package implementation and public namespaces.
- `tests/`: physics, fitting, namespace, IO, and parity regression tests.
- `examples/`: small reproducible tutorials.
- `examples/data/OPC/`, `examples/data/IMFP/`: tutorial OPC and IMFP files.
- `case_studies/`: experimental inputs, maintained runners, and canonical results.
- `benchmarks/`: synthetic fitting studies and performance benchmarks.
- `runs/`: local generated optimizer output, fully ignored by Git.
- `archive/`: local superseded experiments, fully ignored by Git.
- `docs/`: architecture, roadmap, plans, handoff, and historical documents.

Generated attempts never belong in `examples`, `case_studies`, or `benchmarks`.
Only raw inputs, maintained scripts, deterministic fixtures, and selected
canonical results should be versioned there.
