<p align="center">
  <img src="swanx_logo.png" width="360" alt="SWANX logo">
</p>

# SWANX

**SWANX** stands for **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy.

The Python package is imported as `swanx`.

SWANX provides transparent Python tools for multilayer X-ray reflectivity,
standing-wave XPS simulation, fitting, and diagnostics. Its validated NumPy
physics kernels are complemented by maintained JAX fitting paths.

## Beginner simulation API

Start with the compact top-level API:

```python
import numpy as np
import swanx as sx

stack = sx.SimulationStack(
    (
        sx.StackLayer("vacuum", 0.0),
        sx.StackLayer("film", 24.0, delta=5.0e-6, beta=1.0e-7),
        sx.StackLayer("substrate", 0.0, delta=1.0e-5, beta=2.0e-7),
    )
)
result = sx.simulate_reflectivity(
    sx.ReflectivityRequest(
        angles=np.linspace(0.5, 4.0, 200),
        energy_ev=3000.0,
        stack=stack,
    )
)
```

The beginner-facing objects are `sx.SimulationStack`, `sx.StackLayer`,
`sx.ReflectivityRequest`, `sx.RockingCurveRequest`,
`sx.simulate_reflectivity`, and `sx.simulate_rocking_curves`. Advanced objects
live in focused namespaces such as `swanx.stack`, `swanx.optics`, and
`swanx.xps`; these are not competing user APIs.

High-level simulations use unified layer slicing by default. Set
`slicing=None` only to select the legacy fixed-step path. The old `swxps`
namespace is kept temporarily as a compatibility alias; new code should use
`swanx`.

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

## Capabilities

- Parratt-recursion s-polarized reflectivity.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- One unified layer grid for roughness, optics, fields, and XPS integration.
- Local optical-constant and IMFP table loading.
- Concentration profiles and normalized SW-XPS rocking curves.
- JAX/autodiff least squares and gradient fitting, plus Bayesian optimization.
- Covariance, correlation, confidence-interval, and identifiability diagnostics.

Experimental fitting remains physically provisional: bounds, weights, optical
constants, IMFPs, and fitted structures must be reviewed before results are
treated as quantitative.

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
prepared outside the traced function.

## Unified layer slicing

For each finite layer, the default high-level grid uses

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2.0` Angstrom by default. The
same default now applies to simulation and `FittingProblem`.

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
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/fitting/jax_gradient_reflectivity_fit.py
python examples/fitting/jax_least_squares_reflectivity_fit.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

The tutorials use `swanx` imports. The BO benchmark is retained as an explicit
legacy/baseline workflow and selects `slicing=None` where it uses fixed steps.

## Repository layout

- `src/swanx/`: primary package implementation and public namespaces.
- `src/swxps/`: temporary compatibility aliases for old imports.
- `tests/`: physics, fitting, namespace, and parity regression tests.
- `examples/`: small reproducible tutorials.
- `case_studies/`: experimental inputs, maintained runners, and canonical results.
- `benchmarks/`: synthetic fitting studies and performance benchmarks.
- `runs/`: local generated optimizer output, fully ignored by Git.
- `archive/`: local superseded experiments, fully ignored by Git.
- `docs/`: architecture, roadmap, plans, handoff, and historical documents.
- `OPC/`, `IMFP/`: optical-constant and attenuation tables.

Generated attempts never belong in `examples`, `case_studies`, or `benchmarks`.
Only raw inputs, maintained scripts, deterministic fixtures, and selected
canonical results should be versioned there.
