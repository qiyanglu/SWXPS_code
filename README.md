<p align="center">
  <img src="swanx_logo.png" width="360" alt="swanx logo">
</p>

# swanx

**swanx** stands for *standing-wave analysis for X-ray spectroscopy*.

It provides Python tools for multilayer X-ray reflectivity, standing-wave XPS
simulation, fitting, and diagnostics. The package emphasizes transparent,
validated NumPy physics kernels with optional fitting and JAX backends.

## Getting Started

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
request = sx.ReflectivityRequest(
    angles=np.linspace(0.5, 4.0, 200),
    energy_ev=3000.0,
    stack=stack,
)
result = sx.simulate_reflectivity(request)
```

SWANX is a physics simulation and differentiable fitting engine. JAX automatic
differentiation is the primary optimization backend and the recommended
default for new fitting work.

The official user entry pattern is always `import swanx as sx`, followed by a
stack, a request, and a `simulate_*` call. Subpackages such as `swanx.stack`,
`swanx.optics`, `swanx.xps`, and `swanx.workflows` are implementation
namespaces, not competing user APIs. The old `swanx.simulation` and `swxps`
paths remain compatibility-only.

## Capabilities

- Parratt-recursion s-polarized reflectivity.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- One unified layer grid for roughness, optics, fields, and XPS integration.
- Local optical-constant and IMFP table loading.
- Concentration profiles and normalized SW-XPS rocking curves.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.
- Covariance, correlation, confidence-interval, and identifiability diagnostics.

Experimental fitting remains physically provisional: bounds, weights, optical
constants, IMFPs, and fitted structures must be reviewed before results are
treated as quantitative.

## Optimization in SWANX

SWANX supports two optimization strategies.

### 1. Bayesian Optimization (baseline)

- Global black-box optimization.
- Robust, but slow.
- Retained primarily as a baseline and comparison method.

### 2. JAX-based automatic differentiation (primary method)

- Exact gradients through automatic differentiation.
- Significantly faster convergence.
- Scales to high-dimensional parameter spaces.
- Compatible with JIT compilation and GPU execution.

**JAX-based optimization is the recommended default approach.**

## Unified layer slicing

For each finite layer, the default high-level grid uses

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2.0` Angstrom by default.
High-level requests use this unified slicing automatically. `slicing=None`
exists only for regression and compatibility with the legacy fixed-step path.

Generic unified forward calls can use the JAX backend, but Python/NumPy grid
materialization is not itself JAX-traceable. End-to-end gradients require the
maintained JAX-native fixed-shape path with topology prepared outside the trace.

## Install and test

```powershell
python -m pip install -e .
python -m pip install -e ".[plot]"
python -m pytest
```

Optional fitting extras are `fit`, `gradient`, and `least-squares`.

## Examples

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/fitting/jax_gradient_reflectivity_fit.py
python examples/fitting/jax_least_squares_reflectivity_fit.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

New user code should follow the single `import swanx as sx` pattern shown in
Getting Started. Older scripts may retain compatibility imports temporarily.

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
