<p align="center">
  <img src="swanx_logo.png" width="360" alt="swanx logo">
</p>

# swanx

**swanx** stands for *standing-wave analysis for X-ray spectroscopy*.

It provides Python tools for multilayer X-ray reflectivity, standing-wave XPS
simulation, fitting, and diagnostics. The package emphasizes transparent,
validated NumPy physics kernels with optional fitting and JAX backends.

## Start here

```python
import swanx as sx

result = sx.simulate_reflectivity(
    sx.ReflectivityRequest(
        angles=angles,
        energy_ev=energy_ev,
        stack=stack,
    )
)
```

High-level simulations use unified layer slicing by default. Set
`slicing=None` only for the legacy fixed-step path. The old `swxps` namespace
is kept temporarily as a compatibility alias; new code should use `swanx`.

Preferred public imports for the migrated Stage 2 utilities are:

```python
from swanx.stack import LayerSlicingPolicy
from swanx.diagnostics import compute_parameter_diagnostics
```

Flat imports such as `from swanx.slicing import LayerSlicingPolicy` and all
old `swxps.*` paths remain compatibility shims, but are no longer preferred.

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

## Unified layer slicing

For each finite layer, the default high-level grid uses

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2.0` Angstrom by default:

```python
import swanx as sx

policy = sx.LayerSlicingPolicy(
    min_slices=10,
    max_slice_thickness=2.0,
)
request = sx.RockingCurveRequest(..., slicing=policy)
```

For fixed-shape JAX fitting, build a plan from a stack evaluated at thickness
upper bounds and reuse it for every evaluation:

```python
plan = sx.fixed_layer_grid_plan(capacity_stack.optical_layers, policy)
```

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
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

Existing examples may still use `swxps` while the compatibility period is in
effect. New examples and user code should import `swanx`.

## Repository layout

- `src/swanx/`: primary package implementation and public namespaces.
- `src/swxps/`: temporary compatibility aliases for old imports.
- `tests/`: physics, fitting, namespace, and parity regression tests.
- `examples/`: small reproducible tutorials.
- `case_studies/`: experimental inputs, maintained runners, and canonical results.
- `benchmarks/`: synthetic fitting studies and performance benchmarks.
- `runs/`: generated optimizer output, ignored by Git.
- `archive/`: superseded local experiments, ignored by Git.
- `docs/`: architecture, roadmap, plans, handoff, and historical documents.
- `OPC/`, `IMFP/`: optical-constant and attenuation tables.

Generated attempts never belong in `examples`, `case_studies`, or `benchmarks`.
Only raw inputs, maintained scripts, deterministic fixtures, and selected
canonical results should be versioned there.
