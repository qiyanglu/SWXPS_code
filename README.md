# SWXPS

SWXPS is a transparent Python package for multilayer x-ray reflectivity and
standing-wave XPS simulation, with validated NumPy physics kernels and optional
fitting and JAX backends.

## Capabilities

- Parratt-recursion s-polarized reflectivity.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Graded effective slices for rough interfaces.
- Local optical-constant and IMFP table loading.
- Concentration profiles and normalized SW-XPS rocking curves.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.

Experimental fitting remains physically provisional: bounds, weights, optical
constants, IMFPs, and fitted structures must be reviewed before results are
treated as quantitative.

## Default unified layer grid

High-level `ReflectivityRequest` and `RockingCurveRequest` simulations use one
cell-centered grid for roughness, fields, and SW-XPS by default. For each finite
layer,

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2.0` Angstrom by default. Pass a
policy to customize those values:

```python
from swxps import LayerSlicingPolicy, RockingCurveRequest

policy = LayerSlicingPolicy(
    min_slices=10,
    max_slice_thickness=2.0,  # Angstrom; user configurable
)
request = RockingCurveRequest(..., slicing=policy)
```

Set `slicing=None` to select the legacy step-based path explicitly. In legacy
mode, `field_step` and `roughness_step` retain their historical behavior for
regression and compatibility:

```python
request = RockingCurveRequest(
    ...,
    slicing=None,
    field_step=1.0,
    roughness_step=1.0,
)
```

For fitting with JAX, build a fixed plan from a stack evaluated at thickness
upper bounds and reuse it in every request or `FittingProblem`:

```python
from swxps import fixed_layer_grid_plan

plan = fixed_layer_grid_plan(capacity_stack.optical_layers, policy)
```

Generic high-level unified forward calls can use the JAX backend, but Python/
NumPy grid materialization is not itself JAX-traceable. End-to-end gradients
currently require a JAX-native fixed-shape forward model with grid topology
prepared outside the traced objective.

## Install and test

```powershell
python -m pip install -e .
python -m pip install -e ".[plot]"
python -m pytest
```

Optional fitting extras are `fit`, `gradient`, and `least-squares`.

## Quick examples

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

## Repository layout

- `src/swxps/`: package implementation.
- `tests/`: physics and fitting regression tests.
- `examples/`: small reproducible tutorials.
- `case_studies/`: Sample 12/13 experimental data, runners, and canonical results.
- `benchmarks/`: synthetic fitting studies and performance benchmarks.
- `runs/`: generated optimizer output, ignored by Git.
- `archive/`: superseded local experiments, ignored by Git.
- `docs/`: architecture, roadmap, plans, and historical documents.
- `OPC/`, `IMFP/`: optical-constant and attenuation tables.
- `Yeh_Lindau_1985_Xsection_CSV_Database/`: cross-section tables.

Generated attempts never belong in `examples`, `case_studies`, or `benchmarks`.
Only raw inputs, maintained scripts, deterministic fixtures, and selected
canonical results should be versioned there.
