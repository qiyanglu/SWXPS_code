# PROJECT_STATE

## Current state

SWANX uses `swanx` as the only supported Python namespace. The beginner
workflow is:

```text
data/OPC + data/IMFP + data/curves
        -> swanx.io
        -> SimulationStack / CoreLevelRequest / ReflectivityData / RockingCurveData
        -> simulation + fitting + diagnostics
```

Tutorial data live at:

- `data/OPC/`
- `data/IMFP/`
- `data/curves/`

## Implemented workflow

- `swanx.io` reads OPC, IMFP, reflectivity, and rocking-curve files.
- `swanx.io` builds `SimulationStack` and `CoreLevelRequest` objects from
  material tables.
- `swanx.preprocessing` owns rocking-curve normalization algorithms.
- `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`.
- `swanx.io.__all__` is narrow and explicit; it does not export preprocessing
  functions or legacy flat helpers.
- `read_imfp(...)` raises on malformed rows after numeric data begins while
  preserving `.ANG` support.
- OPC validation rejects negative `beta`, allows negative `delta`, rejects
  duplicate photon energies, and requires finite positive photon energies.

## API notes

- User simulations should start with `import swanx as sx`.
- OPC files are interpolated at photon energy.
- IMFP files are interpolated at `E_kin = h nu - E_B`.
- `RockingCurveRequest` does not read files directly.
- Unified slicing is the default high-level simulation path.
- `ReflectivityRequest` and `RockingCurveRequest` support `polarization="s"`
  by default, `polarization="p"`, and mixed dictionaries such as
  `{"s": 0.7, "p": 0.3}`. Mixed reflectivity and SW-XPS raw intensity are
  combined before normalization.
- JAX least-squares/autodiff is the recommended fitting path for fixed-shape
  workflows; BO remains a baseline.


## Current cleanup update

Fitting module cleanup completed:

- obsolete active Sample 13 local case-study test removed;
- optimizer-independent fitting implementation moved to `swanx.fitting.core`;
- `swanx._fitting` remains a thin compatibility shim for old local scripts;
- active docs now describe `case_studies/` as local/private and ignored by Git.

## Latest validation


Fitting cleanup validation:

```bash
python -m pytest tests/test_fitting.py tests/test_unified_fitting.py tests/test_bo.py tests/test_jax_gradient.py tests/test_jax_least_squares.py tests/test_diagnostics.py tests/test_namespace_imports.py -q
# 36 passed

python -m pytest -q --basetemp runs/pytest_fitting_cleanup
# 220 passed, 1 xfailed
```
Polarization fitting integration fix completed:

```bash
python -m pytest tests/test_polarization.py tests/test_fitting.py tests/test_unified_fitting.py tests/test_reflectivity_jax.py tests/test_jax_gradient.py -q
# 36 passed
```

- `FittingProblem` now accepts and validates `polarization`.
- Fitting `evaluate(...)` and `simulate(...)` propagate polarization into reflectivity and rocking-curve requests.
- Mixed polarization weights must sum to 1.


Pre-commit consistency sweep completed:

```bash
python -m pytest tests/test_polarization.py tests/test_namespace_imports.py tests/test_io_curves.py -q --basetemp runs/pytest_sweep_targeted
# 30 passed

python examples/io/opc_imfp_rocking_curve_quickstart.py
# reflectivity points: 201
# La 4d: kinetic energy 795.0 eV, normalized RC mean 1.000

python examples/io/experimental_curve_loading.py
# FittingProblem datasets: reflectivity=tutorial reflectivity, rocking_curves=1

python examples/reflectivity/plot_lno_sto_reflectivity.py
# saved examples/reflectivity/lno_sto_reflectivity.png

python examples/xps/plot_lno_la4d_rocking_curve.py
# saved examples/xps/lno_la4d_o1s_ti2p_rocking_curves.png

python examples/fitting/jax_least_squares_reflectivity_fit.py
# final cost: 4.667158e-21

python -m pytest -q --basetemp runs/pytest_sweep_full
# 218 passed, 1 xfailed, 1 warning
```

Step 10 validation completed:

```bash
python -m pytest -q --basetemp runs/pytest_step10_full
# 210 passed, 1 xfailed, 1 warning

python examples/io/opc_imfp_rocking_curve_quickstart.py
# reflectivity points: 201
# La 4d: kinetic energy 795.0 eV, normalized RC mean 1.000

python examples/io/experimental_curve_loading.py
# FittingProblem datasets: reflectivity=tutorial reflectivity, rocking_curves=1

python examples/reflectivity/plot_lno_sto_reflectivity.py
# saved examples/reflectivity/lno_sto_reflectivity.png

python examples/xps/plot_lno_la4d_rocking_curve.py
# saved examples/xps/lno_la4d_o1s_ti2p_rocking_curves.png

python examples/fitting/jax_least_squares_reflectivity_fit.py
# final cost: 4.667158e-21

python -m pytest -q --basetemp runs/pytest_polarization_full
# 217 passed, 1 xfailed, 1 warning
```
