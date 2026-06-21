# SWXPS

SWXPS is a transparent Python package for multilayer x-ray reflectivity and
standing-wave XPS simulation. It includes validated NumPy physics kernels plus
optional fitting and JAX backends.

## Capabilities

- Parratt-recursion s-polarized reflectivity.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Graded effective slices for rough interfaces.
- Henke-style optical-constant and tabulated IMFP loading.
- Material concentration profiles and normalized SW-XPS rocking curves.
- Optimizer-independent reflectivity/RC objectives.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.

Experimental fitting is supported but remains physically provisional: bounds,
weights, optical constants, IMFPs, and fitted structures must be reviewed before
results are treated as quantitative.

## Install

```powershell
python -m pip install -e .
python -m pip install -e ".[plot]"
```

Optional fitting backends:

```powershell
python -m pip install -e ".[fit]"
python -m pip install -e ".[gradient]"
python -m pip install -e ".[least-squares]"
```

## Test

```powershell
python -m pytest
```

## Start with an example

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

See `examples/README.md` for the maintained example index.

## Repository layout

- `src/swxps/`: package implementation.
- `tests/`: physics and fitting regression tests.
- `examples/`: reproducible tutorials and maintained experimental runners.
- `runs/`: generated optimizer outputs, ignored by Git.
- `archive/`: preserved superseded experiments, ignored by Git.
- `OPC/`, `IMFP/`: local optical-constant and attenuation tables.
- `Yeh_Lindau_1985_Xsection_CSV_Database/`: photoionization cross-section tables.
- `docs/architecture.md`: module and data-flow overview.
- `docs/roadmap.md`: current scope and priorities.
- `docs/history/`: completed development logs and superseded documents.

Generated runs are deliberately separate from examples. Canonical sample results
remain in each case study's `best_results_so_far` folder; repeated attempts and
historical outputs belong under `artifacts/`.
