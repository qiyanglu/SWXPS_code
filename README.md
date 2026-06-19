# SWXPS

Transparent Python tools for simulating x-ray reflectivity and standing-wave
XPS from multilayer thin films.

The current code focuses on readable, testable simulations for:

- Parratt-recursion x-ray reflectivity for multilayers
- Transfer-matrix reflectivity and electric-field profiles
- Error-function interface roughness through graded effective slices
- Optical constants loaded from Henke-style `.dat` files in `OPC`
- IMFP values loaded from tabulated files in `IMFP`
- Normalized standing-wave XPS rocking curves with constant cross sections
- Optimizer-independent fitting objectives for reflectivity and SW-XPS RCs
- Bayesian optimization through `scikit-optimize`
- Optional local JAX-gradient optimization with SciPy L-BFGS-B
- Declarative stack templates, including superlattices
- Stack/concentration/schematic visualization utilities

## Install

From the repository root:

```powershell
python -m pip install -e .
```

For plotting examples:

```powershell
python -m pip install -e ".[plot]"
```

For the standalone JAX-gradient optimizer:

```powershell
python -m pip install -e ".[gradient]"
```

## Run Tests

```powershell
python -m pytest
```

## Examples

Example scripts live in topic-specific subfolders under `examples`.

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/roughness/compare_lno_sto_roughness.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/profiles/plot_lno_sto_stack_profile.py
python examples/synthetic_c_lno_sto/generate_lno_sto_c_synthetic_data.py
python examples/synthetic_c_lno_sto/fit_lno_sto_c_synthetic_bo.py
```

Generated `.png` figures and `.csv` example data are tracked so the GitHub repo
shows representative outputs without requiring a local run first.

## Key Files

Core simulation modules:

- `src/swxps/reflectivity.py`: Parratt recursion, Fresnel amplitudes, and roughness correction.
- `src/swxps/fields.py`: transfer-matrix reflectivity and electric-field profiles.
- `src/swxps/xps.py`: depth integration and normalized SW-XPS rocking-curve helpers.
- `src/swxps/simulation.py`: high-level request/result API for reflectivity and RC simulations.
- `src/swxps/profiles.py`: concentration and material-profile sampling versus depth.

Fitting and optimization modules:

- `src/swxps/fitting.py`: fitting parameters, datasets, weighted objectives, and fit history.
- `src/swxps/bo.py`: `scikit-optimize` Bayesian optimization and staged multi-start fitting.
- `src/swxps/jax_gradient.py`: standalone JAX-gradient L-BFGS-B optimizer.
- `src/swxps/fit_diagnostics.py`: history CSV export and fit/surrogate diagnostic plots.
- `src/swxps/stack_builders.py`: declarative layer and superlattice stack construction.
- `src/swxps/stack_visualization.py`: schematic stack drawings from fitted parameters.

Example outputs:

- `examples/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv`: deterministic synthetic reflectivity and RC data.
- `examples/synthetic_c_lno_sto/lno_sto_c_synthetic_data.png`: simulated C/LNO/STO reflectivity and RCs.
- `examples/synthetic_c_lno_sto/lno_sto_c_bo_best_fit.png`: normal BO best-fit comparison.
- `examples/synthetic_c_lno_sto/lno_sto_c_bo_surrogate_slices.png`: 1D GP surrogate slices.
- `examples/synthetic_c_lno_sto/lno_sto_c_bo_lno_sto_surrogate_2d_3d.png`: LNO/STO 2D/3D surrogate surface.
- `examples/synthetic_c_lno_sto/lno_sto_c_bo_stack_schematic.png`: fitted-stack schematic visualization.

## Current Scope

The package is now a transparent simulation and early fitting backend. The BO
workflow has been validated on synthetic C/LNO/STO data, but experimental-data
fitting still needs careful parameter bounds, weighting choices, and physical
cross-checks before results should be treated as quantitative.

## JAX Gradient Optimizer

The JAX-gradient optimizer is a local L-BFGS-B optimizer, separate from the BO
workflow. It requires JAX and SciPy, and it works best from a physically
reasonable initial structure. Because it is a local optimizer, it can get
trapped in local minima; use it for refinement or controlled synthetic tests,
not as a replacement for global Bayesian optimization.
