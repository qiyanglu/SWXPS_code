# Architecture

The primary package is `swanx`: *standing-wave analysis for X-ray
spectroscopy*. `swxps` is a temporary compatibility package that aliases the
same modules and objects; new code should import `swanx`.

## Public namespaces

- `swanx.stack`: layers, stacks, templates, slicing, and profiles.
- `swanx.optics`: Parratt, transfer-matrix, fields, and unified-grid optics.
- `swanx.xps`: attenuation, XPS intensity, and rocking curves.
- `swanx.fitting`: parameters, objectives, and maintained fitting backends.
- `swanx.diagnostics`: covariance, correlation, plots, and result exports.
- `swanx.io`: optical constants, IMFP tables, and preprocessing.
- `swanx.workflows`: high-level simulation, fitting, and reporting entry points.

These public namespaces now contain the implementations migrated through
Stages 2-5. Compatibility facades preserve older import paths without
rewriting or duplicating numerical kernels.

## Stage 2 implementation locations

Diagnostics covariance/plot implementations and stack slicing/profile implementations now live in their public subpackages. `swanx.slicing`, `swanx.profiles`, and `swanx._diagnostics` are thin compatibility shims. Legacy `swxps.slicing`, `swxps.profiles`, and `swxps.diagnostics` resolve to the same canonical objects. Optics, XPS, simulation, and fitting implementations remain flat for later, separately tested stages.

## Stage 3 optics implementation locations

Parratt reflectivity, transfer-matrix/electric-field, and unified-grid optics implementations now live in `swanx.optics.parratt`, `swanx.optics.fields`, and `swanx.optics.unified_grid`. Flat `swanx.reflectivity`, `swanx.fields`, and `swanx.unified_grid` modules are thin compatibility shims; legacy `swxps.*` paths expose the same objects. Existing `simulate_reflectivity_unified` and `simulate_rocking_curves_unified` remain implemented in `simulation_unified.py` and are lazily re-exported from `swanx.optics.unified_grid` to avoid cycles.

## Stage 4 XPS implementation locations

XPS attenuation, continuous intensity/property sampling, normalized rocking
curves, and cell-centered grid integration now live in
`swanx.xps.attenuation`, `swanx.xps.intensity`,
`swanx.xps.rocking_curve`, and `swanx.xps.grid`. The flat `swanx._xps`
module, the former XPS exports from `swanx.optics.unified_grid`, and legacy
`swxps.*` paths remain identity-preserving compatibility shims. High-level
simulation exports from `swanx.xps` are lazy to avoid initialization cycles.
Simulation, fitting, and workflow implementations were not moved.

## Stage 5 stack-model and simulation-workflow locations

Material-labeled stack data models now live in `swanx.stack.model`, and
high-level simulation request/result classes and forward entry points now live
in `swanx.workflows.simulate`. `swanx.simulation` is a thin compatibility shim;
legacy `swxps.simulation`, preferred subpackage imports, and top-level beginner
exports resolve to the same canonical objects. The `swanx.workflows` facade
loads fitting and diagnostics conveniences lazily to avoid initialization
cycles. Numerical algorithms were not changed.

## Stage 6 slim simulation compatibility layer

`swanx.simulation` now contains compatibility imports only: no classes,
functions, physics helpers, or stack-construction logic are defined there.
Material lookup and emitting-layer filtering utilities live in
`swanx.xps.utils` and are shared by the NumPy, JAX, and unified workflow paths.
Canonical execution remains in `swanx.workflows.simulate`; old
`swanx.simulation` and `swxps.simulation` callers receive the same objects and
numerically identical results.

## Core physics

- `optics/parratt.py`: Parratt amplitudes and reflectivity.
- `optics/fields.py`: transfer-matrix fields and rough-interface effective layers.
- `xps/attenuation.py`: electron attenuation through depth-dependent IMFPs.
- `xps/intensity.py`: continuous XPS integration and graded property sampling.
- `xps/rocking_curve.py`: normalized rocking-curve construction.
- `xps/grid.py`: cell-centered attenuation and midpoint XPS integration.
- `xps/utils.py`: internal material lookup and emitting-layer filtering.
- `stack/profiles.py`: material and concentration profiles versus depth.
- `stack/slicing.py`: adaptive and fixed-plan unified layer grids.
- `stack/model.py`: material-labeled stack and layer data models.

The core uses grazing angles in degrees, energy in eV, lengths in Angstrom, and
`n = 1 - delta + i beta`. Vacuum is first and the semi-infinite substrate last.

## High-level simulation and fitting

- `workflows/simulate.py`: high-level request/result APIs and NumPy workflow.
- `simulation_jax.py`, `simulation_unified.py`: maintained JAX and unified-grid
  forward paths; `simulation.py` preserves former imports as a thin shim.
- `_fitting.py`, `bo.py`, `jax_gradient.py`, `jax_least_squares.py`: datasets,
  parameters, objectives, and optimizers.
- `_diagnostics.py`, `fit_diagnostics.py`, `result_exports.py`: local parameter
  diagnostics, plots, and exports.
- `optical_constants.py`, `imfp.py`, `preprocessing.py`: cached tables and
  experimental preparation.

Fitting diagnostics are implemented. For residual vector `r`, Jacobian `J`,
`N` residuals, and `P` parameters, the local least-squares approximation is

```text
Cov = s^2 (J^T J)^+
s^2 = ||r||^2 / max(1, N - P)
```

`plot_parameter_estimates(...)` normalizes each estimate and confidence interval by its declared finite parameter range by default, so heterogeneous quantities share a 0-1 axis. Raw lower and upper bounds are labeled at the endpoints. Use `normalization=None` for the legacy raw-value view.

Least-squares result adapters recompute covariance from `final_residuals` and `final_jacobian` instead of trusting a cached optimizer matrix. Both computed and explicitly supplied covariance matrices are checked for finiteness, symmetry, non-negative variances, and positive-semidefinite eigenstructure, then symmetrized before correlation. Materially malformed matrices raise rather than producing a plot; tiny roundoff asymmetry/eigenvalue excursions warn and are corrected. Correlations are required to be symmetric and bounded in `[-1, 1]` apart from tiny clipped roundoff.

The resulting uncertainty and correlation estimates remain local
approximations, not substitutes for nonlinear profile likelihoods or Bayesian
posterior sampling.

## Unified slicing boundary

High-level reflectivity and rocking-curve requests use unified slicing by
default. `slicing=None` explicitly selects the validated legacy fixed-step
path. The default policy is

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
min_slices = 10
max_slice_thickness = 2 Angstrom (user configurable)
```

One cell-centered grid supplies roughness grading, effective optical cells,
field locations, concentration and IMFP samples, attenuation, and midpoint
rocking-curve weights. During fitting, counts can be planned from capacity
thicknesses so trial thickness changes values and widths without changing JAX
shapes.

Generic grid materialization uses Python floats and NumPy arrays and is not
JAX-traceable. End-to-end differentiable workflows use the JAX-native
fixed-plan model with topology prepared outside the trace.

## Repository data flow

```text
optical/IMFP tables --cached parse--> interpolated material values
fit parameters ---------------------> StackTemplate -> SimulationStack
SimulationStack --unified slicing---> effective optical cells and field grid
fields + concentration + IMFP ------> normalized SW-XPS rocking curves
simulated + experimental curves ----> objective -> fit -> diagnostics/report
```

Tutorials in `examples/`, experimental runners in `case_studies/`, and
synthetic drivers in `benchmarks/` call the same APIs. Generated output belongs
in `runs/`; superseded local material belongs in `archive/`.
