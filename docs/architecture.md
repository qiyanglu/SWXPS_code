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

These are first-stage facades over the existing implementation modules. The
facades improve discovery without rewriting or duplicating numerical kernels.

## Stage 2 implementation locations

Diagnostics covariance/plot implementations and stack slicing/profile implementations now live in their public subpackages. `swanx.slicing`, `swanx.profiles`, and `swanx._diagnostics` are thin compatibility shims. Legacy `swxps.slicing`, `swxps.profiles`, and `swxps.diagnostics` resolve to the same canonical objects. Optics, XPS, simulation, and fitting implementations remain flat for later, separately tested stages.

## Core physics

- `reflectivity.py`: Parratt amplitudes and reflectivity.
- `fields.py`: transfer-matrix fields and rough-interface effective layers.
- `_xps.py`: attenuation and normalized rocking-curve integration.
- `stack/profiles.py`: material and concentration profiles versus depth.
- `stack/slicing.py`: adaptive and fixed-plan unified layer grids.

The core uses grazing angles in degrees, energy in eV, lengths in Angstrom, and
`n = 1 - delta + i beta`. Vacuum is first and the semi-infinite substrate last.

## High-level simulation and fitting

- `simulation.py`, `simulation_jax.py`, `simulation_unified.py`: request/result
  APIs and maintained NumPy/JAX forward paths.
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
