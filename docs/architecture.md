# Architecture

## Core physics

- `reflectivity.py`: Parratt amplitudes and reflectivity.
- `fields.py`: transfer-matrix fields and rough-interface effective layers.
- `xps.py`: attenuation and normalized rocking-curve integration.
- `profiles.py`: material and concentration profiles versus depth.

The core uses grazing angles in degrees, energy in eV, lengths in Angstrom, and
`n = 1 - delta + i beta`. Vacuum is first and the semi-infinite substrate last.

## High-level simulation

- `layers.py`, `stack_builders.py`: layer and stack construction.
- `simulation.py`, `simulation_jax.py`: NumPy and fixed-shape JAX APIs.
- `optical_constants.py`, `imfp.py`: cached local table loading.
- `preprocessing.py`: experimental preparation and RC normalization.

## Fitting

- `fitting.py`: datasets, parameters, residuals, weights, and histories.
- `bo.py`: Bayesian optimization.
- `jax_gradient.py`: bounded L-BFGS-B optimization.
- `jax_least_squares.py`: bounded TRF least squares.
- `fit_diagnostics.py`, `result_exports.py`, `stack_visualization.py`: outputs.
- `diagnostics.py`: local covariance, parameter correlation, and Jacobian
  identifiability summaries and plots.

### Parameter uncertainty and identifiability

For residual vector `r`, Jacobian `J`, `N` residuals, and `P` parameters, the
reusable diagnostics API uses the local least-squares approximation

```text
Cov = s^2 (J^T J)^+
s^2 = ||r||^2 / max(1, N - P)
rho_ij = Cov_ij / sqrt(Cov_ii Cov_jj)
```

`compute_parameter_diagnostics(...)` accepts raw values, names, optional
bounds, residuals, and a Jacobian. A supplied covariance may be used directly.
`diagnostics_from_least_squares_result(...)` adapts the maintained JAX/TRF
result and `FitParameter` declarations. Matplotlib-only helpers plot parameter
intervals and bounds, the correlation matrix, and Jacobian singular values.

Large confidence intervals indicate weak constraints, correlations near `-1`
or `+1` indicate coupled parameters, and small singular values indicate nearly
unidentifiable parameter combinations. These uncertainties are local
approximations: they assume a locally linear residual model and meaningful
residual weighting. They do not replace profile likelihood or Bayesian
posterior sampling for strongly nonlinear fits.

## Repository data flow

Tutorials in `examples/`, experimental runners in `case_studies/`, and synthetic
drivers in `benchmarks/` call the same package APIs. Generated output belongs in
`runs/`; superseded local material belongs in `archive/`.

## Forward-model data flow

```text
optical/IMFP tables --cached parse--> interpolated material values
fit parameters ---------------------> StackTemplate -> SimulationStack
SimulationStack --roughness grading-> effective optical layers
angles + effective layers ----------> reflectivity and electric fields
fields + concentration + IMFP ------> normalized SW-XPS rocking curves
simulated + experimental curves ----> fitting contributions and objective
```

Parsed tables are static during a fit and cached. Stack construction,
roughness, fields, XPS, normalization, and scoring remain dynamic.

## Legacy discretization boundary

The validated legacy path uses separate step-based roughness and field grids.
Both lengths depend on current thickness through `ceil`, so fitted thickness
changes can change JAX shapes. Existing step APIs remain supported.

Set `slicing=None` on a high-level request to select this path explicitly.
Non-default `field_step` or `roughness_step` values are rejected while unified
slicing is active, because those arguments do not affect the unified grid.

## Default unified-grid boundary

High-level reflectivity and rocking-curve requests use unified slicing by
default. The unified mode separates planning from evaluation:

```text
LayerSlicingPolicy
  min_slices = 10
  max_slice_thickness = 2 Angstrom (user configurable)
                 |
capacity stack --+--> FixedLayerGridPlan (counts and topology)
trial stack ----------> LayerGrid (edges, centers, widths, mappings)
                              |
                              +--> graded optical cells / reflectivity
                              +--> electric fields at cell centers
                              +--> concentration and IMFP at cell centers
                              +--> attenuation and weighted RC integration
```

One effective optical cell and one field/XPS sample share each cell center.
RCs use cell widths as midpoint quadrature weights. During fitting, counts are
fixed from upper-bound capacity thicknesses, while trial thickness changes only
cell widths and values. This preserves JAX shapes without quantizing thickness.

For finite layer thickness `t_i`, adaptive counts follow

```text
N_i = max(min_slices, ceil(t_i / max_slice_thickness))
```

with `min_slices=10` and `max_slice_thickness=2 Angstrom` by default.

### JAX differentiation boundary

Unified high-level forward calls support a JAX-backed reflectivity/field
calculation after materialization. Full end-to-end JAX differentiation through
generic grid materialization is not currently supported: `LayerGrid` and
effective layers are constructed through Python floats and NumPy arrays.
Differentiable optimizers must use a JAX-native fixed-shape forward model whose
fixed-plan topology and nominal-to-cell mappings are prepared outside the
traced objective. Values and gradients through that array-native path are
covered under eager execution and JIT.

See `docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md` and
`docs/plans/default_unified_slicing_2026-06-23.md`. Set `slicing=None` when a
regression or compatibility workflow requires the legacy path.

## Performance boundary

`benchmarks/performance/profile_forward_workflow.py` reports static loading and
dynamic stages separately. The slicing milestone will add a thickness-sweep
benchmark covering compilation, repeated calls, accuracy, and memory.
