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

## Current discretization boundary

The validated legacy path uses separate step-based roughness and field grids.
Both lengths depend on current thickness through `ceil`, so fitted thickness
changes can change JAX shapes. Existing step APIs remain supported.

## Planned unified-grid boundary

The additive unified mode will separate planning from evaluation:

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

See `docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md`. This mode is not
implemented yet; the legacy path remains the only active behavior.

## Performance boundary

`benchmarks/performance/profile_forward_workflow.py` reports static loading and
dynamic stages separately. The slicing milestone will add a thickness-sweep
benchmark covering compilation, repeated calls, accuracy, and memory.
