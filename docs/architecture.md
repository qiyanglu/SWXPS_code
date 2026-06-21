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
- `optical_constants.py`, `imfp.py`: local table loading.
- `preprocessing.py`: experimental preparation and RC normalization.

## Fitting

- `fitting.py`: datasets, parameters, residuals, weights, and histories.
- `bo.py`: Bayesian optimization.
- `jax_gradient.py`: bounded L-BFGS-B optimization.
- `jax_least_squares.py`: bounded TRF least squares.
- `fit_diagnostics.py`, `result_exports.py`, `stack_visualization.py`: outputs.

## Repository data flow

Tutorials in `examples/`, experimental runners in `case_studies/`, and synthetic
drivers in `benchmarks/` call the same package APIs. Generated output is written
to `runs/`. Reviewed case-study results may be promoted into a
`best_results_so_far` directory. Superseded local material belongs in `archive/`.
