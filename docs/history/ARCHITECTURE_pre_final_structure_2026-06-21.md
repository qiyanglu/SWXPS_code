# Architecture

## Core physics

- `reflectivity.py`: Parratt amplitudes and reflectivity.
- `fields.py`: transfer-matrix fields and rough-interface effective layers.
- `xps.py`: attenuation and normalized rocking-curve integration.
- `profiles.py`: depth-dependent material and concentration profiles.

The core uses grazing angles in degrees, energy in eV, lengths in Angstrom, and
`n = 1 - delta + i beta`. Vacuum is first and the semi-infinite substrate last.

## High-level simulation

- `layers.py` and `stack_builders.py`: layer and declarative stack construction.
- `simulation.py`: NumPy request/result API.
- `simulation_jax.py`: fixed-shape JAX high-level backend.
- `optical_constants.py` and `imfp.py`: local table loading.
- `preprocessing.py`: experimental curve preparation and RC normalization.

## Fitting

- `fitting.py`: datasets, parameters, residuals, weighting, and histories.
- `bo.py`: scikit-optimize Bayesian optimization.
- `jax_gradient.py`: bounded local L-BFGS-B optimization.
- `jax_least_squares.py`: bounded TRF least squares with JAX Jacobians.
- `fit_diagnostics.py`, `result_exports.py`, and `stack_visualization.py`: durable outputs and diagnostics.

## Artifact flow

Maintained scripts read raw data and package resources, then write generated
outputs to `artifacts/runs`. A reviewed result may be promoted to the relevant
case study's `best_results_so_far` directory. Historical runs are not package
dependencies and belong in `artifacts/archive`.
