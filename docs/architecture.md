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

## Forward-model data flow

The maintained path through the package is:

```text
optical/IMFP tables --cached parse--> interpolated material values
fit parameters ---------------------> StackTemplate -> SimulationStack
SimulationStack --roughness grading-> effective optical layers
angles + effective layers ----------> reflectivity and electric fields
fields + concentration + IMFP ------> normalized SW-XPS rocking curves
simulated + experimental curves ----> fitting contributions and objective
```

`optical_constants.py` and `imfp.py` cache parsed tables because those inputs
are static during a fit. Cache entries are bounded and keyed by resolved path,
file modification time, and file size; changing a table therefore causes a
fresh parse. `clear_optical_constants_cache()` and `clear_imfp_cache()` are
available for explicit process-level invalidation.

Stack construction, fitted parameter resolution, roughness discretization,
reflectivity, fields, XPS integration, normalization, and scoring remain
dynamic. They are deliberately not hidden behind caches because their inputs
can change at every objective evaluation.

## Performance boundary

`benchmarks/performance/profile_forward_workflow.py` times the static-loading
and dynamic-computation stages separately on a representative
C/[LNO/STO]x8/STO stack. Run it before and after performance work so changes are
judged against the same workload rather than against case-study wall time.
