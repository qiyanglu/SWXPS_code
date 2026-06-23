# Sample 12 case study

- `ExpRCs.dat` and `Reflectivity_Exp.dat`: experimental inputs.
- `jax_gradient_fit/` and `jax_least_squares_fit/`: maintained fitting runners.
- `support/legacy_bo/`: older BO modules still imported by current runners.
- `best_results_so_far/`: canonical promoted result.

Generated runs are stored under `runs/sample_12`; older preserved
experiments are under `archive/sample_12`.
## Diagnostics sanity check

The maintained bounded TRF runner imports `swanx` and saves normalized parameter uncertainty, validated parameter correlation, and correlation CSV outputs. The 2026-06-23 sanity run is under `runs/sample_12/jax_least_squares/diagnostics_sanity_check/`; promotion was disabled, so `best_results_so_far/` was not changed.
