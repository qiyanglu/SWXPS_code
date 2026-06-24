# Sample 12 swanx least-squares sanity check

Status: Complete (2026-06-24).

## Goal

Run the maintained Sample 12 bounded JAX/TRF least-squares fit through the
`swanx` namespace and generate uncertainty and correlation diagnostics in an
isolated directory under `runs/`.

## Scope and safeguards

- Use the maintained runner in `case_studies/sample_12/jax_least_squares_fit/`.
- Keep canonical `best_results_so_far/` unchanged with `--skip-promotion`.
- Write all generated artifacts to
  `runs/sample_12/jax_least_squares/stage4_swanx_sanity_20260624/`.
- Verify NumPy/JAX objective agreement and inspect the saved diagnostics.
- Do not change the fitting model or physical algorithms.

## Validation

1. Run the setup path and confirm the maintained script imports `swanx`.
2. Run bounded TRF least squares to convergence.
3. Confirm uncertainty/correlation PNG and CSV outputs exist and are nonempty.
4. Check correlation symmetry, diagonal, finite values, and absolute range.
5. Record the result in `docs/PROJECT_STATE.md` and `docs/TODO.md`.

## Result

- Converged by `xtol` in 23 function and 3 Jacobian evaluations.
- Optimizer time: 17.54 seconds.
- JAX objective: `0.00338515813538`.
- NumPy objective: `0.00338515813538`.
- The uncertainty and correlation plots are nonempty and readable.
- The 18x18 correlation CSV is finite, symmetric, has unit diagonal, and
  contains no values outside `[-1, 1]`.
- Promotion was skipped; canonical case-study results were not modified.
