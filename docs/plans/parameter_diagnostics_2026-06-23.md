# Parameter uncertainty and identifiability diagnostics plan

> Stage 2 namespace status (2026-06-23): Numerical diagnostics now live in `swanx.diagnostics.covariance`, plotting in `swanx.diagnostics.plots`, and existing report exports are exposed through `swanx.diagnostics.reports`.

> Covariance validation update (2026-06-23): Least-squares diagnostics now recompute covariance from final residuals/Jacobian; all covariance inputs are symmetrized and quality-checked, and invalid correlations cannot silently reach plotting.

> Current status (2026-06-23): Implemented and exported through `swanx.diagnostics`; old `swxps` imports remain compatible.

## Goal

Add reusable local least-squares parameter uncertainty, correlation, and
singular-value diagnostics without changing fitting or simulation behavior.

## Physics background

Near a least-squares optimum, use the local linear approximation
`Cov = s^2 (J^T J)^+`, with `s^2 = ||r||^2 / max(1, N - P)`. Standard errors
come from the covariance diagonal, correlations normalize covariance pairs, and
the Jacobian singular spectrum diagnoses weak or rank-deficient parameter
combinations.

## Files to create or modify

- New `src/swxps/diagnostics.py` numerical and plotting API.
- `src/swxps/__init__.py` exports.
- Focused `tests/test_diagnostics.py` coverage.
- `docs/architecture.md`, `docs/PROJECT_STATE.md`, and `docs/TODO.md`.

## Implementation steps

1. Validate and normalize values, names, bounds, residuals, Jacobian, and an
   optional supplied covariance.
2. Compute covariance, standard errors, robust correlations, singular values,
   degrees of freedom, residual variance, and condition number.
3. Add Matplotlib-only estimate, correlation, and singular-value plots.
4. Add a small adapter for `JaxLeastSquaresOptimizationResult` plus
   `FitParameter` declarations.
5. Run numerical, plotting, integration, and full regression tests.

## Tests

- Full-rank covariance agrees with the stated equation.
- Coupled columns produce near-unit correlation magnitude.
- Rank deficiency remains finite where appropriate and reports infinite
  condition number.
- Names and optional bounds are preserved.
- Plot functions return Matplotlib figure/axis pairs.
- Least-squares result adaptation uses existing residual/Jacobian/covariance.

## Validation

Diagnostics must be deterministic for small arrays, must not modify caller
arrays, and must label local-linear uncertainty limitations clearly. Existing
fit result behavior and covariance estimation remain unchanged.

## Progress log

- 2026-06-23: Reviewed existing fitting diagnostics and selected the JAX
  least-squares result as the only clean convenience integration point.
- 2026-06-23: Implemented validated covariance, standard-error, robust
  correlation, singular-value, condition-number, and Matplotlib plotting APIs.
- 2026-06-23: Added deterministic full-rank, coupled, rank-deficient,
  supplied-covariance, result-adapter, validation, and plotting tests.
- 2026-06-23: Full regression result: 133 passed, 1 pre-existing expected JAX
  materialization failure.
