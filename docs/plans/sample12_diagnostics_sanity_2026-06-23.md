# Sample 12 diagnostics sanity check

Status: Implemented and validated (2026-06-23).

## Goal

Run the maintained Sample 12 bounded JAX/TRF fit through the corrected
`swanx.diagnostics` covariance, uncertainty, and correlation path without
promoting or replacing the canonical experimental result.

## Physics background

This reuses the maintained Sample 12 reflectivity and C 1s, Ni 3p, and La 4d
objective. Diagnostics use the local approximation
`Cov = s^2 (J^T J)^+`; the experimental interpretation remains provisional.

## Files to create or modify

- Sample 12 maintained least-squares runner.
- Local ignored output under `runs/sample_12/jax_least_squares/`.
- Handoff and case-study documentation.

## Implementation steps

1. Migrate the runner imports to `swanx`.
2. Save normalized uncertainty and validated correlation plots/CSV.
3. Run in an isolated folder with promotion disabled.
4. Verify numerical correlation invariants and the full regression suite.

## Tests

- Setup-only and full Sample 12 runner complete.
- Correlation is symmetric, diagonal-one, and bounded by one.
- Full repository tests pass.

## Validation

The 18-parameter matrix was exactly symmetric, had unit diagonal, and maximum
absolute entry one. The rerun converged in 22 function evaluations but did not
replace the better canonical result.

## Progress log

- 2026-06-23: Completed isolated run, generated plots and correlation CSV,
  verified matrix invariants, and passed the full suite (153 passed, 1 expected
  failure).
