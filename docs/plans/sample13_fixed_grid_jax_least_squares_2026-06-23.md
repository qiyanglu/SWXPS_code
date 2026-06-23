# Sample 13 fixed-grid JAX least-squares plan

> Namespace note (2026-06-23): The implementation is now available under `swanx`; `swxps` references below are retained as historical compatibility paths.

## Goal

Fit Sample 13 reflectivity and the C 1s, Ni 3p, and La 4d rocking curves with
bounded SciPy TRF, a fully JAX-traceable forward model, and one fixed-capacity
unified layer grid. Generated artifacts belong under `runs/sample_13`.

## Physics background

Preserve the maintained cap model
`vacuum/C/LNO-1/LNO-2/LNO-bottom/[STO/LNO]x40/STO`, the current background
subtraction, normalized experimental rocking curves, dataset weights, grazing
angles, optical constants, IMFPs, and layer-selective emission. The fixed grid
uses cell-centered roughness grading, transfer-matrix fields, midpoint
attenuation, and XPS integration. Per-layer cell counts are derived once from
all fitted thickness capacities, so parameter changes alter widths and material
values without altering array shapes or recompiling JAX.

## Files to create or modify

- New maintained runner under
  `case_studies/sample_13/jax_least_squares_fixed_grid/`.
- Focused fixed-shape Sample 13 runner tests under `tests/`.
- Generated fit artifacts under `runs/sample_13/jax_least_squares_fixed_grid/`.
- `docs/PROJECT_STATE.md` and `docs/TODO.md` after validation.

## Implementation steps

1. Reuse the maintained Sample 13 preprocessing, parameter bounds, cap stack,
   core-level definitions, and output helpers.
2. Build an independent capacity thickness for every finite nominal layer,
   including both coupled top-LNO slabs and every graded superlattice layer.
3. Express repeat-dependent thicknesses, roughnesses, optical grading, fields,
   concentration/IMFP grading, attenuation, and RC normalization in JAX.
4. Compile residual and Jacobian once, verify fixed shapes and NumPy unified
   parity, and assert compilation counts remain one during fitting.
5. Run bounded TRF from the promoted Sample 13 result and save diagnostics,
   curves, plots, covariance, grid plan, timings, and parity metrics.

## Tests

- Capacity plan accepts the full fitted thickness domain and preserves cell
  count across distinct parameter vectors.
- Sample 13 fixed-grid reflectivity and RC arrays have fixed finite shapes.
- JAX curves agree with the maintained NumPy unified implementation.
- Full regression suite remains passing.

## Validation

Report initial and final objectives, optimizer status, nfev/njev, compilation
count and time, optimization time, grid size, parameter positions, per-curve
contributions, and JAX/NumPy parity. Treat the experimental result as
provisional even if numerical convergence succeeds.

## Progress log

- 2026-06-23: Read repository architecture and Sample 13 maintained runners;
  selected the promoted all-RC result as the new fixed-grid starting point.
- 2026-06-23: Implemented the 1,092-cell fixed-capacity model, removed
  midpoint-interface tie ambiguity with conservative capacities, and obtained
  JAX/NumPy curve parity at approximately `1e-15`.
- 2026-06-23: Full tests passed (`117 passed`). The 60-evaluation fit converged
  by `ftol` after 34 evaluations with one residual and one Jacobian compilation.
  Objective decreased 4.57%, but 14 of 18 parameters reached within 1% of a
  bound, so the result is diagnostic rather than canonical.
- 2026-06-23: Began a separate expanded-bounds experiment for the 13 parameters
  whose physical values pressed the baseline limits. Fraction parameters remain
  constrained to `[0, 1]` by construction.
- 2026-06-23: The expanded fit converged in 36 evaluations at objective
  `0.00646408`, 19.7% below the baseline fixed-grid result. Rank improved from
  15 to 16 and the condition number fell from `3.5e20` to `1.6e18`, but 12 of
  18 parameters remained within 1% of a bound. Reflectivity and C 1s improved;
  Ni 3p and La 4d worsened. Do not widen bounds again without revising weights
  or parameterization.
