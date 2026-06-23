# Synthetic unified-grid JAX least-squares plan

## Goal

Fit the maintained synthetic C/[LNO/STO]x20/STO reflectivity and four RCs with
bounded SciPy TRF, a fully JAX-traceable unified-grid forward model, and JAX
Jacobians. Start deliberately far from the true parameters and verify that the
fixed grid prevents recompilation.

## Physics background

Use the fixed plan derived from all fitted thickness upper bounds with
`min_slices=10` and `max_slice_thickness=2.0` Angstrom. The JAX model must use
the same nearest-interface optical grading, shared cell centers, XPS property
grading, midpoint attenuation, and RC normalization as the high-level unified
model.

The residual contains log10 reflectivity plus normalized La 4d, O 1s, Ti 2p,
and C 1s blocks. Existing dataset weights and bounds are reused. The bad start
must remain strictly inside the declared bounds.

## Files to create or modify

- `src/swxps/jax_least_squares.py` for residual/Jacobian compilation counters;
- `src/swxps/__init__.py` for the counter export;
- focused counter tests;
- new `benchmarks/synthetic_c_lno_sto/fit_unified_jax_least_squares.py`;
- generated local results under
  `runs/synthetic_c_lno_sto/unified_jax_least_squares/`;
- `docs/PROJECT_STATE.md` and `docs/TODO.md` after the run.

## Implementation steps

1. Count Python traces at the JIT-wrapped residual and Jacobian boundaries.
2. Build fixed nominal-to-cell mappings once from the upper-bound capacity stack.
3. Express cell widths, centers, roughness grading, transfer matrices,
   concentration/IMFP grading, attenuation, and RC integration in JAX.
4. Compile residual and Jacobian once at the poor initial point.
5. Run bounded TRF and assert counter values do not increase.
6. Re-evaluate the best parameters with the maintained high-level NumPy unified model.
7. Save history, parameters, curve data, plots, timings, and compilation counts.

## Tests

- Counter increments once for the first fixed-shape residual trace and once for
  the first fixed-shape Jacobian trace.
- Repeated calls with changed values but identical shapes do not increment it.
- Fit residual/Jacobian shapes remain fixed and finite.
- Final parameters remain inside bounds and final cost is below initial cost.
- JAX and high-level NumPy unified curves agree at the final parameters.
- Full regression suite remains passing.

## Validation

Report initial/final cost, optimizer status, nfev/njev, compilation count,
compile time, optimization time, parameter recovery, curve errors, and
high-level NumPy parity. A successful fixed-shape run must show exactly one
residual compilation and one Jacobian compilation.

## Progress log

- 2026-06-23: Created the execution plan; no fitting run yet.
