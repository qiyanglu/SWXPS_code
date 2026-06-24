# Default unified high-level slicing plan

> Stage 3 namespace status (2026-06-23): Unified-grid optics now live in `swanx.optics.unified_grid`; existing high-level unified simulation functions are lazily re-exported there.

> Current status (2026-06-23): Implemented and tested under the primary `swanx` namespace. `swxps` paths below are historical compatibility references.

## Goal

Make unified layer slicing the default for high-level reflectivity and
rocking-curve request objects while retaining the step-based path through
explicit `slicing=None`. Verify the actual JAX differentiation boundary.

## Physics background

Unified slicing assigns each positive finite nominal layer
`N_i = max(min_slices, ceil(t_i / max_slice_thickness))` cells. Defaults are
10 cells minimum and 2 Angstrom maximum cell thickness. One cell-centered grid
is shared by graded optics, fields, concentration/IMFP sampling, attenuation,
and XPS integration. Legacy mode retains separate `roughness_step` and
`field_step` discretizations for regression and compatibility.

## Files to create or modify

- `src/swxps/simulation.py` for request defaults and validation.
- Direct legacy-dependent tests, examples, and benchmark callers.
- Focused default/legacy and JAX differentiability tests.
- `README.md`, `docs/architecture.md`, `docs/PROJECT_STATE.md`, and
  `docs/TODO.md`.

## Implementation steps

1. Use `field(default_factory=LayerSlicingPolicy)` for both request types.
2. Reject non-default step arguments when unified slicing is active.
3. Mark callers that require historical step behavior with `slicing=None`.
4. Probe gradients through high-level fixed-plan materialization and add a
   narrower fixed-plan JAX-native array test if the generic path cannot trace.
5. Run the complete regression suite.

## Tests

- Default requests own non-null unified policies.
- Explicit `slicing=None` preserves step-based behavior.
- Non-default legacy steps are rejected in unified mode.
- Fixed-plan JAX-native reflectivity values and gradients are finite under JIT.
- Unsupported high-level materialization tracing is recorded as an expected
  failure rather than described as working.

## Validation

Default high-level curves must remain finite and physically bounded. Legacy
regressions must retain their previous results when explicitly selected.
Documentation must distinguish JAX-backed forward execution from full tracing
through Python/NumPy grid materialization.

## Progress log

- 2026-06-23: Audited request construction and identified direct callers that
  depend on legacy step semantics. Confirmed `FittingProblem` passes its
  `slicing` choice explicitly and is outside this focused default change.
- 2026-06-23: Made both request defaults unified, added step-argument
  validation, and marked legacy-dependent tests, examples, benchmarks, and
  case-study utilities with `slicing=None`.
- 2026-06-23: Fixed-plan JAX-native array materialization passed eager and JIT
  gradient checks. Generic high-level materialization raises the expected JAX
  `ConcretizationTypeError` at Python float conversion and is covered by a
  strict expected-failure test.
- 2026-06-23: Full regression result: 126 passed, 1 expected failure.
