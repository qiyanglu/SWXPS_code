# Stage 7 API consistency

## Goal

Align the SWANX name, fitting defaults, README guidance, maintained examples,
and roadmap without changing physics algorithms or moving implementation code.

## Physics background

Unified slicing uses one layer grid for interface grading, optical fields, and
XPS integration. The explicit `slicing=None` mode retains the legacy separate
fixed-step discretizations for regression and compatibility.

## Files to create or modify

- `src/swanx/_fitting.py`
- focused fitting/default tests
- `README.md` and README-linked examples
- `docs/roadmap.md`, `docs/PROJECT_STATE.md`, and `docs/TODO.md`

## Implementation steps

1. Make `FittingProblem` use `LayerSlicingPolicy` by default and apply the same
   legacy-step validation as high-level simulation requests.
2. Preserve explicit legacy behavior and annotate affected maintained legacy
   runners with `slicing=None`.
3. Rewrite README identity, API, optimization, installation, and example text.
4. Refresh the roadmap and session-continuity documents.
5. Run targeted and full regression tests.

## Tests

- Verify default fitting propagates unified slicing.
- Verify explicit `slicing=None` propagates legacy mode.
- Verify non-default legacy steps are rejected with unified slicing.
- Run the requested targeted suite and then the full suite.

## Validation

Existing numerical tests must remain unchanged in result. This pass changes
only default selection, validation, imports in maintained examples, and docs.

## Progress log

- 2026-06-24: Audited current defaults, README examples, and roadmap; confirmed
  the fitting/default mismatch and stale roadmap text.
- 2026-06-24: Implemented the unified fitting default and matching validation;
  preserved legacy runners with explicit `slicing=None`; rewrote README and
  roadmap guidance; migrated linked tutorials to `swanx` imports.
- 2026-06-24: Requested targeted suite passed (29 tests). Full suite passed
  with 177 tests and 1 expected failure. No physics algorithm was changed.
