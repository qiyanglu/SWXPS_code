# Stage 2 subpackage implementation migration

Status: Implemented and validated (2026-06-23).

## Goal

Move the low-risk diagnostics, slicing, and profile implementations into their
public `swanx` subpackages while preserving every flat `swanx.*` and legacy
`swxps.*` import used by current scripts.

## Physics background

This change moves Python modules only. Layer-grid construction, concentration
profiles, covariance calculations, plotting behavior, and all numerical
physics must remain unchanged.

## Files to create or modify

- `src/swanx/diagnostics/{covariance,plots,reports}.py`
- `src/swanx/stack/{slicing,profiles}.py`
- Flat `src/swanx/{slicing,profiles,_diagnostics}.py` compatibility shims
- Subpackage initializers, safe internal imports, and `src/swxps` aliases
- Focused module-location tests and current public documentation

## Implementation steps

1. Move slicing and profile bodies into `swanx.stack`.
2. Split covariance/numerical diagnostics from plotting helpers.
3. Re-export existing report helpers without moving unrelated fitting code.
4. Keep top-level and both generations of flat imports working by identity.
5. Run focused import tests and the complete regression suite.

## Tests

- Canonical and compatibility paths expose identical classes/functions.
- Top-level beginner exports remain available.
- Diagnostics covariance and plot regression suites remain green.
- `python -m pytest` passes.

## Validation

No forward-model or fitted numerical result should change. Existing physics,
JAX parity, slicing, profile, and diagnostics tests provide validation.

## Progress log

- 2026-06-23: Dependency inventory started; stack package initialization order
  identified as the main circular-import constraint.
- 2026-06-23: Moved slicing/profiles, split diagnostics covariance/plots/reports, added lazy cycle-safe stack exports and compatibility shims, and passed the full suite (153 passed, 1 expected failure).
