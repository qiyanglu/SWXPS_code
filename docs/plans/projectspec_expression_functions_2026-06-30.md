# ProjectSpec Safe Expression Functions Plan

Date: 2026-06-30

## Goal

Support SWOPT/Sample 12-style scalar formulas in ProjectSpec YAML without
allowing arbitrary Python execution. This should make layer thickness and
roughness expressions expressive enough for synthetic and experimental
multilayer fitting cases while keeping validation predictable.

## Indexing Decision

`repeat_index` remains 1-based inside repeat blocks because that is the
documented ProjectSpec convention and matches generated layer IDs such as
`lno_1`.

A zero-based index is not strictly necessary: users can write
`repeat_index - 1` whenever they need it. Still, ProjectSpec will add
`repeat_index0` as a convenience variable for formulas that naturally use
zero-based layer coordinates, especially SWOPT-style transition and gradient
expressions. This is readability and portability support, not a physics change.

## Scope

- Extend the safe expression AST whitelist to allow calls to a small set of
  built-in numeric functions.
- Add scalar functions:
  - `min(...)`
  - `max(...)`
  - `sqrt(x)`
  - `erf(x)`
  - `linear_map(x, x0, x1, y0, y1)`
  - `transition_erf(x, start, end, center, width)`
- Add `repeat_index0` beside the existing `repeat_index` and `layer_index`
  expression variables.
- Update ProjectSpec docs to describe function support and indexing clearly.
- Add focused tests for allowed functions, zero-based repeat indexing, unknown
  functions, and unsafe calls.

## Non-Goals

- Do not add arbitrary user-defined functions directly in YAML.
- Do not auto-generate JAX residuals from YAML.
- Do not change physical units, layer ordering, roughness conventions, or the
  existing 1-based `repeat_index` behavior.

## Validation

- Run focused ProjectSpec workflow tests.
- Confirm existing repeat-index behavior remains unchanged.
- Confirm unsafe expressions such as imports or attribute calls fail during
  validation.

## Result

Implemented on 2026-06-30:

- ProjectSpec expressions now allow the safe scalar functions listed above.
- `repeat_index` remains 1-based; `repeat_index0` is available as a
  zero-based convenience variable.
- Sample 12 YAML case-study formulas now use `transition_erf` and
  `linear_map`; a follow-up rewrite expanded the YAML into a one-to-one audit
  map of the maintained bounded TRF JAX least-squares code, but the case still
  uses its local runner for preprocessing and fixed-shape residual construction.
- Focused ProjectSpec workflow tests passed with `29 passed`.
- Sample 12 YAML setup-only smoke passed and reported 18 parameters, 227
  residuals, and matching JAX/NumPy initial objectives.
- Full test suite passed with `252 passed, 1 xfailed, 1 warning`.
