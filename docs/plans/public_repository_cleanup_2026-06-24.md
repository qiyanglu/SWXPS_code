# Public repository cleanup

> Current repository note (2026-06-26): `case_studies/` is local/private experimental space ignored by Git; any older wording about tracked case studies is historical context, not current policy.

> Current examples note (2026-06-29): fitting demonstrations now live under
> `examples/04_fitting/`.

Status: Implemented and validated (2026-06-24).

## Goal

Historical goal: reduce top-level clutter without removing the experimental
`case_studies/` tree or rewriting Git history. Current policy is different:
`case_studies/` is local/private and ignored by Git. Generated runs and
archives remain local, and useful standalone fitting demonstrations belong in
the maintained examples.

## Physics background

This is repository organization only. No reflectivity, field, XPS, fitting, or
diagnostics behavior changes.

## Files to create or modify

- Move the two tracked `scripts/` demonstrations to `examples/04_fitting/`.
- Update them to use the `swanx` namespace.
- Update example, root, roadmap, and handoff documentation.

## Implementation steps

1. Historical scope preserved `case_studies/` as requested at that time; current policy keeps it local/private and ignored.
2. Confirm `runs/` and `archive/` remain ignored except contributor guidance.
3. Move and modernize the two useful scripts.
4. Run both examples and the complete test suite.
5. Commit and push Stage 3 plus this cleanup.

## Tests

- Both moved examples execute successfully.
- `python -m pytest` passes.

## Validation

The examples should recover the same synthetic film thickness and reduce their
losses as before. No package numerical code is modified.

## Progress log

- 2026-06-24 historical result: scope kept `case_studies/` tracked at that time and no
  history rewrite was performed.
- 2026-06-24: Moved and modernized both fitting examples; each recovered the 24 Angstrom synthetic truth. Case studies were tracked at that time; current policy keeps them local/private and ignored. Runs/archive remain ignored.
