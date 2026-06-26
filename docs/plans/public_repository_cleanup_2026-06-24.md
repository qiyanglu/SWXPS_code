# Public repository cleanup

Status: Implemented and validated (2026-06-24).

> Current note (2026-06-26): `case_studies/` was later made local/private and ignored by Git. The earlier decision to keep it tracked is historical, not current repository policy.

## Goal

Reduce top-level clutter without removing the experimental `case_studies/`
tree or rewriting Git history. Keep generated runs and archives local, and
move useful standalone fitting demonstrations into the maintained examples.

## Physics background

This is repository organization only. No reflectivity, field, XPS, fitting, or
diagnostics behavior changes.

## Files to create or modify

- Move the two tracked `scripts/` demonstrations to `examples/fitting/`.
- Update them to use the `swanx` namespace.
- Update example, root, roadmap, and handoff documentation.

## Implementation steps

1. Preserve `case_studies/` as requested.
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

- 2026-06-24: Scope confirmed; `case_studies/` will remain tracked and no
  history rewrite will be performed.
- 2026-06-24: Moved and modernized both fitting examples; each recovered the 24 Angstrom synthetic truth. Case studies remain tracked and runs/archive remain ignored.
