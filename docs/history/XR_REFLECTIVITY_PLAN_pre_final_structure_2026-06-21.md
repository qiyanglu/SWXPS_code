# SWXPS active milestone plan

## Goal

Maintain the validated reflectivity, electric-field, SW-XPS, and fitting
platform while improving physical validation and reproducibility of experimental
case studies.

## Physics background

The stable conventions and validated equations are documented in `AGENTS.md`
and `docs/architecture.md`. The original detailed derivations and chronological
implementation record are preserved in
`docs/history/XR_REFLECTIVITY_DEVELOPMENT_LOG.md`.

## Files to create or modify

Future milestones should name only their scoped package, test, example, and
documentation files. Generated fit outputs must go to `artifacts/runs`, not the
maintained example tree.

## Implementation steps

1. Preserve NumPy/JAX numerical parity and the validated reflectivity tests.
2. Review normalized experimental rocking curves and preprocessing assumptions.
3. Validate fitted structures against independent physical expectations.
4. Add cross sections or polarization only through separate planned milestones.
5. Keep canonical case-study results compact and reproducible.

## Tests

- Run the full test suite for every package change.
- Add focused parity tests for every new numerical backend behavior.
- Retain the four required reflectivity validations from `AGENTS.md`.

## Validation

Experimental results are not quantitative until parameter bounds, residual
weights, optical constants, IMFPs, layer chemistry, and optimizer sensitivity
have been reviewed.

## Progress log

- 2026-06-21: The validated suite contains 85 passing tests.
- 2026-06-21: Repository cleanup separated active examples, generated runs,
  historical experiments, current documentation, and the original development log.
- Remaining: physical review of experimental RC normalization and fit robustness.
