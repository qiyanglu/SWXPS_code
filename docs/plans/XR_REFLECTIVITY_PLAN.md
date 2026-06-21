# SWXPS active milestone plan

## Goal

Maintain the validated reflectivity, field, SW-XPS, and fitting platform while
improving physical validation, runtime evidence, and reproducibility of
experimental case studies.

## Physics background

Stable conventions are documented in `AGENTS.md` and `docs/architecture.md`.
The complete derivations and chronological record are preserved in
`docs/history/XR_REFLECTIVITY_DEVELOPMENT_LOG.md`.

## Files to create or modify

Future milestones must name their scoped package, test, example, benchmark, or
case-study files. Generated fit output goes to `runs/`.

## Implementation steps

1. Preserve NumPy/JAX numerical parity and reflectivity tests.
2. Review experimental rocking-curve preprocessing and normalization.
3. Validate fitted structures against independent physical expectations.
4. Profile representative workflows before restructuring performance-critical code.
5. Add cross sections or polarization only through separate planned milestones.
6. Keep canonical case-study results compact and reproducible.

## Tests

- Run the full test suite for every package change.
- Add parity tests for new backend behavior.
- Retain the four reflectivity validations from `AGENTS.md`.

## Validation

Experimental results are not quantitative until bounds, weights, optical
constants, IMFPs, chemistry, and optimizer sensitivity have been reviewed.

## Progress log

- 2026-06-21: Tutorials, case studies, benchmarks, runs, and archives were
  separated into dedicated top-level folders.
- 2026-06-21: Added a representative stage-by-stage performance benchmark and
  bounded caches for optical-constant and IMFP tables.
- 2026-06-21: Full validated suite contains 91 passing tests.
- Remaining: physical review of experimental RC normalization and fit robustness.
