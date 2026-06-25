# Example, case-study, and docs consistency sweep

## Goal

Check active examples, case-study runners, and documentation for stale package
names, outdated data paths, and inconsistencies after the SWANX namespace and
data-directory cleanup.

## Physics background

This is a maintenance pass only. It must not change reflectivity, standing-wave
field, XPS, slicing, fitting, or polarization physics. Any executable examples
should continue using the existing conventions: grazing incidence angle in
degrees, energy in eV, Angstrom layer dimensions, and `n = 1 - delta + i beta`.

## Files to create or modify

- Active example scripts and example docs if stale references are found.
- Active case-study runners if stale references are found.
- `README.md` and active files under `docs/` if project-state or workflow
  descriptions are stale.
- This plan file.

## Implementation steps

1. Scan examples, case studies, source, tests, and active docs for old `swxps`
   imports and old `OPC` / `IMFP` / `examples/data` paths.
2. Inspect any hits to distinguish active stale references from historical
   archive records.
3. Apply focused path/import/doc fixes only where current behavior is affected
   or active documentation is misleading.
4. Run targeted import/example checks and relevant tests.

## Tests

- Run focused tests around changed files.
- Run the maintained IO and tutorial examples if path changes affect them.

## Validation

The sweep is successful if active examples and case-study runners use `swanx`
and the current `data/OPC`, `data/IMFP`, and `data/curves` layout, while
historical records remain clearly historical.

## Progress log

- 2026-06-25: Started sweep before committing the polarization comparison work.
- 2026-06-25: Active examples and case-study scripts were scanned for `swxps`
  imports and stale `OPC` / `IMFP` / `examples/data` paths. No active stale
  examples or case-study runners were found; historical plan/history references
  were left intact.
- 2026-06-25: Updated active docs and AGENTS notes for implemented s/p/mixed
  polarization support and current root-level data layout.
- 2026-06-25: README-linked examples and full pytest suite passed.
