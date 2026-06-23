# Project structure reorganization plan

> Namespace note (2026-06-23): This completed repository plan predates the `swanx` rename. Maintained code now lives under `src/swanx`; `src/swxps` is a compatibility shim.

## Goal

Complete the repository cleanup by separating tutorial examples, experimental
case studies, synthetic/performance benchmarks, generated runs, and historical
archives into dedicated top-level folders without changing package physics.

## Physics background

No equations, layer conventions, numerical kernels, residual definitions, or
optimizer behavior will change. This milestone changes paths and documentation
only.

## Files to create or modify

- Move experimental folders from `examples/` to `case_studies/`.
- Move synthetic and performance folders from `examples/` to `benchmarks/`.
- Move `artifacts/runs` to `runs` and `artifacts/archive` to `archive`.
- Update maintained runner path constants, README files, `.gitignore`, and
  architecture/roadmap documentation.
- Do not modify numerical code under `src/swxps`.

## Implementation steps

1. Inventory every maintained reference to the old locations.
2. Move folders using explicit verified in-workspace paths.
3. Update repository-root discovery and cross-case imports in moved runners.
4. Update output paths to the top-level `runs` directory.
5. Update current documentation; leave historical documents unchanged.
6. Remove empty transitional folders and caches.

## Tests

- Run maintained runners with `--help` and bytecode disabled.
- Verify all current documentation commands resolve.
- Search outside `docs/history` for obsolete paths.
- Run `python -B -m pytest -q -p no:cacheprovider`.

## Validation

- `examples/` contains tutorials only.
- `case_studies/` contains Sample 12, Sample 13, and their comparison.
- `benchmarks/` contains synthetic and performance benchmarks.
- `runs/` and `archive/` are Git-ignored.
- Core package tests remain unchanged and pass.

## Progress log

- 2026-06-21: Completed the final structure reorganization after the initial
  artifact cleanup.
- 2026-06-21: Verified 33 maintained Python files, runner imports, and 85 tests.
