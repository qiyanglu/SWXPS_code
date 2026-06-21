# Repository cleanup plan

## Goal

Separate maintained source, reproducible examples, experimental inputs, canonical
results, generated runs, and historical artifacts so the repository is easier to
navigate without changing the numerical or physical behavior of `swxps`.

## Physics background

No physics model will be changed. The Parratt, transfer-matrix, roughness,
standing-wave XPS, fitting, and JAX implementations remain untouched. File moves
must preserve the existing angle, energy, length, refractive-index, layer-order,
and polarization conventions documented in `AGENTS.md`.

## Files to create or modify

- Documentation under `docs/` and the root `README.md`.
- `examples/README.md` and case-study README files.
- `.gitignore` for generated run directories and local caches.
- Example, benchmark, and case-study paths only where organization requires it.
- Generated caches, duplicate outputs, superseded runs, and historical scripts.

The package implementation under `src/swxps` and its tests are out of scope.

## Implementation steps

1. Preserve the current dirty worktree and classify existing files by purpose.
2. Keep compact tutorial examples under `examples/`.
3. Keep experimental inputs, current runners, and one canonical result per sample
   in clearly named case-study folders.
4. Move generated optimizer runs into a Git-ignored `runs/` hierarchy.
5. Move superseded scripts and historical outputs into a Git-ignored `archive/`
   hierarchy, retaining provenance locally without presenting them as maintained.
6. Remove Python, pytest, and editable-install caches.
7. Split the monolithic milestone document into current roadmap/architecture and
   historical material where practical, while retaining the original log in the
   archive.
8. Update all maintained documentation and path references.

## Tests

- `python -B -m pytest -q -p no:cacheprovider`
- Verify maintained README commands resolve to existing scripts.
- Search maintained documentation for obsolete example paths.
- Confirm `src/swxps` has no cleanup-related edits.

## Validation

- All existing tests pass.
- Raw experimental data and canonical best-result exports remain available.
- Maintained example scripts remain runnable from the repository root.
- Generated runs no longer dominate the active `examples/` tree.
- Git ignores future run artifacts without hiding maintained source or raw data.

## Progress log

- 2026-06-21: Audited the repository: 85 tests passed; `examples/` contained 644
  files and about 79.7 MiB, including 318 PNGs and about 10.1 MiB of exact
  duplicates.
- 2026-06-21: Began the repository cleanup with core package code explicitly out
  of scope.
