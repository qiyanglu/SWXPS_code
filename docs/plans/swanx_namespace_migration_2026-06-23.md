# swanx namespace migration

Status: Implemented and validated (2026-06-23).

## Goal

Make `swanx` (*standing-wave analysis for X-ray spectroscopy*) the primary
package and repository name while preserving the complete `swxps` import
surface as a temporary compatibility namespace.

## Physics background

This is a namespace, packaging, and documentation change only. The validated
Parratt, transfer-matrix, roughness, unified-grid, attenuation, rocking-curve,
and fitting algorithms and conventions must remain unchanged.

## Files to create or modify

- Move maintained modules from `src/swxps/` to `src/swanx/`.
- Add beginner-facing `swanx` subpackage facades and top-level exports.
- Retain `src/swxps/` as a thin module-alias compatibility package.
- Update `pyproject.toml`, `README.md`, architecture/roadmap/handoff docs, and
  relevant plan status notes.
- Add `tests/test_namespace_imports.py`.

## Implementation steps

1. Establish `swanx` as the canonical implementation namespace.
2. Add `stack`, `optics`, `xps`, `fitting`, `diagnostics`, `io`, and
   `workflows` facades using existing objects only.
3. Alias every former `swxps.*` module to its canonical `swanx.*` module.
4. Update packaging and GitHub-facing documentation.
5. Add identity and import compatibility tests.

## Tests

- New and old package imports resolve.
- New facade imports expose the existing implementation objects.
- Old and new imports have object identity.
- The complete regression suite passes with `python -m pytest`.

## Validation

No numerical output should change because the implementation bodies are not
being rewritten. Existing physics and backend regression tests provide the
behavioral validation.

## Progress log

- 2026-06-23: Inventory completed; namespace conflicts and compatibility
  requirements identified.

- 2026-06-23: Created canonical and facade namespaces, added exact legacy module aliases, updated packaging and documentation, verified editable installation, and passed the full suite (137 passed, 1 expected failure).
- 2026-06-23: Completed a diagnostics sanity check with the synthetic LNO/STO JAX/TRF fit; generated uncertainty and correlation plots through `swanx.diagnostics` and reproduced the canonical fit/compilation results.
- 2026-06-23: Improved the public parameter-estimate plot with default bound-range normalization, raw endpoint labels, optional raw/value-label modes, focused coverage, and a regenerated synthetic diagnostic (141 passed, 1 expected failure).
- 2026-06-23: Refined uncertainty-plot layout by moving the CI legend above the axes and enlarging fonts, markers, and bound bars; regenerated the synthetic plot and passed the full suite (142 passed, 1 expected failure).
- 2026-06-23: Hardened covariance/correlation diagnostics, made Jacobian-derived covariance authoritative for least-squares adapters, aligned optimizer covariance regularization, regenerated the synthetic plot, and passed the full suite (149 passed, 1 expected failure).
