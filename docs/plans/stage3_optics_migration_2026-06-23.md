# Stage 3 optics implementation migration

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

Status: Implemented and validated (2026-06-23).

## Goal

Move Parratt reflectivity, transfer-matrix/electric-field, and unified-grid
optics implementations into `swanx.optics` while preserving flat `swanx.*`,
legacy `swxps.*`, and beginner-facing APIs.

## Physics background

This is a module-location change only. The validated s-polarized Parratt,
transfer-matrix, roughness-grading, field-profile, and unified-grid equations
and conventions must remain unchanged.

## Files to create or modify

- `src/swanx/optics/{parratt,fields,unified_grid}.py`
- Flat `src/swanx/{reflectivity,fields,unified_grid}.py` shims
- `swanx.optics` exports, safe internal imports, and compatibility tests
- README, architecture, roadmap, and handoff status

## Implementation steps

1. Move the three implementation bodies and fix package-relative imports.
2. Add thin flat-module shims.
3. Expose existing unified simulation functions lazily to avoid cycles.
4. Update safe internal imports to canonical optics modules.
5. Verify canonical/flat/legacy object identity and run the full suite.

## Tests

- New and old Parratt, fields, and unified-grid paths share objects.
- Existing optics, simulation, XPS, and backend regression tests pass.
- `python -m pytest` passes.

## Validation

No numerical outputs should change. Existing Fresnel, Bragg, field, roughness,
unified-grid, NumPy/JAX parity, and fitting tests provide validation.

## Progress log

- 2026-06-23: Dependency inventory started; unified simulation entry points
  identified for lazy re-export from the canonical unified-grid module.
- 2026-06-23: Moved Parratt, fields, and unified-grid bodies; added flat/legacy shims and lazy unified entry points; verified object identity and passed the full suite (158 passed, 1 expected failure).
