# Stage 4 XPS implementation migration

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

Status: Complete (2026-06-24).

## Goal

Move XPS attenuation, intensity/property sampling, rocking-curve, and unified
grid integration implementations into `swanx.xps` while preserving flat
`swanx`, optics-grid, and legacy `swxps` import identities.

## Physics background

This is a module-location change only. Electron attenuation, trapezoidal XPS
integration, rough-interface property grading, normalized rocking curves, and
cell-centered midpoint integration must retain their validated algorithms and
physical conventions unchanged.

## Files to create or modify

- `src/swanx/xps/{attenuation,intensity,rocking_curve,grid}.py`
- `src/swanx/xps/__init__.py` and flat `src/swanx/_xps.py` shim
- `src/swanx/optics/unified_grid.py` compatibility re-exports
- Safe internal imports in simulation and stack-profile modules
- Stage 3/4 location tests and minimal README/architecture/handoff updates

## Implementation steps

1. Split the existing `_xps.py` implementation without changing function
   bodies or equations.
2. Move cell-centered attenuation and grid XPS integration verbatim from the
   optics unified-grid module.
3. Make `_xps.py` and optics unified-grid names thin identity-preserving
   compatibility re-exports.
4. Use lazy high-level simulation exports from `swanx.xps` to avoid circular
   imports when simulation imports canonical XPS submodules.
5. Update internal imports, location tests, docs, and run the complete suite.

## Tests

- Canonical, package, flat, optics compatibility, and `swxps` paths share
  function/class objects.
- Canonical objects report `swanx.xps.*` implementation modules.
- Existing attenuation, intensity, roughness-grading, RC, unified-grid,
  simulation, fitting, and JAX regressions remain unchanged.
- `python -m pytest` passes.

## Validation

No numerical algorithms or outputs should change. Object-identity assertions
guard against accidental wrapper implementations or duplicated code.

## Progress log

- 2026-06-24: Audited the Stage 1-3 layout and identified the required lazy
  simulation exports needed to keep `swanx.xps` cycle-free.
- 2026-06-24: Relocated the implementation bodies, converted old locations to
  identity-preserving shims, and passed the focused XPS/optics/simulation suite
  (35 tests).
- 2026-06-24: Updated public and handoff documentation and passed the complete
  regression suite (163 passed, 1 expected failure).
