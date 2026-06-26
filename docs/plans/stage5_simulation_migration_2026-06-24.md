# Stage 5 simulation-layer migration

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

Status: Complete (2026-06-24).

## Goal

Separate material-labeled stack data models from high-level simulation
workflows while preserving every maintained public and legacy import identity.

## Physics constraint

This is a source-location change only. Reflectivity, field reuse, XPS
integration, normalization, slicing selection, and validation algorithms must
remain unchanged.

## Canonical locations

- `swanx.stack.model`: `StackLayer`, `SimulationStack`, `stack_from_layers`.
- `swanx.workflows.simulate`: request/result classes, high-level simulation
  functions, and their current private workflow helpers.
- `swanx.simulation`: identity-preserving compatibility re-exports.

## Implementation steps

1. Copy the existing model and workflow bodies into the new canonical modules.
2. Convert `swanx.simulation` to a thin compatibility shim.
3. Point internal imports at `stack.model` or `workflows.simulate` as
   appropriate, without importing through top-level `swanx`.
4. Export preferred objects from `swanx.stack` and `swanx.workflows`; retain
   beginner-facing top-level exports.
5. Add canonical-location, compatibility-identity, and minimal reflectivity
   smoke tests.
6. Update README, architecture, project state, and TODO documentation.
7. Run focused tests and the full `python -m pytest` suite.

## Compatibility coverage

- `swanx.simulation` and legacy `swxps.simulation` resolve to canonical objects.
- Existing top-level `swanx` and `swxps` objects retain identity.
- Stage 2-4 lazy exports remain cycle-free.
- No wrapper implementations or duplicated runtime classes are introduced.

## Result

- Canonical stack and workflow modules were created with the existing
  implementation bodies.
- Compatibility, preferred subpackage, legacy, and top-level exports share
  object identity.
- An eager `swanx.workflows` cycle discovered during import validation was
  removed by making fitting and diagnostics facade exports lazy.
- Focused verification: 51 passed.
- Full verification: 167 passed and 1 expected failure.
