# Stage 6 slim simulation compatibility layer

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

Status: Complete (2026-06-24).

## Goal

Finish the simulation-layer separation by moving material lookup and
emitting-layer filtering utilities into `swanx.xps.utils`, leaving
`swanx.simulation` as public compatibility re-exports only.

## Current Stage 5 baseline

- Stack models already live in `swanx.stack.model`.
- Request/result classes and routing/legacy workflow bodies already live in
  `swanx.workflows.simulate`.
- `swanx.simulation` is already a thin shim, but still imports private workflow
  helpers that should not leak through the compatibility layer.
- Full Stage 5 verification: 167 passed and 1 expected failure.

## Physics constraint

Copy helper bodies verbatim and change imports only. Do not alter
reflectivity, electric-field, attenuation, XPS integration, normalization, or
slicing algorithms.

## Steps

1. Add `swanx.xps.utils` with `_values_by_material` and
   `_apply_emitting_layer_filter`.
2. Import those helpers from the canonical utility module in workflow and
   unified simulation code.
3. Remove private helper imports from `swanx.simulation`.
4. Add structural tests for the thin shim and exact compatibility/canonical
   reflectivity and SW-XPS result parity.
5. Update architecture and continuity documentation.
6. Run focused and full regression suites.

## Result

- Helpers now have one canonical implementation in `swanx.xps.utils`, shared
  by NumPy, JAX, and unified workflows.
- `swanx.simulation` defines no classes or functions and exposes no private
  helper utilities.
- Canonical, `swanx.simulation`, and `swxps.simulation` reflectivity and
  SW-XPS results match with exact array equality.
- Focused verification: 37 passed.
- Full verification: 171 passed and 1 expected failure.
