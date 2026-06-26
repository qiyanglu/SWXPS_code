# Final API freeze and user-experience consolidation

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

Status: Complete (2026-06-24).

## Goal

Freeze one small top-level `swanx` API and make `import swanx as sx` the only
recommended README workflow, with JAX fitting presented as primary and
Bayesian optimization as a baseline comparison.

## Frozen top-level API

- `SimulationStack`, `StackLayer`
- `ReflectivityRequest`, `RockingCurveRequest`, `CoreLevelRequest`
- `simulate_reflectivity`, `simulate_rocking_curves`
- `compute_parameter_diagnostics`
- `plot_parameter_estimates`, `plot_correlation_matrix`

## Compatibility and implementation constraints

- Do not move or modify physics/numerical implementations.
- Keep internal submodules importable for maintained advanced code, but do not
  recommend them as competing user entry points.
- Preserve the broad temporary `swxps` compatibility surface independently of
  the frozen `swanx.__all__` so existing regression and legacy scripts remain
  functional.
- Update maintained scripts that currently import non-frozen names directly
  from `swanx` to their canonical implementation modules.

## Steps

1. Capture the former broad facade as an internal legacy compatibility module.
2. Replace `swanx.__init__` with exactly the frozen public exports.
3. Point `swxps` compatibility exports at the legacy facade.
4. Update namespace tests and maintained `from swanx import ...` consumers.
5. Rewrite README around one `import swanx as sx` getting-started path and a
   JAX-first optimization philosophy.
6. Update architecture, project state, and TODO documentation.
7. Run public-API, focused, and complete regression suites.

## Result

- `swanx.__all__` is exactly the requested ten-name surface.
- README contains one official `import swanx as sx` workflow and explicitly
  presents JAX as primary and BO as baseline.
- Historical acceptance criterion: broad compatibility was temporarily routed through `swxps`; current imports must use `swanx`.
- Four maintained advanced scripts compile with canonical internal imports.
- Focused API/namespace verification: 28 passed.
- Full verification: 173 passed and 1 expected failure.
