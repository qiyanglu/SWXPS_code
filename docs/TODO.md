# TODO

Last updated: 2026-06-22

## Completed unified-grid milestone

- Added user-configurable `LayerSlicingPolicy`; defaults are 10 minimum cells
  and 2 Angstrom maximum cell thickness.
- Added adaptive grids and fixed-capacity plans for JAX/fitting shape stability.
- Shared one cell-centered grid across roughness optics, fields,
  concentration/IMFP, attenuation, and midpoint RC integration.
- Added optional `slicing=` propagation through requests and `FittingProblem`.
- Preserved all legacy `field_step` and `roughness_step` behavior.
- Added focused planner, physics, Fresnel, fitting, and NumPy/JAX parity tests.
- Verified one JAX compilation across a 2-6 Angstrom thickness sweep.
- Full regression result: 113 passed, 46 pre-existing warnings.

## Completed synthetic comparison

- Added a reproducible legacy-versus-unified comparison runner.
- Saved old/new/difference plot, pointwise CSV, and numerical summary under `runs/`.
- Fixed the overlapping-roughness optical grading mismatch revealed by comparison.
- Added exact identical-grid optical parity coverage.
- Full regression result: 114 passed.

## Next review and adoption steps

- Review the synthetic comparison figure and numerical summary.
- Add a concise user example to an appropriate maintained tutorial if desired.
- Migrate one maintained Sample 12 or Sample 13 runner to a fixed capacity plan.
- Compare its legacy and unified-grid curves, objective, compile count, runtime,
  and memory before migrating any other runner.
- Consider per-layer maximum cell thickness only if that case study demonstrates
  a concrete need; the current public value is global.

## Scientific validation priorities

- Audit experimental RC preprocessing and normalization against raw data.
- Quantify sensitivity to weights, bounds, initialization, and local minima.
- Check fitted structure and angular offsets against independent expectations.
- Keep experimental results provisional until these checks are documented.

## Small maintenance items

- Replace deprecated `np.trapz` in a separate parity-tested change.
- Move completed long-form plans to `docs/history/` when no longer active.

## Session handoff checklist

- Update this file and `docs/PROJECT_STATE.md`.
- Record tests, benchmarks, decisions, branch/commit, and blockers.
- Commit handoff documentation with the work it describes.
- Push required commits before switching computers.
