# TODO

Last updated: 2026-06-23

## Completed unified-grid milestone

- Added user-configurable `LayerSlicingPolicy`; defaults are 10 minimum cells
  and 2 Angstrom maximum cell thickness.
- Added adaptive grids and fixed-capacity plans for JAX/fitting shape stability.
- Shared one cell-centered grid across roughness optics, fields,
  concentration/IMFP, attenuation, and midpoint RC integration.
- Added optional `slicing=` propagation through requests and `FittingProblem`.
- Made unified slicing the request default and preserved legacy `field_step`
  and `roughness_step` behavior through explicit `slicing=None`.
- Documented that generic Python/NumPy grid materialization is not yet
  end-to-end JAX differentiable; fixed-plan JAX-native array models are.
- Added focused planner, physics, Fresnel, fitting, and NumPy/JAX parity tests.
- Verified one JAX compilation across a 2-6 Angstrom thickness sweep.
- Full regression result: 113 passed, 46 pre-existing warnings.
- Default-unified follow-up regression: 126 passed, 1 expected JAX
  materialization failure.
- Parameter-diagnostics follow-up regression: 133 passed, 1 expected JAX
  materialization failure.

## Completed synthetic comparison

- Added a reproducible legacy-versus-unified comparison runner.
- Saved old/new/difference plot, pointwise CSV, and numerical summary under `runs/`.
- Fixed the overlapping-roughness optical grading mismatch revealed by comparison.
- Added exact identical-grid optical parity coverage.
- Full regression result: 114 passed.

## Next review and adoption steps

- Review the synthetic comparison figure and numerical summary.
- Add a concise user example to an appropriate maintained tutorial if desired.
- Completed the first maintained Sample 13 fixed-capacity JAX/TRF migration.
- Review the Sample 13 fit's 14/18 near-bound parameters before migrating any
  other experimental runner.
- Compare a reduced/reparameterized Sample 13 model against the preserved
  legacy and fixed-grid results.
- Consider per-layer maximum cell thickness only if that case study demonstrates
  a concrete need; the current public value is global.

## Scientific validation priorities

- Audit experimental RC preprocessing and normalization against raw data.
- Quantify sensitivity to weights, bounds, initialization, and local minima.
- Check fitted structure and angular offsets against independent expectations.
- Keep experimental results provisional until these checks are documented.
- Audit why both independent angle offsets prefer first `+0.35 deg` and then
  the expanded `+0.50 deg` bounds; check angular calibration independently.
- Remove or constrain unidentifiable thickness-transition parameters when both
  fitted thickness deltas collapse to zero.
- Revisit roughness interpolation after all four endpoints separate to their
  opposite bounds under both ranges (`5 -> 2`, then `6 -> 1 Angstrom`).
- Review dataset weighting: expanded bounds improve weighted reflectivity and
  C 1s but worsen Ni 3p and La 4d.

## Small maintenance items

- Added reusable local parameter uncertainty/correlation/singular-value
  diagnostics; apply them to future experimental fits before promotion.
- Replace deprecated `np.trapz` in a separate parity-tested change.
- Move completed long-form plans to `docs/history/` when no longer active.

## Session handoff checklist

- Update this file and `docs/PROJECT_STATE.md`.
- Record tests, benchmarks, decisions, current Git status, and blockers.
- Leave revisions uncommitted unless the user explicitly requests a commit.
- If the user requests a commit or push, verify the exact scope and Git status first.
