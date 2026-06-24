# TODO

Last updated: 2026-06-24

## Completed first-stage `swanx` namespace migration

- Renamed the distribution and primary implementation namespace to `swanx`.
- Preserved `swxps` and all former `swxps.*` imports as compatibility aliases.
- Added stack, optics, XPS, fitting, diagnostics, I/O, and workflow discovery facades.
- Updated the GitHub README, architecture, roadmap, and plan status notes.
- Added namespace identity/import tests and verified editable installation.
- Full namespace-migration regression result: 137 passed and 1 expected failure.
- Diagnostics-plot regression result: 142 passed and 1 expected failure.

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

## Namespace follow-up

- Keep new code and documentation on `swanx`; migrate maintained examples gradually.
- Add deprecation warnings and a removal release only through a separate breaking-change plan.
- Do not move or refactor physics kernels merely to deepen the new subpackage layout.

## Completed Sample 12 diagnostics sanity check

- Migrated the maintained Sample 12 TRF runner to `swanx` imports.
- Added normalized uncertainty, validated correlation, and correlation CSV outputs.
- Ran in an isolated folder with canonical promotion disabled.
- Verified exact correlation symmetry/unit diagonal and `|rho| <= 1`.
- Recorded near-degenerate roughness and angular-offset parameter pairs.
- Full regression result: 153 passed and 1 expected failure.

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

## Completed diagnostics namespace sanity check

- Migrated the synthetic fixed-grid JAX/TRF runner imports to `swanx`.
- Generated uncertainty and correlation plots through `swanx.diagnostics`.
- Normalized uncertainty plots by each finite parameter range and added raw bound endpoint labels; retained `normalization=None` for raw values.
- Moved the CI legend above the axes and increased plot font, marker, and bound-bar sizes.
- Reproduced the canonical optimum and one-time JAX compilation counts.
- Confirmed a rank-6/7 Jacobian; treat substrate-roughness uncertainty as unidentifiable rather than precise.

## Completed public repository cleanup

- Retained `case_studies/` and avoided destructive history rewriting.
- Confirmed `runs/` and `archive/` generated contents remain ignored.
- Moved two standalone fitting scripts into `examples/fitting/`.
- Migrated both examples to `swanx` and verified 24 Angstrom recovery.

## Completed Stage 3 optics migration

- Moved Parratt, field/transfer-matrix, and unified-grid implementation bodies into `swanx.optics`.
- Preserved flat `swanx.*`, legacy `swxps.*`, and beginner top-level APIs.
- Added lazy access to existing high-level unified simulation entry points.
- Added canonical-location and object-identity tests.
- Deferred XPS, simulation, fitting, and workflow implementation moves.
- Full regression result: 158 passed and 1 expected failure.

## Completed Stage 4 XPS migration

- Split attenuation, intensity/property sampling, rocking-curve, and grid XPS
  implementations into focused `swanx.xps` modules.
- Preserved flat `swanx._xps`, former optics-grid, and legacy `swxps.*`
  imports as identity-preserving compatibility paths.
- Updated simulation and stack-profile internals to import canonical XPS
  modules directly.
- Added canonical-location, compatibility identity, and lazy high-level export
  tests without changing numerical algorithms.
- Deferred simulation, fitting, and workflow implementation moves.
- Full regression result: 163 passed and 1 expected failure.

## Completed Stage 2 subpackage migration

- Moved slicing and profile implementation bodies into `swanx.stack`.
- Split diagnostics into covariance, plotting, and report namespaces.
- Preserved flat `swanx.*`, legacy `swxps.*`, and top-level beginner exports.
- Added canonical-location and object-identity tests.
- Deferred optics, XPS, simulation, and fitting implementation moves.
- Full regression result: 153 passed and 1 expected failure.

## Completed covariance/correlation hardening

- Recompute least-squares diagnostics covariance from final residuals/Jacobian by default.
- Validate and symmetrize computed and externally supplied covariance matrices.
- Reject non-finite, negative-variance, and materially indefinite inputs.
- Enforce symmetric, bounded correlations and clip only tiny roundoff excursions.
- Align stored optimizer covariance with `rcond=1e-12` and explicit symmetry.
- Added failure-mode regression coverage; full result: 149 passed, 1 expected failure.

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
