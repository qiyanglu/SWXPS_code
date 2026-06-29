# Docs consistency sweep

## Goal

Scan README, AGENTS.md, active docs, and plans for stale or inconsistent
descriptions after ProjectSpec v1.2 landed.

## Scope

- Update active docs: README, AGENTS.md, architecture, user guide, roadmap,
  project state, TODO, and active plans.
- Keep `docs/history/` as a historical archive, but make active guidance point
  to current behavior.
- Add current-status notes to older plans whose old namespace or case-study
  policy language would otherwise be confusing.

## Non-goals

- No physics, numerical, fitting, or API behavior changes.
- No edits to archived history files except through active index/context docs.

## 2026-06-27 sweep notes

- Updated AGENTS.md with ProjectSpec v1.2 workflow, output, optional dependency,
  and fitting-backend guidance.
- Cleaned active ProjectSpec docs/plans to prefer `swanx init`, project-local
  outputs, `report.md`, JAX least-squares as recommended, and BO as optional.
- Repaired stale validation text and Windows path escape artifacts in
  `docs/PROJECT_STATE.md`.

- 2026-06-27: Refreshed active docs after ProjectSpec v1.2 so README
  starts with self-contained `swanx init`, user guide documents initializer
  choices/inspect, and active plans avoid stale exact pytest counts.

## 2026-06-29 sweep notes

- Rechecked maintained docs after the examples were realigned to the synthetic
  C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 benchmark case.
- Left `docs/history/` and older `docs/plans/` snapshots historical when they
  already carry current-status notes, but corrected active README, architecture,
  roadmap, guide/reference wording, and mojibake in the public README.
- Follow-up: changed `swanx init` from the older single-film starter to the
  C/LaNiO3/SrTiO3 superlattice case, then made the default init project an
  active JAX least-squares fitting starter. Also removed maintainer
  storage-policy text from the examples README.
- Follow-up: after the ProjectSpec off-peak RC normalization fix, clarified
  active docs so `rocking_curve_offpeak_mask` is described as shared by
  experimental and simulated rocking-curve mean normalization, not only as a
  scoring exclusion.
- Follow-up: retired the repository-local `templates/` folder after `swanx init`
  and the maintained ProjectSpec examples became the supported starter surfaces.

- Follow-up: after adding the runnable ProjectSpec fitting example under
  `examples/04_fitting/projectspec_jax_least_squares/`, re-swept active docs so
  README, user guide, ProjectSpec reference, architecture, roadmap, project
  state, TODO, and docs consistency tests describe the same init/tutorial scope.
