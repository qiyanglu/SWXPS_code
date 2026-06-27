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

- 2026-06-27: Refreshed active docs after ProjectSpec v1.2 so README starts with self-contained `swanx init`, user guide documents templates/inspect, and active plans avoid stale exact pytest counts.
