# YAML Project Workflow Hardening

## Goal

Make the YAML ProjectSpec workflow easier for beginners while preserving the
validated optics, XPS, reflectivity, fitting, and numerical code paths.

## Scope

- Add `swanx init my_project` to generate `project.yaml`, `run_project.py`, and
  a short project README.
- Make optional ProjectSpec sections default to empty mappings.
- Require explicit core-level emitting-layer selectors.
- Support `vary: false` constants and keep only varying parameters in fitting.
- Keep JAX least-squares as the recommended fitting path and BO as an optional
  baseline.
- Clarify method-specific report outputs, plots, progress messages, and docs.
- ProjectSpec v1.2: make default `swanx init` self-contained from packaged
  tutorial data, add starter templates and `swanx inspect`, support copied
  example data and explicit data roots, keep default outputs under the project
  folder, and write `report.md` for every run.

## Validation

Run focused ProjectSpec tests first, then the full test suite.

- `python -m pytest tests/test_project_workflow.py -q`: run after ProjectSpec changes.
- `python -m pytest -q`: run before handoff for substantial changes.
- `python templates/run_project.py`: wrote a timestamped minimal project run.
- `python -m swanx.cli validate/run templates/project_minimal.yaml`: passed.
- `python -m pip install -e ".[project]"`: refreshed the console script.
- `swanx inspect`, `swanx validate`, `swanx run`, and `swanx init`: smoke-check after ProjectSpec changes.
- Generated `run_project.py` from `swanx init`: passed.
- ProjectSpec v1.2 smoke: default packaged-data `swanx init`,
  `--copy-example-data`, `--data-root`, cross-CWD generated script run,
  `swanx inspect`, project-local default output, and `report.md`: passed.

## Documentation consistency follow-up

- 2026-06-27: Active docs and AGENTS.md were swept for ProjectSpec v1.2
  wording after the workflow hardening commit.
- 2026-06-27: ProjectSpec reports gained visible run progress, compound overview
  plots, incident-angle labels, and least-squares parameter/correlation plots.
