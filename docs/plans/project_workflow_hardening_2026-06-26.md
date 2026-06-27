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
- Clarify method-specific report outputs, plots, and docs.
- ProjectSpec v1.1: make `swanx init` data paths robust, support copied
  example data and explicit data roots, move default outputs under the project
  folder, and write `report.md` for every run.

## Validation

Run focused ProjectSpec tests first, then the full test suite.

- `python -m pytest tests/test_project_workflow.py -q`: 20 passed.
- `python -m pytest -q`: 240 passed, 1 xfailed.
- `python templates/run_project.py`: wrote a timestamped minimal project run.
- `python -m swanx.cli validate/run templates/project_minimal.yaml`: passed.
- `python -m pip install -e ".[project]"`: refreshed the console script.
- `swanx validate`, `swanx run`, and `swanx init`: passed.
- Generated `run_project.py` from `swanx init`: passed.
- ProjectSpec v1.1 smoke: `swanx init --copy-example-data`, cross-CWD
  generated script run, project-local default output, and `report.md`: passed.

## Documentation consistency follow-up

- 2026-06-27: Active docs and AGENTS.md were swept for ProjectSpec v1.1
  wording after the workflow hardening commit.
