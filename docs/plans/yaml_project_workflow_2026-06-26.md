# YAML Project Workflow Plan

## Goal

Completed: add and harden the initial human-editable YAML ProjectSpec workflow
so users can run a SW-XPS project without writing a custom fitting script.
ProjectSpec v1.1 now includes `swanx init`, copied/example data-root options,
project-local default outputs, and `report.md`.

## Scope

- Add optional PyYAML support under the `project` extra.
- Add `swanx init my_project`, including `--copy-example-data` and
  `--data-root`.
- Add `swanx.project.validate_project(...)`, `run_project(...)`, and CLI
  `swanx validate` / `swanx run`.
- Build ProjectSpec v1.1 into existing `swanx.io`, simulation, and fitting objects.
- Support stable layer IDs, tags, repeat expansion, inline parameters, and safe
  arithmetic expressions.
- Implement complete `simulate_only` output folders, project-local default
  outputs, Markdown `report.md`, and method-specific report writers for
  existing result-like objects.

## Non-goals

- No optics, XPS, reflectivity, fitting, or optimizer algorithm changes.
- No Excel, GUI, JSON input, HTML report, Auger, XES, XMCD, or single-crystal
  functionality.
