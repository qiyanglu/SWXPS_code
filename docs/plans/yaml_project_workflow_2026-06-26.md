# YAML Project Workflow Plan

## Goal

Add a first human-editable YAML ProjectSpec workflow that lets users run a
SW-XPS project without writing a custom fitting script.

## Scope

- Add optional PyYAML support under the `project` extra.
- Add `swanx.project.validate_project(...)` and `swanx.project.run_project(...)`.
- Build ProjectSpec v1 into existing `swanx.io`, simulation, and fitting objects.
- Support stable layer IDs, tags, repeat expansion, inline parameters, and safe
  arithmetic expressions.
- Implement complete `simulate_only` output folders and method-specific report
  writers for existing result-like objects.

## Non-goals

- No optics, XPS, reflectivity, fitting, or optimizer algorithm changes.
- No Excel, GUI, JSON input, HTML report, Auger, XES, XMCD, or single-crystal
  functionality.
