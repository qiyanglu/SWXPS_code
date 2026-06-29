# Examples Redesign

## Goal

Make `examples/` a user-facing learning path now that ProjectSpec and the main
SWANX workflows are stable. Keep tests focused on regression coverage and keep
benchmarks focused on timing, synthetic fitting, and method comparison.

## Scope

- Reorganize examples around user tasks rather than internal package modules.
- Make ProjectSpec the first and most visible example path.
- State clearly that the LNO/STO synthetic/tutorial system is the recurring
  demonstration case used across examples and benchmarks.
- Keep low-level field, roughness, profile, and XPS plotting scripts available
  as advanced examples, not beginner entry points.
- Update root README, examples README, user guide, project state, TODO, and
  ProjectSpec example tests for the new paths.

## Target Layout

```text
examples/
  README.md
  01_quickstart_projectspec/
  02_experimental_data/
  03_python_api/
  04_fitting/
  advanced/
    fields_and_profiles/
    roughness_visualization/
    xps_rocking_curves/
```

## Boundaries

- `examples/`: runnable teaching material and compact user workflows.
- `benchmarks/`: performance measurements and synthetic fitting comparisons.
- `templates/`: repository-local starter template files retained for now.
- `tests/`: regression coverage only, not tutorial material.

## Validation

- Validate all maintained ProjectSpec YAML examples.
- Run focused ProjectSpec tests after path updates.
- Run the full pytest suite before final handoff if time allows.
