# Examples Redesign

## Goal

Make `examples/` a user-facing learning path now that ProjectSpec and the main
SWANX workflows are stable. Keep tests focused on regression coverage and keep
benchmarks focused on timing, synthetic fitting, and method comparison.

## Scope

- Reorganize examples around user tasks rather than internal package modules.
- Make ProjectSpec the first and most visible example path.
- State clearly that the synthetic C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3
  benchmark system is the recurring demonstration case used across examples and
  benchmarks.
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
- Starter YAML now lives in `swanx init` and
  `examples/01_quickstart_projectspec/`; the repository-local `templates/`
  folder has been retired.
- `tests/`: regression coverage only, not tutorial material.

## Validation

- Validate all maintained ProjectSpec YAML examples.
- Run focused ProjectSpec tests after path updates.
- Run the full pytest suite before final handoff if time allows.

## 2026-06-29 Correction

The introductory examples were initially simplified too far toward a single
LNO/STO film. They should instead mimic the benchmark case: C cap on a
20-repeat LaNiO3/SrTiO3 (LNO/STO) superlattice on a SrTiO3 substrate, with
reflectivity and La 4d, O 1s, Ti 2p, and C 1s rocking curves. The examples now
share that case through `examples/synthetic_case.py`, and the README/user-guide
wording points users to the same case consistently.

## 2026-06-29 Follow-Up

The generated `swanx init` project still used the older single-film starter.
It now uses the same C/LaNiO3/SrTiO3 superlattice starter as the maintained
examples, and user-facing docs introduce the LNO/STO abbreviations on first
mention.
