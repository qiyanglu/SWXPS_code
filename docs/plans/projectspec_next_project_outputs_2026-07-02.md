# ProjectSpec Next-Project Outputs - 2026-07-02

## Scope

Add an opt-in ProjectSpec output feature that writes follow-up YAML files after
a fitting run. This is a reporting/workflow feature only: no physics changes,
no optimizer changes, no BO fallback, and no mutation of the user's original
`project.yaml`.

## Proposed YAML

```yaml
run:
  outputs:
    identifiability: true
    next_project:
      best_start: true
      reduced: true
      low_sensitivity_threshold: 0.02
```

Boolean `next_project: true` enables both generated files with default options.

## Outputs

Write a `next_project/` folder inside the run folder:

- `project_best_start.yaml`: same project, but varied parameters use fitted
  best values as the new `initial` values.
- `project_reduced.yaml`: starts from the best-start YAML and fixes parameters
  whose identifiability `relative_sensitivity` is at or below the configured
  threshold.
- `reduction_notes.md`: explains what was written, which parameters were fixed,
  and why reduction was skipped if diagnostics were unavailable.

Generated YAML files should be runnable from their own output location. Relative
material and dataset paths should be rewritten relative to `next_project/`, and
`project.output_dir` should be removed so follow-up runs write fresh local
folders.

## Implementation Steps

1. Extend ProjectSpec `run.outputs` handling with `next_project` options.
2. Add a reporting writer that reads the original YAML, updates parameter
   blocks, rewrites file paths, disables recursive `next_project` output in the
   generated YAML, and writes notes.
3. Call the writer after fit, optimizer, and identifiability outputs have been
   written.
4. Document the new option in the ProjectSpec reference, user guide, TODO, and
   project state.
5. Add tests for best-start YAML generation, reduced YAML generation from
   identifiability CSVs, generated YAML validation, and docs freshness.
6. Expose the feature in the maintained ProjectSpec fitting example so users
   can see the follow-up workflow without writing custom YAML from scratch.

## Validation Log

- `python -m pytest tests\test_project_workflow.py::test_run_section_controls_mode_optimizer_and_outputs tests\test_project_workflow.py::test_next_project_outputs_write_best_start_and_reduced_yaml tests\test_project_workflow.py::test_readme_and_project_state_docs_are_current --basetemp runs\pytest_next_project_focused`
  passed.
- `python -m pytest tests\test_project_workflow.py --basetemp runs\pytest_next_project_workflow`
  passed.
- `python -m pytest --basetemp runs\pytest_next_project_full` passed with the
  existing diagnostics covariance projection warning.
