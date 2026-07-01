# ProjectSpec Identifiability Diagnosis Report

## Goal

Add an optional ProjectSpec report feature that writes a post-fit
`identifiability_analysis/` folder for JAX least-squares runs. The report should
turn the final weighted residual vector and Jacobian into parameter sensitivity,
rank/conditioning, weak SVD modes, correlations, dataset sensitivity, plots, and
a short Markdown summary.

## User-Facing YAML

Use `run:` as the unified section for execution selection, optimizer controls,
and optional run outputs:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 80
    estimate_covariance: true
  outputs:
    plots: true
    identifiability:
      enabled: true
      weak_modes: 5
      active_bound_tol: 0.02
      low_sensitivity_threshold: 0.05
      high_uncertainty_threshold: 0.50
      high_correlation_threshold: 0.90
      high_weak_participation_threshold: 0.50
```

Short form should also work:

```yaml
run:
  mode: "jax_least_squares"
  outputs:
    plots: true
    identifiability: true
```

Backward compatibility:

- Existing `settings.fit_method` remains valid.
- Existing `settings.optimizer` remains valid.
- Existing `report.save_plots` remains valid.
- Existing `report.identifiability` remains valid as an alias.
- If both old and new fields are supplied for the same choice, they must agree;
  conflicting values should fail validation rather than silently choosing one.

If identifiability is omitted or false, no extra diagnosis is written. If
enabled for `simulate_only`, `jax_gradient`, or `bayesian_optimization`, the
runner should skip it with a clear note in `report.md`, because the required
least-squares Jacobian is not available.

## Implementation Outline

1. Move the former benchmark-local identifiability logic into package code,
   likely:
   - `src/swanx/diagnostics/identifiability.py` for computation tables.
   - `src/swanx/diagnostics/identifiability_plots.py` or additions to
     `src/swanx/diagnostics/plots.py` for figures.
   - `src/swanx/project/reporting/identifiability.py` for ProjectSpec file
     writing.
2. Use in-memory ProjectSpec state where possible:
   - parameters from `built.spec.varying_parameters()`;
   - best values from `result.best_parameters`;
   - weighted residuals from `result.final_residuals`;
   - weighted Jacobian from `result.final_jacobian`;
   - optional covariance/correlation from the least-squares diagnostics.
3. Scale each Jacobian column by the declared parameter range before SVD and
   sensitivity norms, matching the current analysis convention.
4. Reuse the existing `fit/residuals.csv` row order only for dataset labels when
   computing `dataset_sensitivity.csv`. This should be documented as
   sensitivity of the current weighted objective, not proof by itself that one
   dataset is physically over- or under-weighted.
5. Add `ProjectSpec.run`, `ProjectSpec.optimizer_settings`, and output helper
   methods while keeping `ProjectSpec.fit_method` as the public mode property.
6. Call the writer from `run_project()` after least-squares optimizer outputs and
   fit residual files are written, then include the generated files or skip note
   in the top-level `report.md`.

## Output Contract

When enabled and available, write:

- `identifiability_analysis/summary.md`
- `identifiability_analysis/parameter_identifiability.csv`
- `identifiability_analysis/singular_values.csv`
- `identifiability_analysis/weak_modes.csv`
- `identifiability_analysis/strong_correlation_pairs.csv`
- `identifiability_analysis/dataset_sensitivity.csv`
- plots when matplotlib is available:
  `scaled_sensitivity.png`, `singular_values.png`,
  `correlation_heatmap.png`, `weak_modes.png`,
  `dataset_sensitivity_heatmap.png`

The summary should explicitly state:

- residual count and varying-parameter count;
- largest/smallest singular values and condition number;
- lowest scaled-sensitivity parameters;
- highest weak-mode participation parameters;
- near-bound parameters;
- strongest correlations;
- weakest SVD parameter combinations;
- a caveat that dataset sensitivity reflects the configured residual scaling and
  weights.

## Validation

- Add unit tests for the pure diagnostics functions:
  rank-deficient Jacobian, zero-sensitivity parameter, weak-mode sorting,
  parameter-range scaling, dataset block sensitivity, and threshold suggestions.
- Add ProjectSpec workflow tests that:
  - `run.mode: "simulate_only"` switches to simulation-only mode;
  - `run.mode` conflicts with legacy `settings.fit_method` fail validation;
  - `run.outputs.plots` maps to the existing plotting switch;
  - `run.outputs.identifiability: true` writes the expected folder for a tiny
    `jax_least_squares` project;
  - disabled/omitted settings write nothing;
  - unsupported methods produce a skip note rather than failing the run.
- Re-run the synthetic C/LNO/STO ProjectSpec benchmark and compare the packaged
  output against the current benchmark-local `identifiability_analysis` folder.
- Run `python -m pytest` after implementation.

## Notes

This feature should not change fitting behavior, physics, optimizer settings, or
objective scaling. It is a post-fit diagnosis of the fitted objective. If
reflectivity dominates `dataset_sensitivity.csv`, the report should frame that
as a weighting/scaling audit signal rather than automatically declaring the
reflectivity scale wrong.

## Implementation Status

Implemented on 2026-07-01.

- Added `run.mode`, `run.optimizer`, and `run.outputs` as the preferred
  ProjectSpec execution controls.
- Preserved legacy `settings.fit_method`, `settings.optimizer`, and
  `report.save_plots` when they do not conflict with `run`.
- Added packaged `swanx.diagnostics.identifiability` analysis utilities.
- Added ProjectSpec `run.outputs.identifiability` reporting for
  `jax_least_squares` outputs.
- Updated generated `swanx init` starters and active ProjectSpec docs.
- Verified with:
  `python -m pytest --basetemp runs\pytest_projectspec_identifiability_full`
  (`257 passed, 1 xfailed, 1 warning`).

## No-Code Fixed-Grid JAX Residual Plan

Goal: make the default ProjectSpec JAX least-squares path build its fixed grid
from YAML stack, parameter, slicing, dataset, and core-level settings. Users
should not need to write a project-local `factory.py` for the common fixed-stack
workflow.

User-facing YAML:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 80
  outputs:
    plots: true
    identifiability: true
```

Implementation steps:

1. Add an internal ProjectSpec JAX residual builder that:
   - expands the YAML stack once, so repeat blocks and material order define the
     fixed topology;
   - evaluates `thickness_A` and `roughness_A` expressions with JAX-compatible
     arithmetic from the trial parameter vector;
   - uses `settings.slicing.mode: fixed_grid` and its reference values to define
     fixed slice counts and fixed array shapes;
   - simulates reflectivity and rocking curves with the existing transfer-matrix
     JAX kernels and the same ProjectSpec RC normalization convention used by
     the maintained fitting path.
2. Update `run_project()` so `run.optimizer.residual: "auto_fixed_grid"` (or an
   omitted residual choice) uses the internal builder. Keep
   `run.optimizer.residual_function_factory: "module:function"` as an advanced
   compatibility hook and reject configs that specify both.
3. Update validation so JAX least-squares no longer requires a manually written
   factory when the auto fixed-grid residual is selected. Fail early with a clear
   message if the auto path is requested without datasets or without fixed-grid
   slicing.
4. Update `swanx init` to stop writing `synthetic_residual_factory.py`; generated
   starter YAML should use `residual: "auto_fixed_grid"`.
5. Update user docs and tests to describe the no-code path first and the factory
   hook as an advanced escape hatch.

Initial scope:

- The auto path preserves the current fixed-shape JAX least-squares behavior for
  YAML stacks whose topology does not change during fitting.
- `normalization: "mean"` and `normalization: "edge_polynomial"` are supported
  for rocking curves. Edge-polynomial is the maintained ProjectSpec default
  after the follow-up normalization sweep.

## Maintained Example Sweep Plan

Follow-up on 2026-07-01:

1. Update maintained YAML examples so execution choices live under `run:`
   instead of legacy `settings.fit_method` or `settings.optimizer`.
2. Keep `run.optimizer.residual: "auto_fixed_grid"` as the visible default for
   ProjectSpec JAX least-squares examples.
3. Remove factory-script references from maintained example documentation unless
   the example is explicitly a custom Python fitting example.
4. Validate all maintained ProjectSpec example YAML files and run the example
   scripts that are expected to execute in this checkout.

## ProjectSpec Reference Expansion Plan

Follow-up on 2026-07-01:

1. Expand `docs/projectspec_reference.md` from a field list into a practical
   reference that explains defaults, current backend limitations, and common
   YAML patterns.
2. Document `run:` as the preferred control surface and keep legacy
   `settings.fit_method`, `settings.optimizer`, and `report.save_plots` only as
   compatibility notes.
3. Clarify rocking-curve normalization choices:
   - `edge_polynomial` with first/last edge fractions for ProjectSpec defaults;
   - `mean` plus `rocking_curve_offpeak_mask` as a supported alternative;
   - auto fixed-grid JAX supports the same ProjectSpec RC normalization choices.
4. Add more detail for optimizer settings, slicing, datasets, core-level
   selectors, output files, and empty placeholder sections such as `report: {}`.

Status: completed. `docs/projectspec_reference.md` now documents the modern
`run:` section, legacy compatibility rules, normalization choices, fixed-grid
slicing, datasets, reports, optimizer details, output files, and common
validation mistakes.
