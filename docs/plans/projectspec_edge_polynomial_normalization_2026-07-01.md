# ProjectSpec Edge-Polynomial RC Normalization

## Goal

Make first/last-edge polynomial rocking-curve normalization the default
ProjectSpec normalization strategy across experimental data, simulation outputs,
Bayesian optimization/generic fitting, and no-code fixed-grid JAX
least-squares.

This should remove the current surprise where ProjectSpec auto-JAX fitting
requires `settings.normalization: "mean"` while other paths already support
`settings.normalization: "edge_polynomial"`.

## User-Facing Default

Use this as the visible default in generated starters, maintained examples, and
benchmark ProjectSpec files:

```yaml
settings:
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
```

Meaning:

- use the first 10 percent and last 10 percent of each rocking curve;
- fit a second-order polynomial background on those edge points;
- normalize each rocking curve by that fitted background;
- apply the same rule to experimental RC data and simulated/fitted RCs.

Reflectivity remains scored as reflectivity, usually in log space for fitting.
The edge-polynomial normalization applies to SW-XPS rocking curves only.

## Implementation Steps

1. Add differentiable edge-polynomial normalization to
   `src/swanx/project/jax_fixed_grid.py`:
   - precompute the edge mask, Vandermonde design matrix, and pseudoinverse
     from the fixed angle grid;
   - inside JAX, compute polynomial coefficients from the raw simulated
     rocking curve edge values;
   - divide the full raw curve by the fitted background.
2. Remove the auto fixed-grid JAX validation error that rejects
   `settings.normalization: "edge_polynomial"`.
3. Keep `settings.normalization: "mean"` and `rocking_curve_offpeak_mask`
   available for backward compatibility and specialized workflows.
4. Update `swanx init`, maintained example YAML files, benchmark ProjectSpec
   YAML files, and active docs so the default RC normalization is
   `edge_polynomial`.
5. Add focused tests that:
   - auto fixed-grid JAX residuals accept and evaluate edge-polynomial
     normalization;
   - generated starters contain the new default;
   - docs stay current.

## Validation

- Run focused ProjectSpec workflow tests after implementation.
- Run benchmark/example YAML validation for the changed YAML files.
- Run `git diff --check`.

## Notes

This is intended to change only the default RC normalization convention and to
make the no-code JAX path match the rest of ProjectSpec. It should not change
reflectivity normalization or the physical transfer-matrix/standing-wave
kernels.

## Implementation Status

Implemented on 2026-07-01.

- Added differentiable edge-polynomial rocking-curve normalization to the
  no-code fixed-grid ProjectSpec JAX residual.
- Removed the auto-JAX limitation to mean normalization.
- Updated generated `swanx init` ProjectSpec YAML, maintained examples, and
  benchmark YAML files so `settings.normalization: "edge_polynomial"` with
  first/last 10 percent and polynomial order 2 is the visible default.
- Kept mean normalization and `rocking_curve_offpeak_mask` available as a
  backward-compatible alternative.
- Added a focused ProjectSpec test that evaluates an auto fixed-grid JAX
  residual with edge-polynomial normalization.

Focused validation passed:

```bash
python -m pytest tests\test_project_workflow.py::test_jax_least_squares_auto_residual_accepts_edge_polynomial_normalization tests\test_project_workflow.py::test_swanx_init_generated_project_validates_and_runs_from_different_cwd tests\test_project_workflow.py::test_swanx_init_copy_example_data_and_data_root tests\test_project_workflow.py::test_projectspec_example_yaml_files_validate tests\test_project_workflow.py::test_projectspec_fitting_example_validates --basetemp runs\pytest_edge_poly_focused
```

Result: `5 passed`.

Full validation passed:

```bash
python -m pytest tests\test_project_workflow.py --basetemp runs\pytest_edge_poly_project_workflow
python -c "from pathlib import Path; from swanx.project import validate_project; base=Path('benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares'); [validate_project(base/name) for name in ('project.yaml','project_bo.yaml','project_simulate_only.yaml')]; print('benchmark ProjectSpec YAML validation passed')"
python -m pytest --basetemp runs\pytest_edge_poly_full
```

Results: ProjectSpec workflow tests `34 passed`; benchmark ProjectSpec YAML
validation passed; full suite `258 passed, 1 xfailed, 1 warning`.
