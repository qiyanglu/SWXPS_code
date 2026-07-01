# ProjectSpec Docs And Reliability Pass

## Goal

Refresh user-facing docs and add focused reliability checks for the current
ProjectSpec workflow without changing physics features or public APIs.

## Scope

- Rewrite `README.md` as a concise landing page.
- Remove stale active-doc language that implies default `swanx init` projects
  need project-local residual factory scripts.
- Add a short fitting interpretation section to `report.md` for fitting runs.
- Add focused `auto_fixed_grid` reliability tests for JAX/NumPy parity,
  finite-difference Jacobian agreement, and normalization consistency.
- Keep BO optional and never a default or fallback.

## Validation Plan

Focused validation passed:

```bash
python -m pytest tests\test_project_workflow.py::test_jax_least_squares_auto_residual_accepts_edge_polynomial_normalization tests\test_project_workflow.py::test_auto_fixed_grid_jax_model_matches_numpy_simulation tests\test_project_workflow.py::test_auto_fixed_grid_jacobian_matches_finite_difference tests\test_project_workflow.py::test_edge_polynomial_normalization_parity_across_projectspec_paths tests\test_project_workflow.py::test_identifiability_report_writer_uses_run_outputs_switch tests\test_project_workflow.py::test_readme_and_project_state_docs_are_current --basetemp runs\pytest_projectspec_docs_reliability_focused
```

Result: `6 passed`.

Full validation passed:

```bash
python -m pytest tests\test_project_workflow.py --basetemp runs\pytest_projectspec_docs_reliability_workflow
python -m pytest --basetemp runs\pytest_projectspec_docs_reliability_full
git diff --check
```

Results:

- Project workflow tests: `38 passed`.
- Full suite: `263 passed, 1 xfailed, 1 warning`.
- `git diff --check`: no whitespace errors; Windows LF-to-CRLF notices only.
