# ProjectSpec Reliability And UX Pass - 2026-07-01

## Scope

Keep this pass focused on trust, diagnostics, and documentation for the current
ProjectSpec workflow. Do not add physics features, GUI/report frontends, new
spectroscopy modalities, online databases, JSON/Excel inputs, or BO fallback
behavior. JAX least-squares remains the recommended differentiable fixed-grid
fitting path.

## Plan

1. Preserve and extend the existing `auto_fixed_grid` tests for JAX/NumPy model
   parity, finite-difference Jacobian agreement, and edge-polynomial
   normalization parity across data loading, simulation, generic fitting, and
   auto fixed-grid residual evaluation.
2. Add an inspect-time Doctor section that reports referenced material and
   dataset file status, optional dependency availability, plotting
   consequences, least-squares/BO dependency readiness, and auto-fixed-grid
   configuration readiness without running simulation or fitting.
3. Add clearer starter aliases while keeping old names working:
   `fit`/`fit-demo`/`minimal` for the fitting starter and
   `simulate`/`multilayer` for the simulation-only starter.
4. Extend fitting `report.md` interpretation with a short recommended-next-checks
   block that points users to bounds, identifiability summaries, correlations,
   and dataset sensitivity as a weighting/scaling audit signal.
5. Refresh README, user guide, ProjectSpec reference, TODO, and PROJECT_STATE so
   active docs are concise, current, and free of default-init factory-script
   wording or pinned exact pytest pass counts.
6. Run focused ProjectSpec tests, the full suite, and `git diff --check`.

## Validation Log

- `python -m pytest tests\test_project_workflow.py --basetemp runs\pytest_projectspec_reliability_ux_workflow`
  passed.
- `python -m pytest --basetemp runs\pytest_projectspec_reliability_ux_full`
  passed with the existing diagnostics covariance projection warning.
- `git diff --check` reported only Windows LF-to-CRLF notices and no whitespace
  errors.
