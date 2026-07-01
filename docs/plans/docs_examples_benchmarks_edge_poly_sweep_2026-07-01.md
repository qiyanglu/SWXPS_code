# Docs, Examples, And Benchmarks Edge-Polynomial Sweep

## Goal

Sweep active documentation, maintained examples, and benchmark ProjectSpec
surfaces after making edge-polynomial rocking-curve normalization the default.
Remove stale descriptions that still imply the default ProjectSpec workflow
uses off-peak mean normalization or project-local residual factory scripts.

## Scope

Active files to check:

- `README.md`
- `docs/projectspec_reference.md`
- `docs/user_guide.md`
- `docs/architecture.md`
- `docs/PROJECT_STATE.md`
- `docs/TODO.md`
- `examples/`
- `benchmarks/`

Historical files under `docs/history/` and older completed plan files may keep
old context, but active handoff/status docs should describe the current package.

## Implementation Steps

1. Search active docs, examples, and benchmarks for stale ProjectSpec wording:
   factory scripts, legacy `settings.fit_method` examples, `report.save_plots`
   examples, off-peak normalization as the default, and old identifiability
   analyzer references.
2. Expand `docs/projectspec_reference.md` where useful:
   - make edge-polynomial the default RC normalization in examples;
   - clarify per-dataset normalization override behavior;
   - clarify auto fixed-grid JAX requirements and outputs;
   - keep mean/off-peak normalization as a supported alternative.
3. Update maintained example and benchmark docs/scripts so the printed text and
   YAML descriptions match the current defaults.
4. Validate maintained ProjectSpec YAML examples and benchmark YAML files.
5. Run focused docs consistency tests and `git diff --check`; run broader tests
   if code examples change.

## Validation

Focused checks passed:

```bash
python -m pytest tests\test_io_curves.py tests\test_preprocessing.py tests\test_project_workflow.py::test_projectspec_example_yaml_files_validate tests\test_project_workflow.py::test_projectspec_fitting_example_validates --basetemp runs\pytest_docs_examples_benchmarks_sweep_focused
```

Result: `25 passed`.

Example and benchmark checks passed:

```bash
python examples\02_experimental_data\load_and_overlay_curves.py
python examples\03_python_api\build_from_opc_imfp.py
python -c "from pathlib import Path; from swanx.project import validate_project; base=Path('benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares'); [validate_project(base/name) for name in ('project.yaml','project_bo.yaml','project_simulate_only.yaml')]; print('benchmark ProjectSpec YAML validation passed')"
python benchmarks\synthetic_c_lno_sto\fit_reflectivity_rc_bo.py --generate-only --angle-count 41
python benchmarks\synthetic_c_lno_sto\compare_slicing_strategies.py --help
```

Full-suite validation and whitespace checks passed after the final
documentation/status edits:

```bash
python -m pytest --basetemp runs\pytest_docs_examples_benchmarks_sweep_full
git diff --check
```

Result: `259 passed, 1 xfailed, 1 warning`. The warning is the existing
rank-deficient covariance projection warning in diagnostics tests. `git diff
--check` reported only Windows LF-to-CRLF notices and no whitespace errors.
