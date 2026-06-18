# CODEX_HANDOFF

## Project Goal

Build a transparent Python package for simulating multilayer x-ray reflectivity and standing-wave XPS from thin-film stacks. The original core goal is physically inspectable Parratt-recursion reflectivity; the current project has grown into transfer-matrix fields, SW-XPS rocking curves, stack templating, visualization, preprocessing, fitting objectives, and Bayesian optimization workflows.

## Current State

- Package source lives under `src/swxps`.
- Core reflectivity is implemented in `src/swxps/reflectivity.py` using manually supplied complex refractive indices.
- Transfer-matrix reflectivity and electric-field profiles are implemented in `src/swxps/fields.py`.
- Standing-wave XPS helpers are implemented in `src/swxps/xps.py`.
- High-level simulation request/result APIs are implemented in `src/swxps/simulation.py`.
- Layer, optical-constant, IMFP, profile, stack-builder, fitting, BO, diagnostics, preprocessing, and stack-visualization utilities exist.
- Tests currently pass locally: `python -m pytest` collected 70 tests and all passed on 2026-06-18.
- README documents installation, examples, and key files, but some listed synthetic example scripts are currently deleted or moved in the local working tree.

## Recent Changes

- Created this root-level handoff document for transfer to another computer or Codex thread.
- Verified the current test suite with `python -m pytest`: 70 passed.
- Inspected the current git state and recorded the dirty working tree below.
- Existing uncommitted work before this handoff includes updates to `XR_REFLECTIVITY_PLAN.md`, fitting/field/simulation modules, tests, synthetic data outputs, new preprocessing support, new OPC data, and new experimental/sample example directories.

## Next Steps

1. Review the dirty working tree with `git status --short --branch` and `git diff --stat`.
2. Decide whether the deleted files in `examples/synthetic_c_lno_sto/` are intentional migrations or accidental removals.
3. Inspect and finish the current reflectivity/RC fitting workflow around `src/swxps/fields.py`, `src/swxps/simulation.py`, `src/swxps/fitting.py`, `src/swxps/bo.py`, and `examples/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py`.
4. Update `README.md` so example commands match the files that actually exist after the example reorganization.
5. Run the full test suite after any changes.
6. If experimental fitting is continued, validate parameter bounds, weighting, optical constants, IMFP values, and fitted stack schematics against physical expectations before treating results as quantitative.
7. Commit logically separated changes once the file moves/deletions and generated artifacts are understood.

## Important Files

- `AGENTS.md`: repository instructions and physics/testing conventions. Read this first.
- `PLANS.md`: required planning format for substantial changes.
- `XR_REFLECTIVITY_PLAN.md`: current project plan and physics notes; it has uncommitted edits.
- `README.md`: user-facing overview, install commands, examples, and current-scope notes.
- `pyproject.toml`: package metadata, dependencies, optional extras, pytest config.
- `src/swxps/reflectivity.py`: Parratt reflectivity core.
- `src/swxps/fields.py`: transfer-matrix reflectivity and field profiles.
- `src/swxps/xps.py`: SW-XPS depth integration and rocking-curve helpers.
- `src/swxps/simulation.py`: high-level simulation API.
- `src/swxps/fitting.py`: datasets, fit parameters, objectives, and fit history.
- `src/swxps/bo.py`: scikit-optimize Bayesian optimization routines.
- `src/swxps/fit_diagnostics.py`: fitting output and diagnostic plotting.
- `src/swxps/preprocessing.py`: newly untracked preprocessing helpers.
- `tests/`: regression coverage for the core physics and fitting helpers.
- `OPC/` and `IMFP/`: tabulated optical constants and inelastic mean free path data.
- `examples/`: scripts, synthetic data, sample-specific fitting experiments, and generated outputs.

## Run Commands

Setup:

```powershell
python -m pip install -e .
python -m pip install -e ".[plot]"
python -m pip install -e ".[fit]"
```

Development:

```powershell
python -m pytest
python examples/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py
```

Testing:

```powershell
python -m pytest
```

Linting:

```powershell
# No linter is configured in pyproject.toml yet.
```

Building:

```powershell
# No build workflow is documented yet. If the build package is installed:
python -m build
```

Useful git inspection:

```powershell
git status --short --branch
git diff --stat
git diff --name-only
```

## Verification

- Run `python -m pytest` from the repository root. The current local baseline is 70 passing tests.
- Confirm the original reflectivity requirements remain covered:
  - vacuum/substrate reproduces Fresnel reflectivity,
  - identical refractive indices give near-zero reflectivity,
  - periodic multilayers show a Bragg peak near `m lambda = 2 d sin(theta)`,
  - reflectivity does not exceed 1 except for small numerical tolerance.
- For fitting/example changes, regenerate plots and CSV outputs, then visually inspect best-fit curves, convergence plots, surrogate slices, and stack schematics.
- For experimental RC fitting, compare fitted layer thicknesses, roughnesses, scale/background terms, and optical constants with physically reasonable ranges.

## Known Issues

- The working tree is dirty and contains both modified tracked files and many untracked files/directories.
- Several tracked files under `examples/synthetic_c_lno_sto/` are deleted in the local tree, while replacement scripts/outputs appear to have been added elsewhere. This needs review before committing.
- README still references some deleted synthetic example scripts.
- No linting tool is configured.
- Build/package artifact generation is not documented or verified.
- Experimental-data fitting is still provisional; weighting, bounds, optical constants, and physical cross-checks require careful validation.
- Git reports line-ending warnings that LF will be replaced by CRLF on several files when touched.

## Decisions Made

- Keep the core reflectivity calculation independent of optical-constant databases.
- Use numpy for core numerical work; use scipy only if needed.
- Keep incidence angles as grazing angles in degrees, photon energy in eV, wavelength/thickness/roughness in Angstrom.
- Represent layer refractive index as `n = 1 - delta + i beta`.
- Treat the first layer as vacuum and the last layer as a semi-infinite substrate.
- Start from s-polarization; p-polarization is a later extension.
- Use explicit, small functions and tests for physics behavior before expanding features.
- Use optimizer-independent fitting abstractions so the Bayesian optimizer backend can be replaced later.
- Use `scikit-optimize` for the current Bayesian optimization implementation.

## Environment Notes

- Current machine: Windows, PowerShell.
- Current repository path on this machine is under the user's OneDrive folder, ending in `SWXPS_code`. The parent directory contains non-ASCII organization text, so verify the exact path with `Get-Location` on Windows.
- Python observed during verification: Python 3.12.7.
- Pytest observed during verification: pytest 7.4.4.
- Package requires Python `>=3.9`.
- Base dependency: `numpy`.
- Optional plotting dependency: `matplotlib` via `.[plot]`.
- Optional fitting dependency: `scikit-optimize` via `.[fit]`.
- No required API keys, environment variables, local services, or ports are known.
- Data dependencies are local files in `OPC/`, `IMFP/`, `references/`, and example subdirectories.

## Git State

- Current branch: `main`.
- Tracking: `main...origin/main`.
- Latest commit: `b6d3148 Document repository outputs and track example artifacts`.
- Current status at handoff time:
  - Modified tracked files include `XR_REFLECTIVITY_PLAN.md`, `src/swxps/__init__.py`, `src/swxps/bo.py`, `src/swxps/fields.py`, `src/swxps/fit_diagnostics.py`, `src/swxps/fitting.py`, `src/swxps/simulation.py`, `src/swxps/stack_visualization.py`, and several tests.
  - Deleted tracked files include multiple old synthetic C/LNO/STO BO scripts, plots, and CSV outputs under `examples/synthetic_c_lno_sto/`.
  - Untracked files/directories include `CODEX_HANDOFF.md`, `src/swxps/preprocessing.py`, `tests/test_preprocessing.py`, new OPC files, new sample-specific example directories, and new synthetic fitting output directories.
- No PR is known from this local state.

## Codex Instructions

- Read `AGENTS.md`, `PLANS.md`, and `XR_REFLECTIVITY_PLAN.md` before substantial edits.
- For any substantial change, create or update an execution plan following `PLANS.md`.
- Do not revert or discard uncommitted changes unless the user explicitly asks.
- Treat existing uncommitted changes as user work and inspect them before editing nearby files.
- Preserve the physics conventions in `AGENTS.md`.
- Keep the reflectivity core transparent and test-driven.
- Prefer scoped changes and focused tests over broad refactors.
- Before committing, resolve whether the example deletions/migrations are intentional and update README/example references accordingly.
