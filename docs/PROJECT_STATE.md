# Project state

Last updated: 2026-06-21

This is the machine-independent handoff for the SWXPS repository. Read this
file, `docs/TODO.md`, `AGENTS.md`, and the root `README.md` before continuing a
substantial coding session.

## Git state at handoff

- Branch: `main`.
- Last implementation commit before this documentation update: `01f8d6f`
  (`Add workflow benchmark and cache scientific tables`).
- Reorganization baseline: `04da44a`
  (`Reorganize project and checkpoint fitting workflows`).
- The implementation worktree was clean after `01f8d6f`.
- Pull the latest remote commits before starting on another machine. Local
  `runs/` and `archive/` contents are ignored and will not travel through Git.

## Current capabilities

- Validated s-polarized Parratt reflectivity using grazing angles in degrees.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Rough-interface grading through effective sharp slices.
- Material and concentration profiles and normalized SW-XPS rocking curves.
- Experimental preprocessing and joint reflectivity/rocking-curve objectives.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.
- Declarative stack builders backed by local optical-constant and IMFP tables.
- NumPy/JAX parity coverage for maintained backend behavior.

Experimental fits remain provisional until bounds, weights, normalization,
optical constants, IMFPs, chemistry, and optimizer sensitivity are physically
reviewed.

## Recent completed work

The repository was reorganized so maintained tutorials, experimental case
studies, synthetic benchmarks, generated runs, archives, and documentation have
separate roles. Root package code now lives under `src/swxps`.

Optical-constant and IMFP parsers now use bounded in-process caches keyed by
resolved path, modification time, and file size. File changes invalidate cache
entries automatically. Explicit clearing helpers are also available. No
reflectivity, field, roughness, XPS, normalization, or scoring calculation is
cached.

The representative C/[LNO/STO]x8/STO benchmark is
`benchmarks/performance/profile_forward_workflow.py`. On the development
machine, its 61-angle best-of-five run measured:

- cached table loading: 8.28x faster than the cold parse;
- fields and SW-XPS forward calculation: 0.012611 seconds;
- complete fitting objective: 0.023589 seconds.

Treat these as a local baseline, not portable performance guarantees.

## Verification status

The last full command was:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Result: 91 passed, 46 warnings. The warnings are existing NumPy deprecation
warnings from `np.trapz` in `src/swxps/xps.py`; they did not indicate numerical
test failures.

The focused cache suite passed 20 tests across table loading, interpolation,
invalidation, and stack builders. The default workflow benchmark also completed
successfully.

## Repository map

- `src/swxps/`: maintained package implementation.
- `tests/`: regression and parity tests.
- `examples/`: compact tutorial scripts and their selected figures.
- `case_studies/`: Sample 12/13 inputs, maintained runners, and canonical results.
- `benchmarks/`: synthetic studies and focused performance scripts.
- `OPC/`, `IMFP/`: local scientific tables used by stack construction.
- `runs/`: local generated optimizer output, ignored by Git.
- `archive/`: superseded local experiments, ignored by Git.
- `docs/architecture.md`: code/data flow and static-versus-dynamic boundary.
- `docs/plans/`: active or scoped execution plans.
- `docs/history/`: superseded handoffs and chronological records.

`PLANS.md` intentionally remains in the repository root. It is short bootstrap
guidance defining the required format for all substantial execution plans. The
technical reflectivity plan has moved to
`docs/plans/XR_REFLECTIVITY_PLAN.md`.

## Setup on another machine

From a fresh clone or updated checkout:

```powershell
python -m pip install -e .
python -m pytest
python benchmarks/performance/profile_forward_workflow.py
```

Install optional extras only for the work being continued, for example:

```powershell
python -m pip install -e ".[plot,gradient,least-squares]"
```

Check `git status` before editing. Do not assume ignored local fit outputs from
the previous computer are available.

## Current direction

The next priority is not a wholesale source rewrite. First profile a real
maintained case-study objective and review duplicated setup/configuration code.
Then extract one small, tested boundary at a time while keeping examples and
case-study runners operational. Scientific validation of experimental
normalization and fit robustness remains the main project-level priority.
