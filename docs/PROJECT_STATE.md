# Project state

Last updated: 2026-06-22

This is the machine-independent handoff for the SWXPS repository. Read this
file, `docs/TODO.md`, `AGENTS.md`, and the root `README.md` before continuing a
substantial coding session.

## Git state at handoff

- Branch: `main`.
- Latest completed handoff commit before the current planning session: `97e27b1`
  (`Add cross-machine project handoff docs`).
- Latest implementation commit: `01f8d6f`
  (`Add workflow benchmark and cache scientific tables`).
- The branch was synchronized with `origin/main` before this plan-only session.
- Local `runs/` and `archive/` contents are ignored and do not travel through Git.

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

## Current slicing limitation

The high-level defaults use 1 Angstrom step-based discretization. Roughness
effective-layer counts and field depth-grid counts are each calculated using a
thickness-dependent `ceil` operation. This has two consequences:

- a several-Angstrom surface or interface layer may receive too few samples for
  the desired accuracy; and
- fitted thickness changes can change JAX input shapes and trigger compilation.

A design-only implementation plan now exists at
`docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md`. No source code has been
changed. The recommended design uses at least ten slices per finite nominal
layer and treats 2 Angstrom as the maximum slice thickness. For fitting/JAX,
slice counts are computed once from capacity thicknesses, preferably parameter
upper bounds, and then held fixed while slice widths vary continuously.

## Recent completed work

The repository was reorganized so maintained tutorials, experimental case
studies, synthetic benchmarks, generated runs, archives, and documentation have
separate roles. Root package code now lives under `src/swxps`.

Optical-constant and IMFP parsers use bounded in-process caches keyed by
resolved path, modification time, and file size. File changes invalidate cache
entries automatically. Explicit clearing helpers are available. No
reflectivity, field, roughness, XPS, normalization, or scoring calculation is
cached.

The representative C/[LNO/STO]x8/STO benchmark is
`benchmarks/performance/profile_forward_workflow.py`. On the original
development machine, its 61-angle best-of-five run measured an 8.28x cached
table-load speedup and a 0.023589-second complete fitting objective. Treat these
as local baselines, not portable performance guarantees.

## Verification status

The last full implementation verification was:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Result: 91 passed, 46 warnings. The warnings are existing NumPy deprecation
warnings from `np.trapz` in `src/swxps/xps.py`; they did not indicate numerical
test failures. This 2026-06-22 session changes documentation only.

## Repository map

- `src/swxps/`: maintained package implementation.
- `tests/`: regression and parity tests.
- `examples/`: compact tutorial scripts and their selected figures.
- `case_studies/`: Sample 12/13 inputs, maintained runners, and canonical results.
- `benchmarks/`: synthetic studies and focused performance scripts.
- `OPC/`, `IMFP/`: local scientific tables used by stack construction.
- `runs/`: local generated optimizer output, ignored by Git.
- `archive/`: superseded local experiments, ignored by Git.
- `docs/architecture.md`: code/data flow and performance boundaries.
- `docs/plans/`: active or scoped execution plans.
- `docs/history/`: superseded handoffs and chronological records.

`PLANS.md` intentionally remains in the repository root as the planning-format
bootstrap. The active reflectivity plan is
`docs/plans/XR_REFLECTIVITY_PLAN.md`.

## Setup on another machine

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
another computer are available.

## Current direction

Resolve the five design decisions in the adaptive/fixed-shape slicing plan
before changing package code. Implement the pure count planner and tests first;
then integrate it behind an opt-in path while preserving every legacy call.
Scientific validation of experimental normalization and fit robustness remains
the main project-level priority after this numerical infrastructure milestone.
