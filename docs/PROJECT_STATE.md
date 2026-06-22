# Project state

Last updated: 2026-06-22

This is the machine-independent handoff for the SWXPS repository. Read this
file, `docs/TODO.md`, `AGENTS.md`, and the root `README.md` before continuing a
substantial coding session.

## Git state at handoff

- Branch: `main`.
- Latest implementation commit: `01f8d6f`
  (`Add workflow benchmark and cache scientific tables`).
- Documentation handoff commit: `97e27b1`.
- Slicing design commits: `cdcf827` and `c9a1d56`.
- Unified-grid implementation is complete in the current worktree and pending its implementation commit.
- Local `runs/` and `archive/` contents are ignored and do not travel through Git.

## Current capabilities

- Validated s-polarized Parratt reflectivity using grazing angles in degrees.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Rough-interface grading through effective sharp slices.
- Material and concentration profiles and normalized SW-XPS rocking curves.
- Experimental preprocessing and joint reflectivity/rocking-curve objectives.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.
- Declarative stack builders backed by cached local optical/IMFP tables.
- NumPy/JAX parity coverage for maintained backend behavior.

Experimental fits remain provisional until bounds, weights, normalization,
optical constants, IMFPs, chemistry, and optimizer sensitivity are reviewed.

## Unified slicing implementation

The optional unified grid is implemented. Users select it with `slicing=` on
`ReflectivityRequest`, `RockingCurveRequest`, or `FittingProblem`.

- `LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0)` provides adaptive counts.
- `max_slice_thickness` is user configurable in Angstrom.
- `fixed_layer_grid_plan(capacity_layers, policy)` freezes counts from fitting upper bounds.
- One cell-centered grid supplies graded optical cells, field locations,
  concentration/IMFP samples, attenuation, and midpoint RC weights.
- JAX trial thickness changes update widths and values without changing shapes.
- Existing `field_step` and `roughness_step` paths remain unchanged when `slicing` is absent.

The small thin-surface case used 20 cells, retained one shape across a 2-6
Angstrom thickness sweep, caused one new JAX compilation, stayed within
`1.31e-5` maximum absolute reflectivity error of a 0.1 Angstrom reference, and
produced finite normalized RCs.
A 160 Angstrom film used 80 film cells (90 total), produced finite reflectivity, and evaluated in about 0.0032 seconds on the development machine.

## Recent completed work

The repository was reorganized into maintained package, tutorials, case
studies, benchmarks, local runs, archive, and documentation areas.
Optical-constant and IMFP parsers use bounded metadata-aware caches.

The C/[LNO/STO]x8/STO benchmark is
`benchmarks/performance/profile_forward_workflow.py`. On the original machine,
its 61-angle run measured an 8.28x cached table-load speedup and a
0.023589-second complete fitting objective. These are local baselines only.

## Verification status

Last full implementation verification:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Result: 113 passed, 46 existing `np.trapz` deprecation warnings. The focused unified-grid suite passed 22 tests, including Fresnel, analytic attenuation, thin-layer convergence, fixed-shape fitting, and NumPy/JAX parity.

## Repository map

- `src/swxps/`: maintained implementation.
- `tests/`: regression and parity tests.
- `examples/`: compact tutorials.
- `case_studies/`: maintained experimental runners and canonical results.
- `benchmarks/`: synthetic and performance benchmarks.
- `OPC/`, `IMFP/`: local scientific tables.
- `runs/`, `archive/`: ignored local outputs and superseded experiments.
- `docs/architecture.md`: code/data flow and performance boundaries.
- `docs/plans/`: active and scoped plans.
- `docs/history/`: superseded handoffs and chronological records.

## Current direction

The unified-grid milestone is implemented and validated. The next safe step is to review the diff and benchmark evidence, then migrate one maintained case-study runner separately. Preserve the legacy path until case-study comparison is complete.
