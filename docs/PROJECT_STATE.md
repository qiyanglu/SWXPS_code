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
- Unified-grid implementation commit: `3c58e6a`.
- The synthetic comparison and optical-grading compatibility fix are complete in the current worktree and pending commit.
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

## Synthetic legacy-versus-unified comparison

The maintained C/[LNO/STO]x20/STO synthetic case was run with identical true
stack, 161 angles, core levels, IMFPs, and off-peak mask. Legacy 1 Angstrom
slicing used 810 effective cells; the fixed unified plan used 450 cells.

- runtime: 0.118 seconds legacy, 0.054 seconds unified;
- reflectivity maximum absolute difference: `6.51e-5`;
- reflectivity maximum relative difference: `0.488%`;
- reflectivity RMS log10 difference: `0.00134` decades;
- largest RC difference: `4.01e-4` (C 1s).

The comparison found and fixed an optical-grading mismatch for overlapping
roughness regions. Unified optical cells now use the exact validated legacy
nearest-interface rule. An identical-1-A-grid regression test protects parity.
Artifacts are under `runs/synthetic_c_lno_sto/slicing_comparison/`.

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

Result: 114 passed, 46 existing `np.trapz` deprecation warnings. Coverage includes Fresnel, analytic attenuation, thin-layer convergence, fixed-shape fitting, NumPy/JAX parity, and exact legacy optical grading on an identical grid.

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

The synthetic validation is complete and shows small differences with lower cell count and runtime. The next safe step is to review these results, then migrate one maintained experimental case-study runner separately. Preserve the legacy path for direct comparison.
