# Project state

Last updated: 2026-06-23

This is the machine-independent handoff for the `swanx` repository. Read this
file, `docs/TODO.md`, `AGENTS.md`, and the root `README.md` before continuing a
substantial coding session.

## Namespace migration

- `swanx` means *standing-wave analysis for X-ray spectroscopy* and is now the primary distribution and import namespace.
- Maintained implementation modules live under `src/swanx/`.
- `src/swxps/` is a temporary compatibility shim; existing `swxps.*` imports resolve to the same implementation objects.
- First-stage discovery facades are available at `swanx.stack`, `optics`, `xps`, `fitting`, `diagnostics`, `io`, and `workflows`.
- High-level unified slicing remains the default; `slicing=None` remains the legacy fixed-step selector.
- The README now uses `swanx_logo.png`, explains the name, and starts with the high-level API.
- No physics algorithms were changed.

## Git state at handoff

- Branch: `main`.
- Published base commit entering this handoff: `9069fdd`
  (`Adopt unified slicing and add fit diagnostics`).
- Documentation handoff commit: `97e27b1`.
- Slicing design commits: `cdcf827` and `c9a1d56`.
- Unified-grid implementation and synthetic comparison commits: `3c58e6a` and
  `5db3a1f`.
- This handoff adds the Sample 13 fixed-grid runner and expanded-bound study,
  unified-by-default request semantics, explicit legacy-mode coverage, the JAX
  materialization boundary test, and reusable parameter diagnostics.
- Local `runs/` and `archive/` contents are ignored and do not travel through Git.

## Collaboration and Git preference

Update this file and `docs/TODO.md` after substantial work, but do not stage,
commit, amend, or push unless the user explicitly requests that Git action.
Uncommitted revisions are intentional review state, not an incomplete handoff.

## Current capabilities

- Validated s-polarized Parratt reflectivity using grazing angles in degrees.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Rough-interface grading through effective sharp slices.
- Material and concentration profiles and normalized SW-XPS rocking curves.
- Experimental preprocessing and joint reflectivity/rocking-curve objectives.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.
- Reusable local least-squares covariance, correlation, confidence-interval,
  and singular-value parameter diagnostics.
- Declarative stack builders backed by cached local optical/IMFP tables.
- NumPy/JAX parity coverage for maintained backend behavior.

Experimental fits remain provisional until bounds, weights, normalization,
optical constants, IMFPs, chemistry, and optimizer sensitivity are reviewed.

## Unified slicing implementation

Unified slicing is the default for `ReflectivityRequest` and
`RockingCurveRequest`. Set `slicing=None` explicitly for the legacy step path;
fitting problems continue to propagate their declared `slicing` choice.

- `LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0)` provides adaptive counts.
- `max_slice_thickness` is user configurable in Angstrom.
- `fixed_layer_grid_plan(capacity_layers, policy)` freezes counts from fitting upper bounds.
- One cell-centered grid supplies graded optical cells, field locations,
  concentration/IMFP samples, attenuation, and midpoint RC weights.
- JAX trial thickness changes update widths and values without changing shapes.
- Existing `field_step` and `roughness_step` paths remain available with
  explicit `slicing=None`.
- Generic high-level grid materialization is not JAX-traceable. End-to-end
  gradients use a JAX-native fixed-plan model with topology prepared outside
  the trace.

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

## Sample 13 fixed-grid JAX least squares

The maintained runner
`case_studies/sample_13/jax_least_squares_fixed_grid/fit_sample13_fixed_grid_jax_least_squares.py`
uses a fully traceable 1,092-cell capacity grid for the 86-layer Sample 13
stack. It preserves the promoted all-RC parameterization, preprocessing, and
weights while differentiating reflectivity and all three RCs through JAX.

The first full run is under
`runs/sample_13/jax_least_squares_fixed_grid/full_60nfev/`. It compiled the
residual and Jacobian once each, retained JAX/NumPy parity near `1e-15`, and
converged by `ftol` in 34 function evaluations. The objective decreased from
`0.00843834` to `0.00805305` in 5.1 optimizer seconds. Fourteen of 18 parameters
ended within 1% of a bound, so this is a model/identifiability diagnostic and
must not replace the canonical promoted result.

An expanded-bounds follow-up is under
`runs/sample_13/jax_least_squares_fixed_grid/expanded_bounds_60nfev/`. It
converged in 36 evaluations at objective `0.00646408`, 19.7% below the baseline
fixed-grid fit. Jacobian rank improved from 15 to 16 and condition number from
`3.5e20` to `1.6e18`, but 12 of 18 parameters still reached within 1% of a
bound. Relative to the baseline, weighted reflectivity and C 1s contributions
improved 41.4% and 8.2%, while Ni 3p and La 4d worsened 4.9% and 4.3%. The
broader solution is also diagnostic rather than canonical.

## Recent completed work

The repository was reorganized into maintained package, tutorials, case
studies, benchmarks, local runs, archive, and documentation areas.
Optical-constant and IMFP parsers use bounded metadata-aware caches.

The C/[LNO/STO]x8/STO benchmark is
`benchmarks/performance/profile_forward_workflow.py`. On the original machine,
its 61-angle run measured an 8.28x cached table-load speedup and a
0.023589-second complete fitting objective. These are local baselines only.

## `swanx` diagnostics sanity check

The maintained synthetic C/[LNO/STO]x20/STO fixed-grid JAX/TRF runner now imports `swanx` and uses `swanx.diagnostics` to save parameter uncertainty and correlation plots. A repeat run reproduced the prior optimum (`final_cost=4.422695921494358e-08`, 34 function evaluations) with one residual and one Jacobian compilation. The final Jacobian has rank 6 for 7 parameters and condition number `4.27e18`; substrate roughness is effectively unidentifiable, so its tiny pseudoinverse standard error is not evidence of precise constraint. Generated plots are under `runs/synthetic_c_lno_sto/unified_jax_least_squares/`. The uncertainty plot now uses each parameter's finite bound range as a 0-1 coordinate, scales confidence intervals by the same range, and labels raw lower/upper endpoints; pass `normalization=None` for raw coordinates. The legend is placed above the axes, with larger endpoint text, axis text, markers, and bound bars to avoid overlap and improve readability.

## Stage 2 subpackage migration

- Canonical slicing implementation: `swanx.stack.slicing`.
- Canonical profile implementation: `swanx.stack.profiles`.
- Canonical diagnostics: `swanx.diagnostics.covariance`, `.plots`, and `.reports`.
- Flat `swanx.slicing`, `swanx.profiles`, and `swanx._diagnostics` remain thin shims.
- Legacy `swxps.slicing`, `swxps.profiles`, and `swxps.diagnostics` expose the same canonical objects.
- `swanx.stack` uses lazy exports for simulation/profile/template conveniences to avoid package-initialization cycles.
- Optics, XPS, simulation, and fitting implementation modules were not moved.
- Full verification: 153 passed and 1 expected failure.

## Covariance/correlation validation fix

Least-squares diagnostics now recompute `s^2 (J^T J)^+` from final residuals and the final physical-space Jacobian rather than consuming `result.covariance`. All covariance inputs are validated and symmetrized; materially indefinite, non-finite, or negative-variance inputs raise. Correlation construction enforces symmetry and the `[-1, 1]` range, clipping only tiny floating-point excursions. The optimizer's stored covariance now uses the same `rcond=1e-12` cutoff and is explicitly symmetrized. The synthetic fit reproduced the same optimum; its rank-deficient covariance emitted the expected warning while projecting a tiny negative eigenvalue to zero, and the corrected plot was regenerated.

## Sample 12 diagnostics sanity check

The maintained Sample 12 bounded JAX/TRF runner now imports `swanx` and saves `parameter_uncertainty.png`, `parameter_correlation.png`, and `parameter_correlation.csv` through `swanx.diagnostics`. An isolated run with promotion disabled converged in 22 function evaluations (23.6 optimizer seconds) at NumPy/JAX objective `0.00332310482394`. This is worse than the preserved canonical TRF objective (`0.00321527039509`), so it remains a diagnostics-only run under `runs/sample_12/jax_least_squares/diagnostics_sanity_check/`. The 18x18 correlation matrix has zero asymmetry, exact unit diagonal, and maximum absolute entry one. Strong couplings include terminal STO/LNO roughness (`-0.9992`) and the reflectivity/RC angular offsets (`+0.9982`).

## Verification status

Last full implementation verification:

```powershell
python -m pytest -q
```

Result: 153 passed and 1 expected failure after the Stage 2 subpackage implementation migration. Coverage includes Fresnel, analytic attenuation,
thin-layer convergence, fixed-shape fitting, NumPy/JAX parity, exact legacy
optical grading on an identical grid, default/legacy request semantics, the
JAX differentiation boundary, and the Sample 13 capacity/parity path. The
expected failure records the unsupported Python-float conversion of a traced
thickness during generic high-level grid materialization.

## Repository map

- `src/swanx/`: maintained implementation and public facades.
- `src/swxps/`: temporary compatibility shim for old imports.
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

The fixed-grid Sample 13 migration is numerically successful but exposes severe
bound pressure under both baseline and expanded ranges. The next safe step is
to reduce or reparameterize correlated thickness-transition and roughness
degrees of freedom, review the dataset weighting, and independently calibrate
the common angular shift before another fit. Preserve the legacy path and
canonical result for direct comparison.
