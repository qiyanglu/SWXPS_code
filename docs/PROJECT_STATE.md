# Project state

Last updated: 2026-06-24

This is the machine-independent handoff for the `swanx` repository. Read this
file, `docs/TODO.md`, and the root `README.md` before continuing a
substantial coding session.

## Namespace migration

- `swanx` means *standing-wave analysis for X-ray spectroscopy* and is now the primary distribution and import namespace.
- Maintained implementation modules live under `src/swanx/`.
- `src/swxps/` is a temporary compatibility shim; existing `swxps.*` imports resolve to the same implementation objects.
- Stack, optics, XPS, fitting, diagnostics, I/O, and workflow subpackages are
  internal implementation namespaces, not competing user entry points.
- The frozen user entry pattern is `import swanx as sx`.
- High-level unified slicing remains the default; `slicing=None` remains the legacy fixed-step selector.
- The README now uses `swanx_logo.png`, explains the name, and starts with the high-level API.
- No physics algorithms were changed.

## Git state at handoff

- Branch: `main`.
- Published base commit entering this handoff: `68c26a1`
  (`Remove local-only files from repository`).
- This handoff freezes the ten-name top-level `swanx` API and consolidates
  README usage around one `import swanx as sx` workflow.
- JAX automatic differentiation is documented as the primary optimizer;
  Bayesian optimization is explicitly a baseline comparison.
- Full API-freeze verification: 173 passed and 1 expected failure.
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

## Public repository cleanup decision

- `case_studies/` remains tracked as explicitly requested; no history rewrite was performed.
- `runs/` and `archive/` are local-only and fully ignored; no placeholder files
  keep those directories present on GitHub.
- Root `AGENTS.md` and `PLANS.md` are local-only and ignored.
- The former top-level `scripts/` demonstrations moved to `examples/fitting/` and now import `swanx`.
- Both examples recover the synthetic 24 Angstrom film thickness from 35 Angstrom starts.

## Stage 3 optics migration

- Canonical Parratt implementation: `swanx.optics.parratt`.
- Canonical transfer-matrix/field implementation: `swanx.optics.fields`.
- Canonical unified-grid optics implementation: `swanx.optics.unified_grid`.
- Flat `swanx.reflectivity`, `swanx.fields`, and `swanx.unified_grid` remain thin shims.
- Legacy `swxps.reflectivity`, `swxps.fields`, and `swxps.unified_grid` expose the same canonical objects.
- Existing high-level unified simulation functions are lazily re-exported from the canonical unified-grid module.
- XPS, simulation, fitting, and workflow implementation modules were not moved.
- Full verification: 158 passed and 1 expected failure.

## Stage 4 XPS migration

- Canonical attenuation implementation: `swanx.xps.attenuation`.
- Canonical continuous intensity/property implementation: `swanx.xps.intensity`.
- Canonical rocking-curve implementation: `swanx.xps.rocking_curve`.
- Canonical cell-centered grid XPS implementation: `swanx.xps.grid`.
- `swanx._xps`, the former optics unified-grid XPS exports, flat unified-grid
  imports, and legacy `swxps.*` paths expose the same canonical objects.
- `swanx.xps` lazily exposes existing high-level simulation request/result
  APIs so internal canonical imports do not create package cycles.
- Electron attenuation, rough-interface grading, integration, and
  normalization algorithms were moved verbatim; no physics behavior changed.
- Simulation, fitting, and workflow implementations were not moved.
- Full verification: 163 passed and 1 expected failure.

## Stage 5 stack-model and simulation-workflow migration

- Canonical stack models: `swanx.stack.model`.
- Canonical high-level request/result classes and forward entry points:
  `swanx.workflows.simulate`.
- `swanx.simulation` is a thin identity-preserving compatibility shim;
  `swxps.simulation` continues to alias it.
- Preferred `swanx.stack`, `swanx.workflows`, and top-level `swanx` exports all
  resolve to the canonical classes and functions.
- Internal fitting, unified/JAX simulation, profile, builder, visualization,
  and result-export imports now use the canonical model/workflow locations.
- Fitting and diagnostics conveniences in `swanx.workflows` are lazy, avoiding
  the package-initialization cycle exposed by the new canonical imports.
- The implementation bodies were moved without changing numerical algorithms.
- Full verification: 167 passed and 1 expected failure.

## Stage 6 slim simulation compatibility layer

- `swanx.simulation` defines no classes or functions; it only re-exports the
  canonical stack-model and workflow APIs required for compatibility.
- `_values_by_material` and `_apply_emitting_layer_filter` now live in
  `swanx.xps.utils` and are imported by the NumPy, JAX, and unified workflows.
- Private workflow helpers are no longer exposed through `swanx.simulation`.
- Structural tests enforce the thin shim, and exact array-equality tests cover
  reflectivity and SW-XPS results through canonical, `swanx.simulation`, and
  `swxps.simulation` paths.
- Helper and workflow bodies were not numerically changed.
- Full verification: 171 passed and 1 expected failure.

## Final API freeze and user experience

- `swanx.__all__` contains exactly the ten approved stack, request,
  simulation, and diagnostics names.
- README presents only `import swanx as sx` as the official entry pattern and
  no longer recommends stack, optics, XPS, workflow, or simulation submodules.
- JAX autodiff is the recommended primary optimization method; Bayesian
  optimization is retained as a slower global baseline.
- The previous broad surface remains available only through the temporary
  `swxps` compatibility package, backed by `swanx._legacy_api`.
- Maintained advanced fitting scripts now import non-public implementation
  types/functions from their canonical modules.
- No physics or numerical implementation changed.
- Full verification: 173 passed and 1 expected failure.

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

A post-Stage-4 `swanx` sanity rerun is under
`runs/sample_12/jax_least_squares/stage4_swanx_sanity_20260624/`. Promotion
was disabled. It converged by `xtol` in 23 function evaluations (17.54
optimizer seconds), with matching JAX/NumPy objective `0.00338515813538`.
The generated 18x18 correlation matrix is finite, exactly symmetric, has an
exact unit diagonal, and satisfies `max(abs(correlation)) = 1`. The strongest
off-diagonal couplings are the two angle offsets (`+0.9981`) and terminal
STO/LNO roughness (`-0.9957`). Several roughness confidence intervals remain
much wider than their bounds, so this run confirms the diagnostics pipeline
but not parameter identifiability or an improved physical fit.

## Verification status

Last full implementation verification:

```powershell
python -m pytest -q
```

Result: 173 passed and 1 expected failure after the final API freeze. Coverage includes exact frozen-surface enforcement, the official `import swanx as sx` smoke workflow, thin-shim structure, exact canonical/compatibility reflectivity and SW-XPS parity, canonical stack/workflow location and compatibility identity, workflow reflectivity smoke coverage, canonical XPS location/identity, Fresnel, analytic attenuation,
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
