# Sample 12 YAML JAX Least-Squares Wrapper

## Goal

Create a new `case_studies/sample_12` folder with a YAML description and runner
that mirror the maintained Sample 12 cap3 bounded TRF JAX least-squares script.

## Scope

- Read the existing `jax_least_squares_fit/fit_sample12_joint_cap3_least_squares.py`.
- Preserve the active stack model:
  `vacuum / C / LNO-1 / LNO-2 / LNO-bottom / [STO/LNO]x40 / STO substrate`.
- Preserve the active fitting parameters, bounds, starting source, data
  preprocessing, dataset weights, and optimizer defaults.
- Keep the new runner conservative: default to setup validation and require an
  explicit flag before running the full optimizer or promoting results.

## Original Note

The Sample 12 superlattice uses erf-based repeat-dependent thickness grading and
linear roughness grading. ProjectSpec now has matching safe scalar helpers
(`transition_erf` and `linear_map`), but the Sample 12 YAML remains a case-study
control file rather than a plain `swanx run` ProjectSpec because it also needs
custom preprocessing and a fixed-shape residual runner. The runner verifies the
YAML against the maintained Python definitions and then uses the same validated
fitting path.

This note describes the first local YAML-wrapper pass. It was superseded later
the same day by the real ProjectSpec correction below, which created a
CLI-runnable `project.yaml` plus project-local residual factory.

## 2026-06-30 Follow-Up

The YAML was rewritten from a compact mirror into a one-to-one audit map of
`case_studies/sample_12/jax_least_squares_fit/fit_sample12_joint_cap3_least_squares.py`.
It now records the imported helper modules, data loaders, angle grids,
background subtraction constants, problem factories, source/default output
paths, start-value fallback, reference-weight source, interior-vector clipping,
stack formulas, core-level emitting indices, optimizer settings, output
artifacts, and promotion behavior.

The local runner's setup check was tightened to validate these YAML values
against the maintained Python source. One intentional runtime safety difference
remains: the original LS script promotes an improved run unless
`--skip-promotion` is supplied, while the YAML wrapper only promotes when
`--promote` is explicitly passed.

## 2026-06-30 ProjectSpec Correction

The user clarified that the more useful target is a real CLI-runnable
ProjectSpec, not only a case-study control YAML. A new local ignored folder was
added at:

```text
case_studies/sample_12/projectspec_jax_least_squares/
```

It contains a standard ProjectSpec `project.yaml`, generated CSV inputs,
`sample12_residual_factory.py`, `run_project.py`, and a README. The ProjectSpec
uses safe expression functions for the graded [STO/LNO]x40 stack, points at a
project-local residual factory, and can be run with `swanx run`.

To support this faithfully, ProjectSpec gained three narrow capabilities:

- separate reflectivity and rocking-curve angle-offset parameter names;
- configurable edge-polynomial normalization edge fraction and order;
- optional per-core-level `vacuum_imfp_from_material` to reproduce the legacy
  Sample 12 vacuum-IMFP convention.

## Validation

- `python case_studies/sample_12/yaml_jax_least_squares_fit/run_project.py --setup-only`
  passed. The YAML matched the maintained Python parameter definitions and
  model constants; the setup reported 18 parameters, 227 residuals, and matching
  JAX/NumPy initial objectives.
- `python case_studies/sample_12/yaml_jax_least_squares_fit/run_project.py`
  completed without promotion. The bounded TRF optimizer succeeded with
  `xtol`, 23 function evaluations, and matching JAX/NumPy best objective
  `0.00332423453945`.
- Follow-up setup validation after the one-to-one YAML rewrite passed with the
  expanded source checks; the setup reported 18 parameters, 227 residuals,
  reflectivity weight `0.0503187`, and matching JAX/NumPy initial objectives.
- `swanx inspect case_studies/sample_12/projectspec_jax_least_squares/project.yaml`
  passed.
- `swanx validate case_studies/sample_12/projectspec_jax_least_squares/project.yaml`
  passed.
- The ProjectSpec residual-factory smoke matched the legacy initial objective:
  227 residuals, residual objective `0.00342457446565`, and NumPy objective
  `0.00342457445517`.
- `swanx run case_studies/sample_12/projectspec_jax_least_squares/project.yaml`
  completed and wrote a normal ProjectSpec report folder with final objective
  `0.003376508606357828`.
- Focused ProjectSpec tests passed with `30 passed`.
- Full validation passed with `253 passed, 1 xfailed, 1 warning`.
