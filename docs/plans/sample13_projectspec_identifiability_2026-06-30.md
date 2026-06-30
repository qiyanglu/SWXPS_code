# Sample 13 ProjectSpec LS and Identifiability Experiments

## Goal

Repeat the Sample 12 identifiability workflow for Sample 13:

1. Create a CLI-runnable ProjectSpec YAML folder that mirrors the maintained
   `case_studies/sample_13/jax_least_squares_all_rcs/fit_sample13_reflectivity_all_rcs_least_squares.py`.
2. Run it and diagnose the final scaled Jacobian.
3. Suggest parameter reductions.
4. Try one reduced-parameter ProjectSpec experiment and compare objective,
   active bounds, scaled condition number, and correlations.

## Baseline Requirements

- Stack: `vacuum / C / LNO-1 / LNO-2 / LNO-bottom / [STO/LNO]x40 / STO`.
- Carbon roughness: `1 + carbon_roughness_fraction * (min(8, carbon_thickness) - 1)`.
- LNO-1 roughness fixed at `0 A`.
- LNO-2 roughness: `lno2_roughness_fraction * top_lno_layer1_thickness`.
- Ni 3p emits only from LNO-2.
- La 4d emits from LNO-1 and LNO-2.
- Reflectivity and RCs keep separate angle offsets.
- Reflectivity window: same as RC window, using 46 points from 12.45 to 14.70 deg.
- Raw RCs are normalized by 10% per-edge quadratic background.
- Reflectivity weight from the maintained runner: `0.061684671063760285`.
- Bounds include the all-RC TRF overrides for STO/LNO starts and angle offsets.

## Validation Plan

- `swanx inspect` and `swanx validate` on the ProjectSpec YAML.
- Run the ProjectSpec with the project-local residual factory.
- Run the existing Sample 12 identifiability script on the Sample 13 run output.
- Use those diagnostics to select reduced parameters.

## Completed Baseline

- Created local ignored ProjectSpec folder:
  `case_studies/sample_13/projectspec_jax_least_squares_all_rcs/`.
- The ProjectSpec mirrors the maintained all-RC TRF JAX least-squares runner with
  18 varying parameters, the cap3 stack, Ni 3p emitted only from LNO-2, La 4d
  emitted from LNO-1 and LNO-2, separate reflectivity and rocking-curve angle
  offsets, and the maintained reflectivity weight.
- The ProjectSpec uses pre-normalized rocking-curve CSV data and
  `settings.normalization: mean` so the data side follows the maintained
  per-edge quadratic background subtraction while simulated rocking curves keep
  the same mean normalization as the original Python runner.
- `swanx inspect` and `swanx validate` passed.
- Corrected 1:1 run:
  `case_studies/sample_13/projectspec_jax_least_squares_all_rcs/runs/sample13_projectspec_jax_least_squares_all_rcs_20260630_160057/`.
- Baseline final objective: `0.008341284463588628`.
- Baseline fit contributions:
  reflectivity `0.00412306279703268`, C 1s `0.0013620406681872663`,
  Ni 3p `0.002336068038705104`, La 4d `0.0005201129596635781`.

## Baseline Diagnosis

Diagnostic outputs:
`runs/sample13_baseline_identifiability_20260630_160057/`.

- Residuals / varying parameters: `223 / 18`.
- Scaled-Jacobian condition number: `5.77824e+18`.
- The exact zero-sensitivity direction is `lno2_roughness_fraction`.
- Weak/high-uncertainty directions are dominated by roughness endpoints,
  substrate roughness, carbon roughness fraction, and transition shape.
- Major bound-hit parameters include carbon roughness fraction, the thickness
  deltas, transition repeat/width, first roughness endpoints, and both angle
  offsets.
- Strongest correlations are
  `sto_thickness_start` versus `lno_thickness_start` (`-0.9970`),
  `reflectivity_angle_offset` versus `rc_angle_offset` (`0.9968`),
  `sto_roughness_last` versus `lno_roughness_last` (`-0.9928`),
  `sto_roughness_first` versus `lno_roughness_first` (`-0.9923`), and
  `sto_thickness_delta` versus `lno_thickness_delta` (`-0.9211`).

## Reduced V1 Experiment

- Created local ignored reduced ProjectSpec folder:
  `case_studies/sample_13/projectspec_jax_least_squares_reduced_v1/`.
- Fixed to the corrected baseline best values:
  `carbon_roughness_fraction`, `top_lno_layer1_thickness`,
  `lno2_roughness_fraction`, `sto_thickness_delta`, `lno_thickness_delta`,
  `thickness_transition_repeat`, `thickness_transition_width`,
  `sto_roughness_first`, `sto_roughness_last`, `lno_roughness_first`,
  `lno_roughness_last`, and `substrate_roughness`.
- Left six varying parameters:
  `carbon_thickness`, `top_lno_total_thickness`, `sto_thickness_start`,
  `lno_thickness_start`, `reflectivity_angle_offset`, and `rc_angle_offset`.
- `swanx inspect`, `swanx validate`, a full ProjectSpec run, and the
  identifiability diagnosis passed.
- Reduced run:
  `case_studies/sample_13/projectspec_jax_least_squares_reduced_v1/runs/sample13_projectspec_jax_least_squares_reduced_v1_20260630_160620/`.
- Reduced diagnosis:
  `runs/sample13_reduced_v1_identifiability_20260630_160620/`.
- Reduced final objective: `0.007912524625745022`, slightly lower than the
  18-parameter baseline.
- Reduced fit contributions:
  reflectivity `0.0037394017830585328`, C 1s `0.0013256468706988511`,
  Ni 3p `0.002351406738783725`, La 4d `0.0004960692332039131`.
- Reduced scaled-Jacobian condition number: `2349.35`.

## Interpretation And Next Parameter-Reduction Ideas

- Reduced v1 removes the singular roughness/transition directions without
  hurting the scalar objective, so those fixed parameters are good first
  candidates to keep out of routine fits.
- The remaining fit still pushes `carbon_thickness` to its lower bound and both
  angle offsets to the upper bound. This should be treated as a calibration or
  model-bound issue before adding roughness freedom back.
- `sto_thickness_start` and `lno_thickness_start` remain strongly anti-
  correlated. A useful next experiment is a physical superlattice
  reparameterization: fit total period and LNO fraction, or fit one period plus
  one contrast/fraction parameter, rather than independent STO and LNO starts.
- Because reflectivity and rocking curves may not share the same experimental
  angular calibration, keep separate offsets by default for Sample 13 unless
  instrument metadata justify tying them.
- If the carbon lower-bound result is not physically plausible, try a lower
  carbon-thickness prior from sample handling/XPS attenuation, or tighten the
  carbon bound instead of allowing it to act as a compensating amplitude knob.
