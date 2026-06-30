# Sample 12 Reduced v2: Tied Offset and Period/Fraction Thickness

## Goal

Create a second reduced Sample 12 ProjectSpec experiment that tests two model
simplifications after reduced-v1:

- tie the reflectivity and rocking-curve angle offsets to one fitted
  `angle_offset`;
- replace separate STO/LNO start and delta thickness parameters with
  superlattice period and LNO-fraction parameters.

## Proposed Parameterization

Keep the reduced-v1 fixed roughness/split-cap parameters. Replace:

```text
sto_thickness_start
lno_thickness_start
sto_thickness_delta
lno_thickness_delta
```

with:

```text
sl_period_start
sl_lno_fraction_start
sl_period_delta
sl_lno_fraction_delta
```

Then compute each repeat as:

```text
period = transition_erf(repeat_index0, sl_period_start,
                        sl_period_start + sl_period_delta,
                        thickness_transition_repeat,
                        thickness_transition_width)
lno_fraction = transition_erf(repeat_index0, sl_lno_fraction_start,
                              sl_lno_fraction_start + sl_lno_fraction_delta,
                              thickness_transition_repeat,
                              thickness_transition_width)
STO thickness = period * (1 - lno_fraction)
LNO thickness = period * lno_fraction
```

## Validation Plan

1. Create a new ignored folder under `case_studies/sample_12/`.
2. Validate with `swanx inspect` and `swanx validate`.
3. Run the ProjectSpec fit once.
4. Run the existing identifiability script on the new run.
5. Compare objective, active bounds, scaled condition number, and strong
   correlations against reduced-v1.

## Result

- Added ignored local folder
  `case_studies/sample_12/projectspec_jax_least_squares_reduced_v2_period_fraction/`.
- `inspect`, `validate`, and one full run passed. The run folder is:
  `runs/sample12_projectspec_jax_least_squares_reduced_v2_period_fraction_20260630_154430/`
  under the v2 case-study folder.
- The diagnostic script needed a short explicit output directory because the
  long v2 path exceeded the practical Windows path length for some file writes.
  Diagnostic CSV/PNG outputs were written to:
  `runs/sample12_v2_identifiability_20260630_154430/`.
- v2 final objective is `0.0034975009751623707`, worse than reduced-v1
  (`0.0034093268637466366`) and the 18-parameter ProjectSpec baseline
  (`0.003376508606357828`).
- v2 has no exactly active bounds and improves greatly over the 18-parameter
  baseline condition number, but its scaled-Jacobian condition number is
  `1180.37`, worse than reduced-v1 (`230.594`).
- The strongest v2 correlations are within the new superlattice
  period/fraction/transition parameter block, especially
  `sl_period_start` with `sl_lno_fraction_delta` and `sl_period_delta` with
  `thickness_transition_repeat`.

## Interpretation

Tying the angle offsets is probably not worth it for this sample, which matches
the experimental fact that reflectivity and rocking curves were measured
separately. The period/fraction parameterization is physically cleaner than
separate STO/LNO thicknesses, but allowing both period and LNO fraction to have
independent erf transitions creates a new correlated block. A better next test
would keep separate reflectivity/RC offsets and simplify the superlattice model
to fewer transition degrees of freedom, for example period start + period delta
+ constant LNO fraction.
