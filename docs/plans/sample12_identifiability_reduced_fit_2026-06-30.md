# Sample 12 Identifiability Diagnostics and Reduced Fit

## Goal

Use the final Sample 12 ProjectSpec least-squares Jacobian to identify weak or
strongly coupled parameter directions, visualize the diagnosis, and create a
separate reduced-parameter ProjectSpec experiment.

## Plan

1. Extend `case_studies/sample_12/analyze_lsq_identifiability.py` with optional
   matplotlib plots:
   - scaled parameter sensitivity;
   - singular-value spectrum;
   - correlation heatmap;
   - weakest SVD mode coefficients;
   - dataset-by-parameter sensitivity heatmap.
2. Keep the diagnostic script post-processing only. It should read the existing
   ProjectSpec run outputs and not rerun the fit.
3. Add a new ignored case-study folder for a reduced Sample 12 ProjectSpec.
   Start conservatively by fixing the least identifiable roughness/split-cap
   parameters to the current best values while keeping the main thickness,
   transition, and angle-offset parameters free.
4. Use a reduced-folder residual factory that accepts the ProjectSpec varying
   parameter list instead of requiring the original 18-parameter vector.
5. Smoke-test the reduced ProjectSpec with `swanx inspect` and, if feasible,
   run the reduced fit once to compare objective quality and diagnostics against
   the 18-parameter baseline.

## Notes

- This is a case-study experiment, not a package feature yet.
- The first reduction should favor cleaner identifiability over absolute best
  scalar objective.
- If the reduced fit behaves well, the next candidate is a more physical
  reparameterization of superlattice thickness as average period, LNO fraction,
  and gradient terms.

## Result

- Added `case_studies/sample_12/analyze_lsq_identifiability.py` as a
  post-processing script for ProjectSpec JAX least-squares outputs.
- The script now writes CSV diagnostics and PNG plots for scaled sensitivity,
  singular values, correlation, weak SVD modes, and dataset sensitivity.
- The 18-parameter Sample 12 ProjectSpec run has a scaled-Jacobian condition
  number of about `2.47e4`. Its weakest modes are dominated by individual
  roughness parameters, carbon roughness fraction, substrate roughness, and the
  split top-LNO cap thickness.
- Added ignored local folder
  `case_studies/sample_12/projectspec_jax_least_squares_reduced_v1/` with 10
  varying parameters. The weak roughness/split-cap parameters are fixed to the
  current 18-parameter best values.
- `inspect`, `validate`, and one full reduced ProjectSpec run passed. The
  reduced run wrote
  `runs/sample12_projectspec_jax_least_squares_reduced_v1_20260630_152036/`.
- Reduced-v1 final objective is `0.0034093268637466366`, about 1% worse than
  the 18-parameter ProjectSpec objective `0.003376508606357828`, but the scaled
  condition number improves to about `230.6` and the reporter shows no exactly
  active bounds.
- Remaining reduced-v1 correlations suggest the next model cleanup should test
  tied reflectivity/RC angle offsets and a more physical superlattice thickness
  parameterization.
