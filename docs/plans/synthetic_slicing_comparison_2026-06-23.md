# Synthetic LNO/STO slicing comparison plan

## Status

Completed and validated on 2026-06-23.

## Goal

Compare legacy 1 Angstrom roughness/field slicing with the optional unified
fixed-shape grid on the maintained synthetic C/[LNO/STO]x20/STO benchmark.

## Physics background

Both simulations must use the same photon energy, nominal stack, optical
constants, IMFPs, angles, core levels, roughness profiles, and off-peak
normalization mask. Only discretization changes.

The legacy calculation uses `roughness_step=1.0` and `field_step=1.0`. The new
calculation uses `LayerSlicingPolicy(min_slices=10,
max_slice_thickness=2.0)` and a fixed plan built from the fitting thickness
upper bounds. At the true stack thicknesses, fixed counts stay within the 2
Angstrom maximum while representing the shape used during fitting.

## Files to create or modify

- new `benchmarks/synthetic_c_lno_sto/compare_slicing_strategies.py`;
- generated local figure, CSV, and text summary under
  `runs/synthetic_c_lno_sto/slicing_comparison/`;
- `docs/PROJECT_STATE.md` and `docs/TODO.md` after validation.

## Implementation steps

1. Reuse the benchmark's `TRUE_VALUES`, stack builder, data angles, and core requests.
2. Compute legacy and fixed-plan unified reflectivity and four normalized RCs.
3. Save old/new/difference panels with identical y-scales for old and new columns.
4. Save pointwise data and maximum/RMS difference metrics.
5. Inspect the rendered plot and report whether differences are scientifically visible.

## Tests

- Both paths return finite arrays with identical angle shapes.
- Reflectivity remains non-negative and does not exceed one.
- Every normalized RC remains finite and positive.
- The comparison runner completes without modifying the synthetic fixture data.

## Validation

Report maximum absolute and RMS differences for reflectivity and each RC.
Reflectivity should also be compared in log space because its dynamic range is
large. Differences must be shown rather than assumed negligible.

## Progress log

- 2026-06-23: Created and ran the matched comparison at 161 angles.
- Legacy used 810 effective cells; unified fixed-plan slicing used 450 cells.
- Runtime for reflectivity plus four RCs was 0.118 seconds legacy and 0.054 seconds unified on this machine.
- Reflectivity: maximum absolute difference `6.51e-5`, maximum relative difference `0.488%`, RMS log10 difference `0.00134` decades.
- Maximum RC differences: La 4d `1.79e-4`, O 1s `5.98e-5`, Ti 2p `3.47e-4`, C 1s `4.01e-4`.
- The first run revealed that unified optics used the XPS overlap rule instead of the validated nearest-interface optical grading rule. Corrected it and added an exact identical-grid regression test.
- Full suite after correction: 114 passed, 46 pre-existing warnings.
- Artifacts saved under `runs/synthetic_c_lno_sto/slicing_comparison/`.
