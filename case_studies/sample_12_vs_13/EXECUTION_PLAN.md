# Sample #12 and #13 Best-Fit Comparison Plan

## Goal

Create reproducible, publication-ready head-to-head figures comparing the current
best fitting results for Sample #12 and Sample #13: measured/fitted x-ray
reflectivity and rocking curves, plus near-surface element concentration-depth
maps.

## Physics background

The horizontal coordinate in the reflectivity and rocking-curve panels is the
measured grazing-incidence angle in degrees. Reflectivity is dimensionless and is
shown on a logarithmic scale because it spans several orders of magnitude. Each
rocking curve is a normalized, dimensionless photoemission intensity. The depth
maps show the exported element concentration profiles as fractions between zero
and one versus depth below the surface in Angstrom. The near-surface comparison
is restricted to 0--20 Angstrom.

## Files to create or modify

- `case_studies/sample_12_vs_13/compare_best_results.py`
- `case_studies/sample_12_vs_13/README.md`
- `case_studies/sample_12_vs_13/reflectivity_and_rcs_comparison.png`
- `case_studies/sample_12_vs_13/concentration_depth_maps_comparison.png`
- `case_studies/sample_12_vs_13/lno_aligned_concentration_depth_maps_comparison.png`
- `case_studies/sample_12_vs_13/EXECUTION_PLAN.md`

## Implementation steps

1. Load the exported best-fit curve and concentration CSV files directly from
   each sample's `best_results_so_far` directory.
2. Validate required columns, datasets, finite values, and concentration ranges.
3. Draw two adjacent 2x2 measured/fitted grids--one complete grid for Sample #12
   on the left and one complete grid for Sample #13 on the right.
4. Draw paired vertical concentration-depth maps over the first 20 Angstrom,
   keeping C, La, and Ni in distinct columns and mapping zero concentration to
   the paper background exactly.
5. Save high-resolution PNG outputs and inspect them for clipping and overlap.

## Tests

- Run the script from the repository root and from its own directory.
- Confirm both expected PNG files are written.
- Confirm both sample CSVs contain all four curve datasets.
- Confirm all plotted values are finite and concentrations lie in `[0, 1]`
  within a small numerical tolerance.
- Confirm the curve output contains two side-by-side 2x2 sample grids and the
  maps stop at 20 Angstrom.

## Validation

The experimental points and fitted curves should reproduce the current exported
best-fit results without rerunning optimization. Reflectivity remains positive
and is displayed logarithmically. The maps should show carbon concentrated at
the surface and La/Ni appearing below the carbon-rich cap, with Sample #12 and
Sample #13 using identical visual encodings.

## Progress log

- 2026-06-20: Located and inspected both `best_results_so_far` exports; defined
  the comparison layout and validation checks.
- 2026-06-20: Implemented and visually checked the initial comparison figures.
- 2026-06-20: Revised the curves to paired 2x2 grids and restored separate
  vertical element columns. Verified zero concentration renders white.
- 2026-06-20: Checked the original exports: Sample #12 C = 0.5 at depth zero is
  the rough-interface midpoint; Sample #13 has a source-defined Ni-free top
  LNO-1 layer.
- 2026-06-20: Planned per-measurement curve colors, tightly scaled RC axes, a
  second LNO-aligned map, and smooth reconstruction of overlapping C interfaces.
- 2026-06-20: Implemented per-measurement colors and tight shared RC scales;
  generated and visually checked both surface- and LNO-referenced maps.
- Remaining: none.


