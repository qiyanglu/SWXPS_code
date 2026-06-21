# Sample #12 / Sample #13 comparison

This folder compares the current exported best fitting results in:

- `case_studies/sample_12/best_results_so_far`
- `case_studies/sample_13/best_results_so_far`

The script reads the best-fit curve, stack, and vertical-concentration CSV files
directly; it does not rerun an optimizer or modify either source folder.

From the repository root, regenerate all figures with:

```powershell
python .\case_studies\sample_12_vs_13\compare_best_results.py
```

Outputs:

- `reflectivity_and_rcs_comparison.png`: two adjacent 2x2 grids containing
  reflectivity, C 1s, Ni 3p, and La 4d--Sample #12 on the left and Sample #13
  on the right. Each measurement has one color shared by both samples; RC axes
  use tight measurement-specific ranges.
- `concentration_depth_maps_comparison.png`: separate vertical C, La, and Ni
  columns over the first 20 Angstrom below the physical surface.
- `lno_aligned_concentration_depth_maps_comparison.png`: La and Ni-only maps
  with each nominal top LNO interface aligned to depth zero.

Experimental values use circular markers and best fits use solid lines. Element
colors are consistent between maps, and zero concentration is rendered white.

At depth zero, both source CSVs report C = 0.5, the midpoint of the rough
vacuum/carbon interface. For display, the script reconstructs carbon as the
product of entering the C layer and not yet entering LNO. This removes the
Sample #12 discontinuity caused by overlapping roughness windows in the source
profile export. The Sample #13 La/Ni onset offset comes from its fitted Ni-free
top LNO-1 layer.
