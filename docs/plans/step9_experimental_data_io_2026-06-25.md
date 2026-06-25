# Step 9 experimental data IO and workflow cleanup

## Goal

Complete the practical SWANX workflow from external OPC/IMFP/experimental curve files into simulation, fitting, and diagnostics without changing optics, XPS, reflectivity, or fitting physics algorithms.

## Scope

- Add maintained reflectivity and rocking-curve experimental data readers under `swanx.io`.
- Reuse existing fitting data classes and preprocessing normalization routines.
- Narrow `swanx.io` public exports to the new explicit IO surface.
- Tighten malformed-row handling for IMFP data and validation for optical constants.
- Add tutorial curve files and a compact IO example.
- Update README, user guide, architecture, roadmap, project state, and TODO.
- Run the full regression suite and requested examples.

## Validation

- `python -m pytest -q`
- `python examples/io/opc_imfp_rocking_curve_quickstart.py`
- `python examples/io/experimental_curve_loading.py`
- `python examples/reflectivity/plot_lno_sto_reflectivity.py`
- `python examples/xps/plot_lno_la4d_rocking_curve.py`
- `python examples/fitting/jax_least_squares_reflectivity_fit.py`

## Non-goals

- No physics algorithm changes.
- No fitting optimizer redesign.
- No reintroduction of `swxps` imports.
