# Step 9 experimental data IO and workflow cleanup

> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

> Current examples note (2026-06-29): the maintained example learning path now
> lives under `examples/01_quickstart_projectspec/`,
> `examples/02_experimental_data/`, `examples/03_python_api/`,
> `examples/04_fitting/`, and `examples/advanced/`.

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
- `python examples/03_python_api/build_from_opc_imfp.py`
- `python examples/02_experimental_data/load_and_overlay_curves.py`
- `python examples/03_python_api/simulate_reflectivity.py`
- `python examples/advanced/xps_rocking_curves/plot_lno_la4d_rocking_curve.py`
- `python examples/04_fitting/jax_least_squares_reflectivity_fit.py`

## Non-goals

- No physics algorithm changes.
- No fitting optimizer redesign.
- No reintroduction of `swxps` imports.
