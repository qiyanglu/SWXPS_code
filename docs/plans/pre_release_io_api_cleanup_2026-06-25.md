> Current namespace note (2026-06-26): This plan is historical. Maintained code now lives under `src/swanx`, and `import swxps` is expected to fail; any `swxps` paths below are old planning context, not current guidance.

> Historical status (2026-06-25): Active implementation plan for the pre-release SWANX API cleanup and realistic OPC/IMFP IO workflow.

# Pre-release API cleanup and OPC/IMFP IO workflow

## Goals

1. Make `swanx` the only supported Python namespace before public release.
2. Remove the temporary `swxps` compatibility package and active imports.
3. Add user-facing `swanx.io` readers/builders for CXRO optical constants,
   IMFP tables, material table loading, stack construction, and core-level
   request construction.
4. Move tutorial OPC/IMFP files into `examples/data`.
5. Update README/docs/examples/tests so realistic SW-XPS workflows load files
   before constructing explicit simulation requests.

## Constraints

- Do not change physics algorithms or numerical behavior.
- Keep flat `swanx.*` compatibility modules for a later cleanup pass.
- `RockingCurveRequest` remains explicit and does not read files internally.
- File IO and interpolation happen outside JAX-traced functions.

## Validation

- Add focused IO tests for OPC, IMFP, and material/core-level builders.
- Update namespace tests so `import swanx` works and `import swxps` fails.
- Run `python -m pytest -q`.
- Run `python examples/io/opc_imfp_rocking_curve_quickstart.py`.
