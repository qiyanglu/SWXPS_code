# PROJECT_STATE

Last updated: 2026-06-25

## Current state

SWANX is in a pre-release API cleanup stage. The only supported Python namespace is now `swanx`; the early development `swxps` compatibility package has been removed from `src/` and namespace tests now expect `import swxps` to fail.

The package identity is:

```text
SWANX = Standing-Wave Analysis for Nanoscale X-ray spectroscopy
Python import = swanx
```

## Implemented in the current working tree

- Removed `src/swxps/` and `src/swanx/_legacy_api.py`.
- Replaced active Python `swxps` imports with `swanx` canonical imports.
- Added `swanx.io` readers/builders:
  - `read_optical_constants(...)` with default CXRO `Energy(eV), Delta, Beta` parsing.
  - `OpticalConstantTable.at_energy(...)` returning `(delta, beta)`.
  - `read_imfp(...)` and `IMFPTable.at_kinetic_energy(...)`.
  - `load_material_tables(...)`.
  - `stack_from_layer_specs(...)`.
  - `core_level_from_tables(...)` and `core_levels_from_specs(...)`.
- Moved tutorial OPC/IMFP files from root `OPC/` and `IMFP/` into:
  - `examples/data/OPC/`
  - `examples/data/IMFP/`
- Added `examples/io/opc_imfp_rocking_curve_quickstart.py`.
- Updated README quickstart to use OPC/IMFP files instead of manually typed `delta`, `beta`, and IMFP values.
- Updated `docs/user_guide.md`, `docs/architecture.md`, and `docs/roadmap.md` for the swanx-only namespace and realistic file-based workflow.
- Replaced deprecated `np.trapz` with `np.trapezoid` in XPS intensity integration.

## API/workflow notes

- Beginner users should start with `import swanx as sx`.
- Advanced users should import from `swanx.stack`, `swanx.optics`, `swanx.xps`, `swanx.fitting`, `swanx.diagnostics`, and `swanx.io`.
- `RockingCurveRequest` stays explicit. It receives a stack with already-resolved optical constants and core-level requests with already-resolved IMFP dictionaries.
- OPC files are interpolated at photon energy.
- IMFP files are interpolated at photoelectron kinetic energy, `E_kin = h nu - E_B`.
- JAX fitting remains the recommended path for differentiable fixed-shape workflows. OPC/IMFP file reading remains outside traced residual functions.

## Validation status

Validation completed in this session:

```bash
python -m pytest -q
# 192 passed, 1 xfailed

python examples/io/opc_imfp_rocking_curve_quickstart.py
# reflectivity points: 201
# La 4d kinetic energy: 795.0 eV
```

The quickstart intentionally uses one finite emitting LNO core level. A substrate-only STO core level in this minimal stack would produce zero finite-depth XPS signal and fail normalization, which is correct behavior rather than an IO issue.

## Git status

No commit has been requested for this stage yet. Leave changes in the working tree unless the user explicitly asks to commit/push.
