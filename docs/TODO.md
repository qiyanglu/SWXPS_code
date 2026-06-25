# TODO

Last updated: 2026-06-25

## Current pre-release cleanup

- [x] Remove the old `swxps` namespace before public release.
- [x] Replace active Python `swxps` imports with `swanx` imports.
- [x] Add `swanx.io` OPC and IMFP readers.
- [x] Add material table loading, stack builder, and core-level builder helpers.
- [x] Move tutorial OPC/IMFP files to `examples/data/OPC` and `examples/data/IMFP`.
- [x] Add a maintained OPC/IMFP rocking-curve quickstart example.
- [x] Update README and active docs for the file-based workflow.
- [x] Replace deprecated `np.trapz` with `np.trapezoid`.
- [x] Run full tests and quickstart example for this stage.
- [x] Fix fallout from namespace removal or IO parser edge cases found by tests.

## Near-term project priorities

1. Validate the new OPC/IMFP workflow on tutorial data and representative case-study inputs.
2. Keep README-linked examples executable and based on `swanx` only.
3. Improve JAX least-squares fitting documentation for fixed-shape workflows.
4. Add experimental reflectivity/rocking-curve CSV readers when the desired file conventions are clear.
5. Add fit-result export workflow docs for users.
6. Continue physical validation of RC preprocessing, weighting, angular offsets, parameter identifiability, and fitted structures.

## Maintenance

- Avoid adding new core physics until the user-facing API and validation workflows settle.
- Keep generated outputs in `runs/`; keep superseded experiments in `archive/`.
- Do not commit/push unless the user explicitly requests it in the current request.

## Latest validation

- `python -m pytest -q`: 192 passed, 1 xfailed.
- `python examples/io/opc_imfp_rocking_curve_quickstart.py`: completed successfully.
