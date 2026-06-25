# TODO

## Current cleanup

- [x] Keep `swanx` as the only supported namespace.
- [x] Keep tutorial inputs under root-level `data/`.
- [x] Keep `swanx.io` narrow: OPC, IMFP, material-table builders, stack/core
  builders, and experimental curve readers.
- [x] Keep rocking-curve normalization under `swanx.preprocessing`.
- [x] Keep fitting data consumption under `swanx.fitting`.
- [x] Run and record Step 10 final validation.

## Near-term priorities

1. Validate root `data/OPC`, `data/IMFP`, and `data/curves` on tutorial and
   representative case-study inputs.
2. Keep README-linked examples executable and based on `swanx`.
3. Improve fixed-shape JAX least-squares fitting documentation.
4. Add richer experimental-data formats only when real lab conventions require
   them.
5. Add fit-result and best-fit-curve export workflow docs.
6. Continue validating RC preprocessing, weighting, angular offsets, parameter
   identifiability, and fitted structures.

## Maintenance

- Avoid new core physics until user-facing workflows and validation settle.
- Keep generated outputs in `runs/`; keep superseded experiments in `archive/`.
- Do not commit/push unless the user explicitly requests it in the current
  request.
