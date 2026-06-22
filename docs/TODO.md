# TODO

Last updated: 2026-06-22

## Confirmed slicing requirements

- User-configurable `max_slice_thickness`; default 2 Angstrom.
- Configurable `min_slices`; proposed default 10 per positive finite layer.
- One cell-centered grid for roughness, fields, concentration/IMFP,
  attenuation, and RC integration.
- Fixed counts from upper-bound capacity thicknesses during JAX fitting.
- Existing APIs and legacy step-based behavior remain intact.

## First implementation checkpoint

- Follow `docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md`.
- Record legacy numerical outputs and array shapes.
- Add `src/swxps/slicing.py` with policy and fixed-plan types.
- Materialize cell edges, centers, widths, and nominal/effective mappings.
- Test default and user-changed maximum thickness values.
- Test 4, 16, and 160 Angstrom layers and exact thickness conservation.
- Stop for review before changing any physics kernel.

## Later integration checkpoints

- Build graded effective optical cells from the shared grid.
- Evaluate fields at the same centers with no second depth grid.
- Add cell-centered concentration/IMFP, attenuation, and midpoint RC integration.
- Propagate the optional grid through NumPy, JAX, and `FittingProblem`.
- Verify fixed shapes across thickness sweeps larger than 1 Angstrom.
- Verify NumPy/JAX parity and fine-grid convergence.
- Benchmark compilation, repeated calls, memory, and a 160 Angstrom layer.
- Migrate one maintained case-study runner only after package validation.

## Scientific validation priorities

- Audit experimental RC preprocessing and normalization against raw data.
- Quantify sensitivity to weights, bounds, initialization, and local minima.
- Check fitted structure and angular offsets against independent expectations.
- Keep experimental results provisional until these checks are documented.

## Small maintenance items

- Replace deprecated `np.trapz` in a separate parity-tested change.
- Move completed long-form plans to `docs/history/` when no longer active.

## Session handoff checklist

- Update this file and `docs/PROJECT_STATE.md`.
- Record tests, benchmarks, design decisions, branch/commit, and blockers.
- Commit handoff documentation with the work it describes.
- Push required commits before switching computers.
