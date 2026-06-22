# TODO

Last updated: 2026-06-22

## Next decision: slicing semantics

- Confirm that 2 Angstrom is the maximum permitted slice thickness, not the minimum.
- Confirm initial defaults of at least 10 slices per finite nominal layer and a
  maximum 2 Angstrom slice thickness.
- Confirm that the new strategy is opt-in for the first implementation while
  legacy `field_step` and `roughness_step` behavior remains unchanged.
- Use a capacity stack built at fit upper bounds to freeze JAX slice counts,
  unless a better explicit mapping is demonstrated.
- Prototype whether roughness and field/XPS sampling can share one planned grid
  without changing the validated physical integration.

## Planned slicing implementation

- Follow `docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md`.
- Record legacy outputs and shapes before editing numerical code.
- Implement a pure, backend-independent slice-count planner with focused tests.
- Add an opt-in fixed-count roughness grid while preserving the existing step path.
- Add an opt-in fixed-shape field/XPS depth grid with correct nominal-material mapping.
- Verify fixed shapes across thickness sweeps larger than 1 Angstrom.
- Verify NumPy/JAX parity and convergence against a fine legacy reference.
- Benchmark compilation, repeated calls, memory, and a 160 Angstrom layer.
- Migrate only one maintained case-study runner after package-level validation.

## Scientific validation priorities

- Audit experimental rocking-curve preprocessing and normalization against raw data.
- Quantify sensitivity to weights, bounds, initialization, and local minima.
- Check fitted thickness, roughness, chemistry, and angular offsets against
  independent physical expectations.
- Keep experimental results labeled provisional until those checks are documented.

## Small maintenance items

- Replace deprecated `np.trapz` usage with `np.trapezoid` in a separate,
  parity-tested change.
- Review completed plans under `docs/plans/` and move long-form completed logs
  to `docs/history/` when they are no longer active references.

## Session handoff checklist

- Update this file to distinguish completed, remaining, and newly discovered work.
- Update `docs/PROJECT_STATE.md` with tests, benchmark results, important design
  decisions, current branch/commit context, and any known blockers.
- Commit the handoff documentation with the code it describes.
- Ensure required commits are pushed before switching computers; ignored
  `runs/` and `archive/` files must be transferred separately if needed.
