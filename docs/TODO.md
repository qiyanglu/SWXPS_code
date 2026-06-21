# TODO

Last updated: 2026-06-21

## Start of the next session

- Pull the latest `main` branch and confirm `git status` is clean.
- Install the package and only the optional extras required for the task.
- Run `python -B -m pytest -q -p no:cacheprovider`.
- Run `python benchmarks/performance/profile_forward_workflow.py` and record
  the new machine's baseline before comparing performance.

## Next implementation milestone

- Profile one maintained Sample 12 or Sample 13 fitting-objective evaluation.
- Identify duplicated stack, dataset, core-level, and fitting configuration in
  the maintained case-study runners.
- Propose one small shared configuration/builder extraction; do not begin a
  wholesale source rewrite or rewrite every example at once.
- Add focused parity tests before changing the case-study runners.
- Keep the existing public APIs and all core physical behavior unchanged.

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
