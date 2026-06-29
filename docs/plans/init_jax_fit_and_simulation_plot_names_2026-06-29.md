# Init JAX fit and simulation plot names

## Goal

Make the default `swanx init` project demonstrate the full synthetic
C/LaNiO3/SrTiO3 fitting workflow instead of only `simulate_only`, and make
simulation-only plot filenames/readme notes stop using fit-oriented names.

## Scope

- Change generated init YAML to use the synthetic C/LaNiO3/SrTiO3 datasets,
  JAX least-squares, and a project-local residual-factory entry point.
- Keep explicit simulation-only examples available for users who want only
  forward modeling.
- Update ProjectSpec plot naming so `simulate_only` writes
  `simulation_overview.png`, `reflectivity_simulation.png`, and
  `rocking_curves_simulation.png`; fitting runs keep the existing fit filenames.
- Sweep README, examples, benchmarks, and docs for stale statements after the
  init workflow change.

## Validation

- Run a fresh `swanx init` project and verify it performs a JAX least-squares
  fit, writes fitted parameters, and produces fit-named plots.
- Run a simulation-only ProjectSpec and verify it produces simulation-named
  plots.
- Run focused ProjectSpec workflow tests and the full test suite.

## Result

- Default `swanx init` now writes an active JAX least-squares fitting project
  with local `synthetic_residual_factory.py`.
- `simulate_only` now writes simulation-specific curve plot filenames.
- Validation completed on 2026-06-29 with a fresh default-init fitting smoke,
  a simulation-only plotting smoke, focused ProjectSpec tests, and the full
  test suite passing.
