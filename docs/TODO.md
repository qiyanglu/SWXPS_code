# TODO

## Current priorities

- [ ] Keep README, user guide, templates, and maintained examples executable and
      aligned with `swanx`.
- [ ] Keep `examples/` organized as a user learning path and avoid adding
      benchmark-style or regression-test material there.
- [ ] Keep maintained examples aligned with the synthetic C/[LNO/STO]x20/STO
      benchmark case unless a future user-facing case is intentionally chosen.
- [ ] Add richer ProjectSpec examples only when they reuse existing IO,
      simulation, fitting, and report APIs.
- [ ] Validate root `data/OPC`, `data/IMFP`, benchmark CSV, and packaged
      starter-data workflows on tutorial inputs and local representative
      case-study inputs when available.
- [ ] Continue validating RC preprocessing, weighting, angular offsets,
      parameter identifiability, fitted structures, and s/p/mixed polarization.
- [ ] Add richer experimental-data formats only when real lab conventions
      require them.
- [ ] Add another ProjectSpec JAX callback example using real experimental data
      once the lab-facing fitting conventions settle.

## Recently completed

- [x] Add optional PyYAML dependency under the `project` extra.
- [x] Add `swanx.project.validate_project(...)` and `run_project(...)`.
- [x] Add `swanx validate` and `swanx run` CLI wrappers.
- [x] Add `swanx init my_project` for beginner YAML project setup.
- [x] Add ProjectSpec initializer options for copied example data and
      explicit data roots.
- [x] Add ProjectSpec initializer templates, packaged tutorial data,
      `swanx inspect`, and richer Markdown report notes.
- [x] Improve ProjectSpec fitting plots with a compound overview, incident-angle
      labels, stack schematics, LS convergence/parameter/correlation diagnostics,
      BO convergence/surrogate diagnostics, and no default residual PNG.
- [x] Add visible ProjectSpec run progress messages for `swanx run` and
      generated beginner scripts.
- [x] Add a synthetic C/LNO/STO ProjectSpec JAX least-squares benchmark with
      an explicit fixed-shape residual factory callback.
- [x] Make default ProjectSpec run outputs project-local and write `report.md`.
- [x] Add `templates/project_minimal.yaml` and `templates/run_project.py`.
- [x] Support stack layer IDs/tags, repeat blocks, inline parameters, safe
      arithmetic expressions, and core-level layer/tag resolution.
- [x] Implement `simulate_only` report folder outputs without best-fit parameter
      tables.
- [x] Add method-specific report writers for least-squares, gradient, and BO
      result-like objects, including split BO evaluations and best-so-far files.
- [x] Remove obsolete active tests that depended on ignored local
      `case_studies/` files.
- [x] Move optimizer-independent fitting implementation to
      `swanx.fitting.core` while keeping `swanx._fitting` as a local-script
      compatibility shim.
- [x] Keep `swanx` as the only supported namespace.
- [x] Keep rocking-curve normalization under `swanx.preprocessing`.
- [x] Keep fitting data consumption under `swanx.fitting`.
- [x] Move maintained fitting backends under `swanx.fitting` while keeping
      root backend modules as compatibility shims.
- [x] Split ProjectSpec report writers under `swanx.project.reporting` while
      keeping `swanx.project.reports` as the compatibility facade.
- [x] Add a background-first README, practical ProjectSpec user guide, detailed
      YAML reference, and copy-pasteable ProjectSpec examples.
- [x] Redesign `examples/` around beginner ProjectSpec workflows, compact
      Python API scripts, fitting examples, and advanced visualizations.
- [x] Realign maintained examples from the simple LNO/STO film tutorial to the
      synthetic C/[LNO/STO]x20/STO benchmark case.
- [x] Sweep maintained docs for stale example-case wording, old public README
      mojibake, and outdated active guidance after the examples realignment.

## Deferred or out of scope

- [ ] Excel frontend.
- [ ] GUI frontend.
- [ ] JSON ProjectSpec input.
- [ ] HTML report frontend; the current ProjectSpec report is Markdown only.
- [ ] Online optical-constant database integration.
- [ ] New optimizers without a demonstrated validation need.

## Maintenance rules

- Avoid new core physics until user-facing workflows and validation settle.
- Keep generated outputs in project-local or root `runs/`; keep superseded
  experiments in `archive/`.
- Keep local/private experimental inputs and runners in ignored `case_studies/`.
- Do not commit/push unless explicitly requested in the current request.
