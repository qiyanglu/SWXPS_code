# TODO

## Current priorities

- [ ] Keep README, user guide, and maintained examples executable and
      aligned with `swanx`.
- [ ] Keep `examples/` organized as a user learning path and avoid adding
      benchmark-style or regression-test material there.
- [ ] Keep maintained examples aligned with the synthetic
      C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 benchmark case unless a future
      user-facing case is intentionally chosen.
- [ ] Add richer ProjectSpec examples only when they reuse existing IO,
      simulation, fitting, and report APIs.
- [ ] Validate root `data/OPC`, `data/IMFP`, benchmark CSV, and packaged
      starter-data workflows on tutorial inputs and local representative
      case-study inputs when available.
- [ ] Keep the default `swanx init` JAX least-squares starter lightweight enough
      for a first run while still demonstrating the full ProjectSpec workflow.
- [ ] Continue validating RC preprocessing, weighting, angular offsets,
      parameter identifiability, fitted structures, and s/p/mixed polarization.
- [ ] Add richer experimental-data formats only when real lab conventions
      require them.
- [ ] Add another ProjectSpec JAX custom-residual example using real
      experimental data once the lab-facing fitting conventions settle.
- [ ] For Sample 12, test reduced-v3 with separate reflectivity/rocking-curve
      angle offsets and a simpler superlattice model, such as period start,
      period delta, and constant LNO fraction.
- [ ] For Sample 13, test a reduced-v2 ProjectSpec with separate
      reflectivity/rocking-curve angle offsets but a physical superlattice
      parameterization such as total period plus LNO fraction. Also review
      whether the angle-offset bounds should be widened or the raw angle grids
      should be pre-aligned, since reduced-v1 pushed both offsets to the upper
      bound.

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
- [x] Add a synthetic C/LaNiO3/SrTiO3 ProjectSpec JAX least-squares benchmark
      using fixed-shape least-squares residuals.
- [x] Make default ProjectSpec run outputs project-local and write `report.md`.
- [x] Retire repository-local `templates/` now that `swanx init` and
      examples are the starter surfaces.
- [x] Support stack layer IDs/tags, repeat blocks, inline parameters, safe
      arithmetic expressions, and core-level layer/tag resolution.
- [x] Add safe ProjectSpec expression functions and `repeat_index0` for
      SWOPT-style layer-gradient and transition formulas without enabling
      arbitrary Python execution.
- [x] Add a local CLI-runnable Sample 12 ProjectSpec JAX least-squares workflow
      that mirrors the maintained bounded TRF script through a project-local
      residual factory.
- [x] Add local Sample 12 least-squares identifiability diagnostics and a
      reduced-v1 ProjectSpec experiment that fixes weak roughness/split-cap
      parameters.
- [x] Test Sample 12 reduced-v2 with tied angle offset and period/fraction
      superlattice parameters; it ran cleanly but was less favorable than
      reduced-v1.
- [x] Add a local CLI-runnable Sample 13 ProjectSpec JAX least-squares workflow
      mirroring the maintained all-RC TRF script, diagnose its scaled Jacobian,
      and test a reduced-v1 fit that removes weak roughness/profile/transition
      directions.
- [x] Add a synthetic benchmark ProjectSpec least-squares identifiability
      analyzer that reports range-scaled parameter sensitivity, SVD weak modes,
      dataset sensitivity, correlations, and plots from existing run artifacts.
- [x] Complete a full handoff-oriented repo sweep after the Sample 12/13
      ProjectSpec and identifiability work, update status docs, validate active
      benchmark ProjectSpecs, and rerun focused plus full test suites.
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
- [x] Realign maintained examples from the older single-film tutorial to the
      synthetic C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 benchmark case.
- [x] Sweep maintained docs for stale example-case wording, old public README
      mojibake, and outdated active guidance after the examples realignment.
- [x] Change default `swanx init` from simulation-only to a packaged
      C/LaNiO3/SrTiO3 JAX least-squares fitting starter using the internal
      `auto_fixed_grid` residual path.
- [x] Rename ProjectSpec `simulate_only` curve plots to simulation-specific
      filenames while preserving fit-specific filenames for fitting runs.
- [x] Fix ProjectSpec off-peak rocking-curve normalization so experimental
      datasets and simulated/fitted curves use the same denominator.
- [x] Add a runnable ProjectSpec JAX least-squares fitting example under
      `examples/04_fitting/` so the four numbered example folders collectively
      match the default `swanx init` tutorial scope.
- [x] Re-sweep active docs after adding the runnable ProjectSpec fitting example
      and keep status, roadmap, examples, and callback-factory guidance aligned.
- [x] Add unified ProjectSpec `run:` controls for mode, optimizer settings, and
      output switches while preserving legacy `settings`/`report` compatibility.
- [x] Promote least-squares identifiability analysis into packaged ProjectSpec
      reporting via `run.outputs.identifiability`, including scaled sensitivity,
      weak-mode, correlation, dataset-sensitivity, plot, and summary outputs.
- [x] Add internal ProjectSpec `run.optimizer.residual: "auto_fixed_grid"` for
      fixed-topology JAX least-squares fits and stop generating project-local
      residual factory scripts in `swanx init`.
- [x] Sweep maintained examples so ProjectSpec YAML uses unified `run:`
      controls, remove factory-script guidance from examples, and rerun the
      maintained YAML and Python examples.
- [x] Sweep benchmark ProjectSpec YAMLs so they use unified `run:` controls,
      remove obsolete benchmark-local factory/analyzer files, and delete old
      generated benchmark runs/caches.
- [x] Expand the ProjectSpec YAML reference with detailed `run:` controls,
      normalization, slicing, datasets, optimizer, report/output, and common
      mistake guidance.
- [x] Make edge-polynomial rocking-curve normalization the ProjectSpec default
      across experimental data, simulation outputs, BO/generic fitting, and
      auto fixed-grid JAX least-squares.
- [x] Sweep active docs, examples, and benchmarks after the edge-polynomial
      default; align Python examples and synthetic benchmark scripts with the
      first/last 10 percent polynomial RC normalization convention.
- [x] Use the fitting-mode core-level color scheme for simulation-only
      ProjectSpec rocking-curve overview plots.
- [x] Rewrite README as a concise landing page, add main-report fit
      interpretation notes, and add focused auto-fixed-grid reliability tests.
- [x] Add ProjectSpec inspect Doctor diagnostics, clearer `fit`/`simulate`
      starter aliases, report recommended-next-checks guidance, and docs
      freshness tests that avoid pinned exact pytest counts.
- [x] Add optional ProjectSpec `run.outputs.next_project` reports that write
      best-start and low-sensitivity reduced YAML files for follow-up fits.
- [x] Expose `run.outputs.next_project` in the maintained ProjectSpec fitting
      example and align README, example docs, and YAML reference wording.

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
