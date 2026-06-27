# AGENTS.md

SWANX (`swanx`) provides transparent Python tools for multilayer X-ray
reflectivity, standing-wave XPS, fitting, and diagnostics.

## Core rules

- Preserve validated physics and numerical behavior unless the user explicitly
  asks for a physics change.
- Prefer small, readable, tested changes over broad rewrites.
- Use Python/NumPy for core numerical code; use SciPy only for optimization or
  numerical tooling.
- Keep file IO, preprocessing, fitting, and physics kernels separated.
- Do not stage, commit, amend, or push unless the user explicitly asks for that
  Git action in the current request.

## Physics conventions

- Angles are grazing incidence angles in degrees.
- Photon energy is in eV.
- Wavelength, thickness, roughness, depth, and IMFP are in Angstrom.
- Refractive index convention: `n = 1 - delta + i beta`.
- Layer stacks start with vacuum and end with a semi-infinite substrate.
- s-polarization is the default and backward-compatible baseline; p and mixed
  polarization paths are supported and require focused regression checks when
  touched.

## Tests and validation

- Preserve reflectivity regression coverage for Fresnel limits, identical-index
  near-zero reflectivity, multilayer Bragg peaks, and `R <= 1` tolerance.
- Add focused tests for new field, XPS, preprocessing, fitting, IO, ProjectSpec,
  or backend behavior.
- Run `python -m pytest` for substantial code changes; use a `runs/pytest_*`
  basetemp if Windows/OneDrive temp permissions interfere.

## YAML ProjectSpec workflow

- The primary human-editable workflow is `swanx init my_project`, edit
  `my_project/project.yaml`, then run `python my_project/run_project.py`.
- YAML support is optional via `python -m pip install -e ".[project]"`; keep
  PyYAML lazy-loaded and out of core dependencies.
- Default ProjectSpec outputs belong under the YAML/project folder,
  `my_project/runs/<project_name>_<timestamp>/`, and every run writes
  `report.md`.
- `swanx init --copy-example-data` should create a self-contained tutorial
  starter; `--data-root` should write paths relative to `project.yaml` when
  possible and must not silently generate invalid YAML paths.
- JAX least-squares is the recommended fitting path for differentiable
  fixed-shape workflows. Bayesian optimization remains an optional global
  black-box baseline. YAML JAX fitting may require explicit factory callbacks;
  do not add automatic no-code JAX residual generation without a separate plan.
- ProjectSpec values use Angstrom for `thickness_A` and `roughness_A`;
  `roughness_A` is the upper-interface roughness of that layer, and
  `repeat_index` is 1-based inside repeat blocks.

## Repository map

- `src/swanx`: maintained package and only supported Python namespace.
- `tests`: regression tests.
- `examples`: compact tutorial scripts and small tutorial data.
- `case_studies`: local/private experimental inputs and runners; ignored by Git.
- `benchmarks`: synthetic fitting and performance benchmarks.
- `runs`: local generated outputs and smoke projects; ignored by Git.
- `archive`: local superseded experiments; ignored by Git.
- `docs`: architecture, roadmap, plans, project state, TODO, and history.

## Planning and continuity

- For substantial work, create or update a concise plan under `docs/plans`.
- At the end of substantial coding sessions, update `docs/PROJECT_STATE.md` and
  `docs/TODO.md` so work can continue from another machine.
