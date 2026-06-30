# PROJECT_STATE

## Current state

SWANX uses `swanx` as the only supported Python namespace. The early `swxps`
namespace was removed before public release and is not an active compatibility
surface.

The primary human-editable project workflow is:

```text
swanx init my_project
        -> edit my_project/project.yaml
        -> python my_project/run_project.py
        -> my_project/runs/<project_name>_<timestamp>/ report folder
```

For custom Python workflows, the maintained object flow is:

```text
OPC + IMFP + optional experimental curves
        -> swanx.io
        -> SimulationStack / CoreLevelRequest / ReflectivityData / RockingCurveData
        -> simulation + fitting + diagnostics
```

Starter data are packaged with `swanx.project` for `swanx init` and are also
mirrored in the repository under `data/OPC/`, `data/IMFP/`, and `data/curves/`.
Maintained examples use the synthetic C/LaNiO3/SrTiO3 (C/LNO/STO) benchmark CSV
when they need reflectivity and rocking-curve data.

## Implemented workflow

- `swanx.project` validates and runs YAML ProjectSpec files.
- `swanx init my_project` creates `project.yaml`, `run_project.py`, a project
  README, a project-local `synthetic_residual_factory.py`, and by default a
  local `data/` copy of packaged C/LaNiO3/SrTiO3 starter data. The default
  project runs a JAX least-squares fit against the packaged synthetic
  reflectivity and four rocking-curve datasets.
- `swanx init --template minimal`, `--template multilayer`, and
  `--template fit-demo` generate beginner starters for the default fitting
  workflow, a simulation-only repeated multilayer, and an explicit fitting
  starter alias.
- `--copy-example-data` creates a self-contained copy from a chosen data root;
  `--data-root` points at another tutorial data root and writes relative paths
  when possible.
- `swanx inspect ...`, `swanx validate ...`, and `swanx run ...` are thin CLI
  wrappers for review, validation, and automation.
- PyYAML is optional via `python -m pip install -e ".[project]"`.
- `docs/projectspec_reference.md` is the detailed YAML ProjectSpec reference,
  and `examples/01_quickstart_projectspec/` contains copy-pasteable ProjectSpec
  examples.
- `examples/` is organized as a user learning path: ProjectSpec quickstarts,
  experimental-data loading, compact Python API scripts, fitting examples, and
  advanced low-level visualizations. The four numbered example folders now
  collectively cover the same tutorial scope as the default `swanx init`
  starter, including a runnable ProjectSpec JAX least-squares fit with an
  explicit residual factory. All maintained examples share the synthetic
  C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 case used by the benchmark folder.
- `swanx.io` reads OPC, IMFP, reflectivity, and rocking-curve files and builds
  `SimulationStack` and `CoreLevelRequest` objects from material tables.
- `swanx.preprocessing` owns rocking-curve normalization algorithms.
- `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`; maintained fitting backends live under `swanx.fitting.bo`, `swanx.fitting.jax_gradient`, and `swanx.fitting.jax_least_squares`.
- `swanx.io.__all__` is narrow and explicit; it does not export preprocessing
  functions or legacy flat helpers.

## YAML ProjectSpec notes

The current YAML ProjectSpec supports sections for `project`, `settings`, `materials`,
`parameters`, `stack`, `core_levels`, `datasets`, and `report`. The required
sections are `project`, `settings`, `materials`, `stack`, and `core_levels`;
`parameters`, `datasets`, and `report` default to empty mappings.

Supported YAML workflow features include:

- stable concrete stack layer IDs;
- layer tags and explicit core-level selection by `layer_ids`, `tags`, or
  `all: true`;
- compact repeat blocks for multilayers;
- inline parameter references and AST-whitelisted arithmetic expressions,
  including safe scalar functions (`min`, `max`, `sqrt`, `erf`,
  `linear_map`, `transition_erf`) and `repeat_index0` for zero-based repeat
  formulas;
- polarization strings `"s"`, `"p"`, and `"unpolarized"`;
- project-local default output folders and a simple Markdown `report.md`;
- opt-in progress messages for `run_project(..., progress=True)`, enabled by
  `swanx run` and generated beginner scripts;
- per-plot skipped-output notes and experimental-overlay notes in `report.md`;
- compound reflectivity-plus-rocking-curve overview plots with incident-angle
  labels and no default residual PNG;
- method-aware plot filenames: fitting runs write `fit_overview.png`,
  `reflectivity_fit.png`, and `rocking_curves_fit.png`; `simulate_only` runs
  write `simulation_overview.png`, `reflectivity_simulation.png`, and
  `rocking_curves_simulation.png`;
- stack schematic plots for all run methods;
- least-squares convergence, parameter-range, and correlation plot images when
  diagnostics are available;
- Bayesian-optimization convergence and surrogate-slice plots when diagnostics
  are available;
- optional dataset weights/log floors, off-peak RC masks, and fixed-grid slicing
  settings that pass through to the existing `FittingProblem` APIs;
- optional separate reflectivity and rocking-curve angle-offset parameters for
  ProjectSpec fitting, plus edge-polynomial normalization controls and
  per-core-level `vacuum_imfp_from_material` for legacy workflow parity;
- complete `simulate_only` report output without best-fit parameter tables;
- method-specific report writers for least-squares, gradient, and BO result
  objects.

All thickness, roughness, depth, and IMFP values are in Angstrom. In YAML,
`roughness_A` on layer j means roughness/interdiffusion at the upper interface
of layer j, i.e. the interface between layer j-1 and layer j. `repeat_index`
is 1-based inside repeat blocks; `repeat_index0` is a zero-based expression
convenience and is equivalent to `repeat_index - 1` inside repeats.

## API notes

- Beginner project runs should start with `swanx init my_project` followed by
  `python my_project/run_project.py`; advanced scripts can call
  `from swanx.project import run_project`.
- Custom simulations can start with `import swanx as sx`.
- OPC files are interpolated at photon energy.
- IMFP files are interpolated at `E_kin = h nu - E_B`.
- `RockingCurveRequest` does not read files directly.
- Unified slicing is the default high-level simulation path.
- `ReflectivityRequest`, `RockingCurveRequest`, and `FittingProblem` support
  `polarization="s"` by default, `polarization="p"`, and mixed dictionaries
  such as `{"s": 0.7, "p": 0.3}`.
- JAX least-squares/autodiff is the recommended fitting path for differentiable
  fixed-shape workflows; BO remains an optional global black-box baseline.
- YAML ProjectSpec fitting still requires user-provided factories for
  `jax_least_squares` and `jax_gradient`; no automatic no-code JAX residual
  builder is implemented. The packaged init starter includes an explicit
  factory for the synthetic C/LaNiO3/SrTiO3 case.
- ProjectSpec v1.3 package layout cleanup moves maintained backend
  implementations under `swanx.fitting` and report implementations under
  `swanx.project.reporting`; root backend modules and `swanx.project.reports`
  remain compatibility shims.

## Repository policy

- `src/swanx/` is the maintained package and only supported Python namespace.
- `tests/` contains regression tests.
- `examples/` contains compact tutorials built around the synthetic
  C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 benchmark case.
- `case_studies/` is local/private experimental input and runner space ignored
  by Git.
- `benchmarks/` contains synthetic fitting and performance benchmarks.
- `runs/` and `archive/` are local generated/superseded outputs ignored by Git.
- `docs/history/` contains archived historical handoffs and may intentionally
  mention old paths or retired namespaces.

## Cross-machine handoff notes

Git-tracked files contain the maintained package, tests, docs, public examples,
benchmark inputs, and plan/status records. The local experimental folders under
`case_studies/` and generated diagnostics under `runs/` are intentionally not
tracked, so they will not appear automatically on another computer after
`git pull`. The current Sample 12 and Sample 13 ProjectSpec case-study folders
and their run outputs should be treated as local lab artifacts; the committed
handoff record is the combination of this file, `docs/TODO.md`, and the
corresponding plan files under `docs/plans/`.

To resume on another computer, start from `docs/TODO.md`, then read the latest
dated entries at the end of this file and the plans:

- `docs/plans/projectspec_expression_functions_2026-06-30.md`
- `docs/plans/sample12_yaml_jax_ls_wrapper_2026-06-30.md`
- `docs/plans/sample12_identifiability_reduced_fit_2026-06-30.md`
- `docs/plans/sample12_reduced_v2_period_fraction_2026-06-30.md`
- `docs/plans/sample13_projectspec_identifiability_2026-06-30.md`

## Latest validation

Run these before handing off substantial changes:

```bash
python -m pytest tests/test_project_workflow.py -q
python -m pytest -q
```

ProjectSpec smoke checks:

```bash
swanx init runs/projectspec_smoke
python runs/projectspec_smoke/run_project.py
swanx inspect runs/projectspec_smoke/project.yaml
swanx validate runs/projectspec_smoke/project.yaml
```

Repository-local `templates/` were retired after `swanx init` became the
supported starter workflow and `examples/01_quickstart_projectspec/` became the
maintained copy-paste YAML surface.

Default init JAX-fit smoke validation completed on 2026-06-29: a fresh
`swanx init` project loaded `synthetic_residual_factory.py`, ran
`jax_least_squares`, wrote `fit/best_parameters.csv`, and produced fit-named
plots. A simulation-only ProjectSpec smoke wrote `simulation_overview.png`,
`reflectivity_simulation.png`, and `rocking_curves_simulation.png`. Focused
ProjectSpec workflow tests and the full suite passed afterward; the full suite
kept its expected xfail and one existing diagnostics warning.

ProjectSpec rocking-curve normalization fix completed on 2026-06-29: configured
`rocking_curve_offpeak_mask` now normalizes experimental rocking-curve datasets
with the same off-peak denominator used for simulated curves, and the packaged
C/LaNiO3/SrTiO3 JAX starter residual uses `problem.offpeak_mask` instead of a
hard-coded peak window. Re-running `myproject/run_project.py` reduced the final
objective from `0.0029710860635918292` in `myproject_20260629_194603` to
`8.906614807793117e-08` in `myproject_20260629_195454`; full validation passed
with `250 passed, 1 xfailed`.

Examples fitting-scope sweep completed on 2026-06-29: added
`examples/04_fitting/projectspec_jax_least_squares/` as a runnable ProjectSpec
JAX least-squares example matching the default init tutorial's data scope,
factory callback, off-peak rocking-curve normalization, fixed-grid slicing, and
report outputs. Example docs and workflow validation tests now point to it.

Repository documentation sweep completed on 2026-06-29: README, user guide,
ProjectSpec reference, architecture, roadmap, example docs, active plan notes,
PROJECT_STATE, TODO, and docs consistency tests were rechecked against the
current init workflow, retired `templates/` folder, and four-folder examples
scope.

Validation after this docs sweep: `python -m pytest tests\test_project_workflow.py -q`
passed with `27 passed`, and `python -m pytest -q` passed with `250 passed,
1 xfailed`.

ProjectSpec safe expression functions completed on 2026-06-30: YAML layer
expressions now support `min`, `max`, `sqrt`, `erf`, `linear_map`, and
`transition_erf` through the existing AST whitelist. `repeat_index` remains
1-based and `repeat_index0` is available as a zero-based formula convenience.
The local ignored Sample 12 YAML case-study wrapper was also rewritten as a
one-to-one audit map of the maintained bounded TRF JAX least-squares code,
including source modules, preprocessing constants, stack formulas, start-value
fallback, reference weighting, optimizer defaults, output artifacts, and
promotion behavior. It still uses its local runner because the workflow needs
custom preprocessing and a fixed-shape residual callback.

Validation after this expression-function change:
`python -m pytest tests\test_project_workflow.py -q --basetemp=runs\pytest_expression_functions`
passed with `29 passed`;
`python case_studies\sample_12\yaml_jax_least_squares_fit\run_project.py --setup-only`
passed and reported 18 parameters, 227 residuals, and matching JAX/NumPy
initial objectives; `python -m pytest -q --basetemp=runs\pytest_expression_functions_full`
passed with `252 passed, 1 xfailed, 1 warning`.

Follow-up Sample 12 YAML setup validation after the one-to-one rewrite passed
on 2026-06-30. The tightened runner checked data paths, angle grids,
background-subtraction settings, stack formulas, core-level emitting indices,
optimizer settings, and the source default output path before reporting 18
parameters, 227 residuals, reflectivity weight `0.0503187`, and matching
JAX/NumPy initial objectives.

Sample 12 ProjectSpec CLI workflow completed on 2026-06-30 in local ignored
`case_studies/sample_12/projectspec_jax_least_squares/`. This is a real
ProjectSpec `project.yaml` plus project-local `sample12_residual_factory.py`,
standard CSV inputs generated from `Reflectivity_Exp.dat` and `ExpRCs.dat`,
and a `run_project.py` wrapper. It mirrors the maintained bounded TRF JAX
least-squares setup with the C / LNO cap / graded [STO/LNO]x40 / STO stack,
edge-polynomial rocking-curve normalization, separate reflectivity and
rocking-curve angle offsets, and the legacy vacuum-IMFP convention. Validation:
`swanx inspect` and `swanx validate` both passed; the residual-factory smoke
matched the legacy initial objective (`0.00342457445517` NumPy objective,
227 residuals); `swanx run case_studies/sample_12/projectspec_jax_least_squares/project.yaml`
completed and wrote a ProjectSpec report folder with final objective
`0.003376508606357828`. Focused ProjectSpec tests passed with `30 passed`;
the full suite passed with `253 passed, 1 xfailed, 1 warning`.

Sample 12 identifiability post-processing and reduced-fit experiment were added
on 2026-06-30 in ignored local case-study files. The script
`case_studies/sample_12/analyze_lsq_identifiability.py` reads ProjectSpec
least-squares `jacobian.csv` and `residual_vector.csv`, scales Jacobian columns
by parameter ranges, writes SVD/sensitivity/correlation/dataset-sensitivity CSVs,
and generates diagnostic PNGs. On the 18-parameter Sample 12 ProjectSpec run,
the scaled-Jacobian condition number was about `2.47e4`, with weakest modes
dominated by individual roughness terms, carbon roughness fraction, substrate
roughness, and the split top-LNO cap thickness. A reduced local ProjectSpec at
`case_studies/sample_12/projectspec_jax_least_squares_reduced_v1/` fixes those
weak parameters to the 18-parameter best values and fits 10 remaining
parameters. `inspect`, `validate`, and one full run passed; the reduced run
`sample12_projectspec_jax_least_squares_reduced_v1_20260630_152036` reached
final objective `0.0034093268637466366` versus the 18-parameter objective
`0.003376508606357828`, while improving the scaled condition number to about
`230.6`. Next Sample 12 cleanup candidates are tied reflectivity/RC angle
offsets and a physical superlattice parameterization using average period,
LNO fraction, and gradient terms.

Sample 12 reduced-v2 period/fraction experiment completed on 2026-06-30 in
ignored local folder
`case_studies/sample_12/projectspec_jax_least_squares_reduced_v2_period_fraction/`.
This variant ties reflectivity and rocking-curve angle offsets to one fitted
`angle_offset` and replaces separate STO/LNO start+delta thicknesses with
`sl_period_start`, `sl_lno_fraction_start`, `sl_period_delta`, and
`sl_lno_fraction_delta`. `inspect`, `validate`, and one full run passed. The run
`sample12_projectspec_jax_least_squares_reduced_v2_period_fraction_20260630_154430`
reached objective `0.0034975009751623707` with no active bounds. Identifiability
outputs were written to root `runs/sample12_v2_identifiability_20260630_154430/`
to avoid Windows path-length limits. The scaled condition number was `1180.37`,
better than the 18-parameter baseline but worse than reduced-v1 (`230.6`), and
the strongest correlations moved into the period/fraction/transition block.
Current interpretation: tying the offsets is likely not justified for this
sample because reflectivity and RCs were measured separately; the next better
test is separate offsets plus a simpler superlattice model such as period start,
period delta, and constant LNO fraction.

Sample 13 ProjectSpec JAX least-squares mirror and reduced-fit experiment were
added on 2026-06-30 in ignored local folders under `case_studies/sample_13/`.
The 1:1 ProjectSpec folder
`projectspec_jax_least_squares_all_rcs/` mirrors the maintained all-RC TRF JAX
least-squares runner with the cap3 stack, Ni 3p emitted only from LNO-2, La 4d
emitted from LNO-1 and LNO-2, separate reflectivity/rocking-curve angle offsets,
pre-normalized edge-polynomial rocking-curve data, and simulated rocking curves
kept on mean normalization. `inspect`, `validate`, a full ProjectSpec run, and
identifiability diagnosis passed. Corrected run
`sample13_projectspec_jax_least_squares_all_rcs_20260630_160057` reached final
objective `0.008341284463588628`; diagnosis outputs are in
`runs/sample13_baseline_identifiability_20260630_160057/` and show a nearly
singular scaled Jacobian with condition number `5.77824e+18`, dominated by
roughness, transition-shape, and bound-hit directions.

Sample 13 reduced-v1 in
`case_studies/sample_13/projectspec_jax_least_squares_reduced_v1/` fixed the
weak roughness/profile/transition/cap-split parameters to the corrected
baseline best values and fitted only `carbon_thickness`,
`top_lno_total_thickness`, `sto_thickness_start`, `lno_thickness_start`,
`reflectivity_angle_offset`, and `rc_angle_offset`. `inspect`, `validate`, a
full run, and identifiability diagnosis passed. Run
`sample13_projectspec_jax_least_squares_reduced_v1_20260630_160620` reached
objective `0.007912524625745022`, slightly better than the 18-parameter
baseline, and reduced the scaled-Jacobian condition number to `2349.35`.
Remaining issues are carbon thickness at the lower bound, both angle offsets at
the upper bound, and strong anti-correlation between STO and LNO starting
thicknesses. Next Sample 13 tests should prioritize angular calibration/bounds
and a period/fraction superlattice parameterization before reintroducing
individual roughness parameters.

Full repository handoff sweep completed on 2026-06-30 before committing this
snapshot. Active README, roadmap, user guide, ProjectSpec reference, examples,
benchmarks, TODO, plans, and PROJECT_STATE were checked for stale ProjectSpec,
example-case, namespace, and case-study-transfer wording. One stale benchmark
ProjectSpec note from before safe expression functions was corrected: the
synthetic C/LaNiO3/SrTiO3 benchmark ProjectSpec YAMLs now use the exact
`min(5.0, carbon_thickness)` carbon-roughness expression directly. The
cross-machine handoff note above clarifies that local `case_studies/` and
`runs/` artifacts are intentionally not transferred by Git.

Validation after this handoff sweep:
`swanx validate` passed for the three benchmark ProjectSpec YAMLs under
`benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/`;
`python -m pytest tests\test_project_workflow.py -q --basetemp=runs\pytest_handoff_project_workflow`
passed with `30 passed`; and
`python -m pytest -q --basetemp=runs\pytest_handoff_full` passed with
`253 passed, 1 xfailed, 1 warning`. The warning is the existing diagnostics
rank-deficient covariance projection warning.
