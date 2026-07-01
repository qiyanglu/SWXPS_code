# ProjectSpec YAML Reference

This reference describes the human-editable SWANX `project.yaml` format.
ProjectSpec YAML resolves file paths, material tables, stack layers, core-level
selectors, datasets, fitting parameters, run controls, and report options before
calling the maintained SWANX simulation, fitting, and reporting APIs.

For a runnable full fitting example, see
`examples/04_fitting/projectspec_jax_least_squares`.

Useful commands while editing a ProjectSpec:

```bash
swanx inspect project.yaml
swanx validate project.yaml
swanx run project.yaml
```

## Top-Level Structure

```yaml
project:
run:
settings:
materials:
parameters:
stack:
core_levels:
datasets:
report:
```

Required sections:

- `project`
- `settings`
- `materials`
- `stack`
- `core_levels`

Optional sections that default to empty mappings:

- `run`
- `parameters`
- `datasets`
- `report`

The preferred modern control surface is `run:`. Older fields
`settings.fit_method`, `settings.optimizer`, `report.save_plots`, and
`report.identifiability` are still accepted when they do not conflict with
`run:`.

## Minimal Skeleton

```yaml
project:
  name: "my_project"

run:
  mode: "simulate_only"
  outputs:
    plots: true

settings:
  photon_energy_ev: 1000.0
  angle_start_deg: 6.9
  angle_stop_deg: 10.9
  angle_count: 161
  polarization: "s"
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2

materials: {}
parameters: {}
stack: []
core_levels: []
datasets: {}
report: {}
```

`report: {}` and `datasets: {}` mean "this section is intentionally empty."
They do not enable any output or data behavior by themselves.

## `project`

```yaml
project:
  name: "my_project"
  output_dir: "optional_relative_or_absolute_path"
```

Fields:

- `name`: used in the default output folder name and report title.
- `output_dir`: optional output folder.

Path behavior:

- Relative `output_dir` values are resolved relative to `project.yaml`.
- Absolute `output_dir` values are respected.
- If `output_dir` is absent, output goes to:

```text
project_yaml_dir/runs/<project_name>_<timestamp>/
```

## `run`

`run:` is the preferred section for execution mode, optimizer settings, and
optional run outputs.

Simulation only:

```yaml
run:
  mode: "simulate_only"
  outputs:
    plots: true
```

Auto fixed-grid JAX least-squares:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 80
    ftol: 1.0e-8
    xtol: 1.0e-8
    gtol: 1.0e-8
    record_history: true
    estimate_covariance: true
  outputs:
    plots: true
    identifiability: true
```

Bayesian optimization:

```yaml
run:
  mode: "bayesian_optimization"
  optimizer:
    n_calls: 100
    n_initial_points: 30
    acquisition_function: "EI"
    random_state: 7
    show_progress: false
  outputs:
    plots: true
```

Fields:

- `mode`: one of `"simulate_only"`, `"jax_least_squares"`,
  `"jax_gradient"`, or `"bayesian_optimization"`.
- `optimizer`: method-specific settings.
- `outputs.plots`: write plots when matplotlib is available.
- `outputs.identifiability`: boolean or mapping. When enabled for
  `jax_least_squares`, SWANX writes `identifiability_analysis/`.

Conflict rules:

- `run.mode` conflicts with `settings.fit_method` if both are set differently.
- `run.optimizer.*` conflicts with `settings.optimizer.*` if the same key has a
  different value.
- `run.outputs.plots` conflicts with `report.save_plots` if the values differ.
- `run.outputs.identifiability` conflicts with `report.identifiability` if the
  enabled/disabled choice differs.

## `settings`

`settings:` contains physical simulation settings and preprocessing choices.

Common angle grid:

```yaml
settings:
  photon_energy_ev: 1000.0
  angle_start_deg: 6.9
  angle_stop_deg: 10.9
  angle_count: 161
  polarization: "s"
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
```

Explicit angle grid:

```yaml
settings:
  photon_energy_ev: 1000.0
  angles_deg: [6.9, 7.0, 7.1]
  polarization: "unpolarized"
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
```

Fields:

- `photon_energy_ev`: photon energy in eV. Required.
- `angle_start_deg`, `angle_stop_deg`, `angle_count`: generated incident-angle
  grid in degrees.
- `angles_deg`: explicit incident-angle list in degrees.
- `polarization`: `"s"`, `"p"`, or `"unpolarized"`.
- `normalization`: default rocking-curve normalization mode, usually
  `"edge_polynomial"`.
- `normalization_edge_fraction`: edge fraction used by
  `"edge_polynomial"`. Default `0.10`, meaning the first 10 percent and last
  10 percent of each rocking curve.
- `normalization_polynomial_order`: polynomial order used by
  `"edge_polynomial"`. Default `2`.
- `field_step`: legacy depth grid step in Angstrom when `slicing: "legacy"` is
  selected. Default `1.0`.
- `roughness_step`: legacy roughness discretization step in Angstrom when
  `slicing: "legacy"` is selected. Default `1.0`.
- `roughness_profile`: roughness profile name for compatible paths. Default
  `"erf"`.
- `simulation_backend`: backend hint used by generic fitting paths. Default
  `"numpy"`.

Angles in fitting runs:

- With datasets, fitting and overlays use the dataset angles.
- Without datasets, `simulate_only` needs `angles_deg` or
  `angle_start_deg`/`angle_stop_deg`/`angle_count`.

### Polarization

```yaml
settings:
  polarization: "s"
```

Allowed values:

- `"s"`: s polarization.
- `"p"`: p polarization.
- `"unpolarized"`: 50/50 s/p mixture.

Circular polarization is not implemented.

### Angle Offsets

By default, a parameter named `angle_offset` is used as a shared calculation
angle offset if it exists and is varied or fixed in `parameters`.

```yaml
parameters:
  angle_offset:
    initial: 0.03
    lower: -0.25
    upper: 0.25
    vary: true
```

Advanced split offsets:

```yaml
settings:
  angle_offset_parameter: null
  reflectivity_angle_offset_parameter: "reflectivity_angle_offset"
  rocking_curve_angle_offset_parameter: "rc_angle_offset"
```

Use this when reflectivity and rocking-curve scans need independent angular
offsets.

## Rocking-Curve Normalization

### Edge-Polynomial Background Normalization

```yaml
settings:
  normalization: "edge_polynomial"
  normalization_edge_fraction: 0.10
  normalization_polynomial_order: 2
```

This is the recommended default for ProjectSpec rocking curves. It uses the
first and last fraction of each rocking curve to fit a polynomial background,
then normalizes by that background. `0.10` means first 10 percent and last 10
percent. Values greater than 1 are treated as percentages, so `10` also means
10 percent.

Use this as the universal default unless you have already prepared
pre-normalized rocking curves outside SWANX. The same mode is applied to
experimental data, simulation-only rocking curves, fitted rocking curves, and
the internal auto fixed-grid JAX residual. This avoids comparing RC data and
model curves that were scaled by different rules.

Supported paths:

- experimental rocking-curve data loaded from YAML datasets;
- simulation-only reports;
- generic fitting and Bayesian optimization;
- no-code fixed-grid JAX least-squares with
  `run.optimizer.residual: "auto_fixed_grid"`.

This setting applies to SW-XPS rocking curves only. Reflectivity remains
reflectivity and is usually scored in log space for fitting.

Practical requirements:

- enough points must be available at the two curve edges to fit the requested
  polynomial order;
- with the default order `2`, the selected edge points must be more than two
  total points;
- for very sparse curves, increase `normalization_edge_fraction` or lower
  `normalization_polynomial_order`.

### Mean Normalization

```yaml
settings:
  normalization: "mean"
```

Mean normalization divides each rocking curve by a scalar mean. If no mask is
configured, the mean uses all rocking-curve points. It remains supported for
backward compatibility and for workflows where a scalar normalization is
preferred.

### Off-Peak Mask For Mean Normalization

```yaml
settings:
  normalization: "mean"
  rocking_curve_offpeak_mask:
    mode: "exclude_reflectivity_peak"
    half_width_deg: 1.25
```

This finds the maximum point in the reflectivity dataset and excludes
rocking-curve points within `+/- half_width_deg` of that angle from the mean
normalization denominator.

What it controls:

- experimental rocking-curve mean normalization;
- simulated/fitted rocking-curve mean normalization;
- the denominator used to put rocking curves on a comparable scale.

What it does not do:

- it does not remove those points from output files;
- it does not remove those points from plots;
- it does not by itself remove those points from the residual vector.

Requirements:

- `datasets.reflectivity` must be present.
- `mode` must currently be `"exclude_reflectivity_peak"`.
- `half_width_deg` must be positive. Default is `1.25`.

## Slicing

Slicing controls how finite layers are converted into effective calculation
cells for roughness and standing-wave XPS calculations.

Default adaptive/unified slicing:

```yaml
settings:
  slicing:
    mode: "adaptive"
    min_slices: 3
    max_slice_thickness_A: 1.0
```

Fixed-grid slicing:

```yaml
settings:
  slicing:
    mode: "fixed_grid"
    min_slices: 3
    max_slice_thickness_A: 1.0
    reference_values:
      carbon_thickness: 16.0
      lno_thickness: 22.0
      sto_thickness: 22.0
```

Allowed forms:

- omitted: adaptive/unified slicing with defaults;
- `"adaptive"` or `"unified"`: adaptive/unified slicing;
- `"legacy"`: legacy fixed-step roughness/depth path;
- mapping with `mode: "adaptive"` or `mode: "unified"`;
- mapping with `mode: "fixed"` or `mode: "fixed_grid"`;
- mapping with `mode: "legacy"` or `mode: "none"` to select the legacy path.

Fields:

- `min_slices`: minimum cells per finite layer. Must be positive.
- `max_slice_thickness_A`: target maximum cell thickness in Angstrom. Must be
  positive.
- `reference_values`: optional parameter values used to build the capacity
  stack for a fixed grid. Keys must be known parameter names.

The auto fixed-grid JAX residual requires:

```yaml
settings:
  slicing:
    mode: "fixed_grid"
```

## `materials`

```yaml
materials:
  C:
    opc_file: "data/OPC/C.dat"
    imfp_file: "data/IMFP/C.ANG"
  LNO:
    opc_file: "data/OPC/LaNiO3.dat"
    imfp_file: "data/IMFP/LNO.ANG"
  STO:
    opc_file: "data/OPC/SrTiO3.dat"
    imfp_file: "data/IMFP/STO.ANG"
```

Rules:

- Material labels must match `stack[*].material`.
- Non-vacuum stack materials require `opc_file`.
- Materials that emit a core level require `imfp_file`.
- `vacuum` does not need a material definition.
- Paths are resolved relative to `project.yaml` unless absolute.
- In starter examples, `LNO` means LaNiO3 and `STO` means SrTiO3.

## `parameters`

Varying parameter:

```yaml
parameters:
  lno_thickness:
    initial: 40.0
    lower: 30.0
    upper: 50.0
    vary: true
```

Constant parameter:

```yaml
parameters:
  repeat_center:
    value: 20.0
    vary: false
```

Rules:

- If `initial`, `lower`, and `upper` are present and `vary` is omitted,
  `vary` defaults to `true`.
- If only `value` is present, `vary` defaults to `false`.
- Varying parameters require `initial`, `lower`, and `upper`.
- Constant parameters require `value` or `initial`.
- `lower < upper`.
- `lower <= initial <= upper`.
- Only `vary: true` parameters become optimizer variables.
- Thickness and roughness values are in Angstrom; units are not repeated per
  parameter.

## `stack`

The stack is ordered from vacuum to substrate.

```yaml
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - id: "film"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "$lno_thickness"
    roughness_A: "$interface_roughness"

  - id: "sto_substrate"
    material: "STO"
    tags: ["substrate"]
    thickness_A: 0.0
    roughness_A: 0.0
```

Fields:

- `id`: stable unique layer identifier.
- `material`: material label or `"vacuum"`.
- `tags`: optional list used by `core_levels[*].emit_from`.
- `thickness_A`: number, parameter reference, or expression. Default `0.0`.
- `roughness_A`: number, parameter reference, or expression. Default `0.0`.

Rules and conventions:

- First layer should be vacuum.
- Final substrate layer has `thickness_A: 0.0`.
- `roughness_A` is the upper-interface roughness of that layer, i.e. the
  interface between layer `j-1` and layer `j`.
- Every expanded concrete layer id must be unique.
- Legacy material-only core-level selection is not used; use layer ids or tags.

## Repeat Blocks

```yaml
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - repeat:
      times: 20
      layers:
        - id: "lno_{repeat_index}"
          material: "LNO"
          tags: ["lno_layers", "oxide_layers"]
          thickness_A: "$lno_thickness"
          roughness_A: "$superlattice_roughness"
        - id: "sto_{repeat_index}"
          material: "STO"
          tags: ["sto_layers", "oxide_layers"]
          thickness_A: "$sto_thickness"
          roughness_A: "$superlattice_roughness"

  - id: "sto_substrate"
    material: "STO"
    tags: ["substrate", "sto_layers"]
    thickness_A: 0.0
    roughness_A: "$substrate_roughness"
```

Repeat rules:

- `repeat.times` must be positive.
- `repeat.layers` must be a list.
- `repeat_index` is 1-based.
- `repeat_index0` is zero-based.
- Layer ids may use `{repeat_index}` and `{layer_index}` formatting.
- Use tags to select repeated layer groups in `core_levels`.

## Expressions

Parameter reference:

```yaml
thickness_A: "$lno_thickness"
```

Arithmetic expression:

```yaml
thickness_A: "period * lno_fraction"
```

Safe function expression:

```yaml
roughness_A: "linear_map(repeat_index0, 0, 19, roughness_top, roughness_bottom)"
```

Allowed variables:

- any parameter name;
- `repeat_index`;
- `repeat_index0`;
- `layer_index`.

Allowed operators:

- `+`
- `-`
- `*`
- `/`
- parentheses

Allowed functions:

- `min(...)`
- `max(...)`
- `sqrt(x)`
- `erf(x)`
- `linear_map(x, x0, x1, y0, y1)`
- `transition_erf(x, start, end, center, width)`

Expressions are parsed with an AST whitelist, not raw Python `eval`. Imports,
attributes, indexing, lambdas, comprehensions, and arbitrary Python calls are
rejected.

## `core_levels`

```yaml
core_levels:
  - name: "La 4d"
    binding_energy_ev: 105.0
    emit_from:
      tags: ["lno_layers"]
    concentration: 1.0
    emission_angle_deg: 0.0
```

Fields:

- `name`: core-level name. Match dataset names for overlays and residual labels.
- `binding_energy_ev`: binding energy in eV.
- `emit_from`: required layer selector.
- `concentration`: scalar concentration assigned to selected emitting layers.
  Default `1.0`.
- `emission_angle_deg`: electron emission angle in degrees. Default `0.0`.
- `vacuum_imfp_from_material`: optional material label used to assign a finite
  vacuum IMFP for legacy workflow parity.

Selectors:

```yaml
emit_from:
  layer_ids: ["lno_1", "lno_2"]
```

```yaml
emit_from:
  tags: ["lno_layers"]
```

```yaml
emit_from:
  all: true
```

Selector rules:

- `emit_from` is required.
- `all: true` cannot be combined with `layer_ids` or `tags`.
- If `all: true` is absent, at least one of `layer_ids` or `tags` is required.
- Unknown tags or layer ids fail validation.
- Selected non-vacuum emitting materials must have IMFP tables.

## `datasets`

No datasets:

```yaml
datasets: {}
```

Reflectivity dataset:

```yaml
datasets:
  reflectivity:
    path: "data/curves/lno_sto_c_synthetic_data.csv"
    name: "Reflectivity"
    angle_column: "angle_deg"
    intensity_column: "reflectivity"
    sigma_column: "reflectivity_sigma"
    weight: 1.0
    log_floor: 1.0e-12
```

Rocking-curve datasets:

```yaml
datasets:
  rocking_curves:
    - path: "data/curves/lno_sto_c_synthetic_data.csv"
      name: "La 4d"
      angle_column: "angle_deg"
      intensity_column: "la4d_rc"
      sigma_column: "la4d_sigma"
      weight: 5.0
```

Rules:

- Dataset paths are resolved relative to `project.yaml` unless absolute.
- `angle_column` defaults to `"angle_deg"`.
- Reflectivity `intensity_column` defaults to `"reflectivity"`.
- Rocking-curve `intensity_column` defaults to `"intensity"`.
- `sigma_column` is optional.
- `weight` is optional and must be non-negative. A zero weight is allowed.
- Reflectivity `log_floor` is optional and must be positive.
- Rocking-curve `name` should match a `core_levels[*].name` value for overlays
  and residual interpretation.
- Rocking-curve `normalization` can override `settings.normalization` per
  dataset. If omitted, `settings.normalization` is used. If set to empty/null,
  experimental data are left as read.
- Prefer omitting per-dataset `normalization` unless one dataset truly needs a
  different rule. Setting it to empty/null is intended only for pre-normalized
  data or advanced compatibility cases; otherwise experimental data can end up
  scaled differently from simulated/fitted curves.

Fitting methods require at least one dataset. `simulate_only` can run with or
without datasets.

## `report`

Modern ProjectSpec files usually keep this empty:

```yaml
report: {}
```

That means no legacy report options are set in this section. Prefer:

```yaml
run:
  outputs:
    plots: true
    identifiability: true
```

Legacy form still accepted:

```yaml
report:
  save_plots: true
  identifiability: true
```

Do not set both legacy and `run.outputs` values differently.

## Optimizer Details

### `simulate_only`

```yaml
run:
  mode: "simulate_only"
```

No optimizer is run. SWANX still writes input, resolved, simulation, fit summary,
and report files. With datasets present, experimental data and residual files
are also written. No `fit/best_parameters.csv` table is written.

### `jax_least_squares`

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual: "auto_fixed_grid"
    max_nfev: 100
    ftol: 1.0e-8
    xtol: 1.0e-8
    gtol: 1.0e-8
    record_history: true
    estimate_covariance: true
```

Auto fixed-grid residual requirements:

- at least one dataset;
- `settings.slicing.mode: "fixed_grid"`;
- fixed stack topology;
- `settings.normalization: "edge_polynomial"` or `"mean"` when rocking curves
  are present.

The auto residual supports ProjectSpec stack expressions handled by the
internal JAX expression evaluator. If a model needs arbitrary Python logic,
external state, or a residual shape that cannot be derived from the YAML stack
and datasets, use the custom factory hook below.

Advanced custom residual hook:

```yaml
run:
  mode: "jax_least_squares"
  optimizer:
    residual_function_factory: "fit_factory:build_residual"
```

Use this only when the model cannot be represented by the ProjectSpec
fixed-grid residual. The factory path is `module:function` and is loaded
relative to the project folder.

### `jax_gradient`

```yaml
run:
  mode: "jax_gradient"
  optimizer:
    value_and_grad_factory: "fit_factory:build_value_and_grad"
    maxiter: 100
    record_history: true
```

This backend requires a custom fixed-shape value-and-gradient factory when
datasets are used.

### `bayesian_optimization`

```yaml
run:
  mode: "bayesian_optimization"
  optimizer:
    n_calls: 40
    n_initial_points: 10
    acquisition_function: "EI"
    random_state: 0
    show_progress: false
```

Fields:

- `n_calls`: total objective evaluations. Default `40`.
- `n_initial_points`: initial random/sampling evaluations. Default `10`.
- `acquisition_function`: acquisition rule passed to the BO backend. `"EI"`
  means Expected Improvement.
- `random_state`: seed for reproducibility.
- `show_progress`: print/display BO progress when supported. Default `false`.

BO is an optional global black-box baseline. It is not the default and is not a
fallback for JAX methods.

## Output Folder Contents

Every run writes the core run folder:

- `report.md`
- `input/project_original.yaml`
- `input/project_resolved.yaml`
- `input/run_metadata.json`
- `resolved/stack_resolved.csv`
- `resolved/materials_resolved.csv`
- `resolved/core_levels_resolved.csv`
- `resolved/parameters_resolved.csv`
- `resolved/datasets_resolved.csv`
- `simulation/reflectivity_simulated.csv`
- `simulation/rocking_curves_simulated.csv`
- `fit/fit_summary.json`

When datasets are present, SWANX also writes:

- `data/reflectivity_experimental.csv`
- `data/rocking_curves_experimental.csv`
- `fit/residuals.csv`

When an optimizer is run and returns fitted parameters, SWANX also writes:

- `fit/best_parameters.csv`
- method-specific files under `optimizer/`

When plots are enabled:

- fitting runs use `plots/fit_overview.png`,
  `plots/reflectivity_fit.png`, and `plots/rocking_curves_fit.png`;
- `simulate_only` runs use `plots/simulation_overview.png`,
  `plots/reflectivity_simulation.png`, and
  `plots/rocking_curves_simulation.png`;
- all plotted runs can include `plots/stack_schematic.png`.

When identifiability is enabled and least-squares diagnostics are available:

- `identifiability_analysis/summary.md`
- `identifiability_analysis/parameter_identifiability.csv`
- `identifiability_analysis/singular_values.csv`
- `identifiability_analysis/weak_modes.csv`
- `identifiability_analysis/strong_correlation_pairs.csv`
- `identifiability_analysis/dataset_sensitivity.csv`

Dataset sensitivity is computed from the final weighted least-squares Jacobian.
It is a weighting/scaling audit signal, not automatic proof that one data type
was physically scaled incorrectly.

## Common Mistakes

**Missing material OPC file**

Every non-vacuum material used in `stack` needs `materials.<name>.opc_file`.

**Missing IMFP for emitting material**

If a core level emits from a non-vacuum material, that material needs
`materials.<name>.imfp_file`.

**Duplicate layer id**

Every expanded concrete layer must have a unique `id`. In repeat blocks, include
`{repeat_index}` when needed.

**Unknown tag in `emit_from`**

Tags must exist on at least one expanded stack layer.

**Missing `emit_from`**

Every core level must explicitly specify `layer_ids`, `tags`, or `all: true`.

**Bad parameter expression**

Expressions can only use numbers, parameter names, `repeat_index`,
`repeat_index0`, `layer_index`, arithmetic operators, parentheses, and the
documented safe functions.

**Using `repeat_index` outside a repeat block**

Outside repeat blocks, `repeat_index` and `repeat_index0` resolve to `0`.

**Switching to auto JAX least-squares without fixed-grid slicing**

Use:

```yaml
settings:
  slicing:
    mode: "fixed_grid"
```

or provide a custom `residual_function_factory`.

**Too few edge points for edge-polynomial normalization**

With `settings.normalization: "edge_polynomial"`, the first and last edge
segments must contain enough points for the polynomial order. For sparse
rocking curves, lower `settings.normalization_polynomial_order` or increase
`settings.normalization_edge_fraction`.

**Accidentally bypassing data normalization**

If a rocking-curve dataset sets `normalization: null` or `normalization: ""`,
SWANX leaves that experimental curve as read. Use that only when the file has
already been normalized with the same convention you want to compare against.

**Relative path assumed from the shell current directory**

Relative paths in ProjectSpec YAML are resolved from the directory containing
`project.yaml`, not from the shell current working directory.
