# ProjectSpec YAML Reference

This reference describes the human-editable SWANX `project.yaml` format. A
ProjectSpec is a plain YAML file that resolves materials, stack layers, core
levels, datasets, and report settings before calling existing SWANX IO,
simulation, and fitting APIs.

## Top-Level Structure

```yaml
project:
settings:
materials:
parameters:
stack:
core_levels:
datasets:
report:
```

Required sections are `project`, `settings`, `materials`, `stack`, and
`core_levels`. The `parameters`, `datasets`, and `report` sections may be empty
mappings.

## `project`

```yaml
project:
  name: "my_project"
  output_dir: "optional_relative_or_absolute_path"
```

- `name` is used in the default output folder name.
- Relative `output_dir` values are resolved relative to `project.yaml`.
- Absolute `output_dir` values are respected.
- If `output_dir` is absent, SWANX writes to:

```text
project_yaml_dir/runs/<project_name>_<timestamp>/
```

## `settings`

A common simulation setup is:

```yaml
settings:
  photon_energy_ev: 1000.0
  angle_start_deg: 6.9
  angle_stop_deg: 10.9
  angle_count: 161
  polarization: "p"
  normalization: "mean"
  fit_method: "simulate_only"
```

You can also provide explicit angles:

```yaml
settings:
  photon_energy_ev: 1000.0
  angles_deg: [6.9, 7.0, 7.1]
  polarization: "s"
  normalization: "mean"
  fit_method: "simulate_only"
```

Fields:

- `photon_energy_ev`: photon energy in eV.
- `angle_start_deg`, `angle_stop_deg`, `angle_count`: generated incident-angle
  grid in degrees.
- `angles_deg`: explicit incident-angle list in degrees.
- `polarization`: `"s"`, `"p"`, or `"unpolarized"`.
- `"unpolarized"` maps to a 50/50 s/p mixture.
- Circular polarization is not implemented.
- `normalization`: default rocking-curve normalization mode, commonly `"mean"`.
- `fit_method`: `"simulate_only"`, `"jax_least_squares"`, `"jax_gradient"`, or
  `"bayesian_optimization"`.

JAX methods require callback factories when datasets are used. Bayesian
optimization is an optional global black-box baseline, not a default and not a
fallback.

Optional fixed-shape or unified-grid controls can be set with `slicing`:

```yaml
settings:
  slicing:
    mode: "fixed_grid"
    min_slices: 3
    max_slice_thickness_A: 1.0
    reference_values:
      lno_thickness: 40.0
```

Optional rocking-curve masking can exclude the reflectivity peak region from RC
scoring:

```yaml
settings:
  rocking_curve_offpeak_mask:
    mode: "exclude_reflectivity_peak"
    half_width_deg: 1.25
```

## `materials`

```yaml
materials:
  LNO:
    opc_file: "data/OPC/LaNiO3.dat"
    imfp_file: "data/IMFP/LNO.ANG"
  STO:
    opc_file: "data/OPC/SrTiO3.dat"
    imfp_file: "data/IMFP/STO.ANG"
```

Rules:

- Material labels must match `stack[*].material` values.
- Non-vacuum stack materials require `opc_file`.
- Materials used by emitting layers require `imfp_file`.
- `vacuum` does not need a material definition.
- Paths are resolved relative to `project.yaml` unless absolute.

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

- Only `vary: true` parameters become fitting parameters.
- If `initial`, `lower`, and `upper` are present and `vary` is omitted, `vary`
  defaults to `true`.
- If only `value` is present, `vary` defaults to `false`.
- Thickness and roughness values are in Angstrom; do not add per-parameter unit
  fields.
- Bounds must satisfy:

$$lower \le initial \le upper$$

and `lower < upper`.

## `stack`

Simple stack:

```yaml
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - id: "lno_1"
    material: "LNO"
    tags: ["lno_layers"]
    thickness_A: "$lno_thickness"
    roughness_A: "$interface_roughness"

  - id: "sto_substrate"
    material: "STO"
    thickness_A: 0.0
    roughness_A: 0.0
```

Rules:

- The first layer should be vacuum.
- The substrate is last and can use `thickness_A: 0.0`.
- Every concrete layer needs a stable unique `id`.
- `tags` are optional but recommended for selecting emitting layers.
- `roughness_A` is the layer's upper-interface roughness: layer j roughness is
  the interface between layer j-1 and layer j.
- Legacy material-only emitting-layer selection is not used.

Inline parameter reference:

```yaml
thickness_A: "$lno_thickness"
```

Safe arithmetic expression:

```yaml
thickness_A: "A_LNO + B_LNO * repeat_index"
```

Expression variables are parameter names, `repeat_index`, and `layer_index`.
Expressions use an AST whitelist, not raw `eval`. Allowed operators are `+`,
`-`, `*`, `/`, and parentheses.

## Repeat Block

```yaml
stack:
  - id: "vacuum"
    material: "vacuum"
    thickness_A: 0.0
    roughness_A: 0.0

  - repeat:
      times: 40
      layers:
        - id: "lno_{repeat_index}"
          material: "LNO"
          tags: ["lno_layers"]
          thickness_A: "$lno_thickness"
          roughness_A: "$interface_roughness"

        - id: "sto_{repeat_index}"
          material: "STO"
          tags: ["sto_layers"]
          thickness_A: "$sto_thickness"
          roughness_A: "$interface_roughness"

  - id: "sto_substrate"
    material: "STO"
    thickness_A: 0.0
    roughness_A: 0.0
```

Rules:

- `repeat_index` is 1-based.
- Generated IDs must be unique.
- Use tags for all repeated layer groups so core levels can select layer groups.

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

Rules:

- `emit_from` is required.
- `all: true` cannot be combined with `tags` or `layer_ids`.
- Unknown tags or layer IDs fail validation.
- `binding_energy_ev` is used to compute kinetic energy:

$$E_\mathrm{kin} = h\nu - E_B$$

- Every emitting material needs an IMFP table.

## `datasets`

No datasets:

```yaml
datasets: {}
```

Reflectivity dataset:

```yaml
datasets:
  reflectivity:
    path: "../../benchmarks/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv"
    name: "Reflectivity"
    angle_column: "angle_deg"
    intensity_column: "reflectivity"
    sigma_column: "sigma"
    weight: 1.0
    log_floor: 1.0e-12
```

Rocking-curve datasets:

```yaml
datasets:
  rocking_curves:
    - path: "../../benchmarks/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv"
      name: "La 4d"
      angle_column: "angle_deg"
      intensity_column: "la4d_rc"
      sigma_column: "sigma"
      normalization: "mean"
      weight: 1.0
```

Rules:

- Dataset paths are resolved relative to `project.yaml`.
- A rocking-curve `name` should match a core-level name for overlay and residual
  comparison.
- `sigma_column` is optional.
- Normalization can be set globally in `settings.normalization` or per rocking
  curve.
- Weights must be non-negative.
- Reflectivity `log_floor` must be positive.

## `report`

```yaml
report:
  save_plots: true
```

When matplotlib is available and `save_plots: true`, common plots are:

- `plots/fit_overview.png`
- `plots/reflectivity_fit.png`
- `plots/rocking_curves_fit.png`
- `plots/stack_schematic.png`
- least-squares diagnostic plots when available;
- Bayesian optimization convergence/surrogate plots when available.

Skipped plot reasons are written to `report.md`.

## Fitting Settings

JAX least-squares placeholder:

```yaml
settings:
  fit_method: "jax_least_squares"
  optimizer:
    residual_function_factory: "fit_factory:build_residual"
    max_nfev: 100
    estimate_covariance: true
```

The factory module can live next to `project.yaml`. ProjectSpec does not
auto-generate JAX residual functions, and SWANX does not fall back to BO when the
factory is missing.

BO optional baseline:

```yaml
settings:
  fit_method: "bayesian_optimization"
  optimizer:
    n_calls: 40
    n_initial_points: 10
    random_state: 0
```

BO requires the scikit-optimize extra (`.[fit]`). BO is an optional global black-box baseline / robustness check. BO reports do not write least-squares
covariance or correlation files.

## Common Mistakes

**Missing material OPC file**

Every non-vacuum material used in `stack` needs `materials.<name>.opc_file`.

**Missing IMFP for emitting material**

If a core level emits from a material, that material needs
`materials.<name>.imfp_file`.

**Duplicate layer ID**

Every concrete layer must have a unique `id`. In repeat blocks, include
`{repeat_index}`.

**Unknown tag in `emit_from`**

Tags must exist on at least one expanded layer.

**Missing `emit_from`**

Every core level must explicitly specify `layer_ids`, `tags`, or `all: true`.

**Bad parameter expression**

Expressions can only use numbers, parameter names, `repeat_index`,
`layer_index`, arithmetic operators, and parentheses.

**Using `repeat_index` outside a repeat block**

`repeat_index` is available only inside repeated layers. Use a constant or
parameter outside repeat blocks.

**Switching to `jax_least_squares` without a factory**

Add `settings.optimizer.residual_function_factory` or use `simulate_only` while
preparing the project. There is no BO fallback.

**Dataset rocking-curve name not matching a core level**

For overlays and residuals, the dataset `name` should match `core_levels[*].name`.

**Relative path assumed from CWD instead of `project.yaml`**

Relative paths are resolved from the directory containing `project.yaml`, not
from the shell current working directory.
