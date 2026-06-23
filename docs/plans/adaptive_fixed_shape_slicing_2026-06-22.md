# Adaptive unified-grid and fixed-shape slicing plan

> Stage 2 namespace status (2026-06-23): The maintained slicing implementation now lives in `swanx.stack.slicing`; flat `swanx.slicing` and `swxps.slicing` imports are compatibility shims.

> Current status (2026-06-23): Implemented. Unified slicing is now the default high-level path; `slicing=None` selects the legacy fixed-step path. The primary namespace is `swanx`, with `swxps` retained as a compatibility alias.

## Status

Implemented and validated on 2026-06-22. The legacy step-based path remains unchanged; the unified grid is selected explicitly with `slicing=`.

## Goal

Add a user-configurable, layer-aware slicing strategy that:

1. gives thin finite layers enough cells for accurate interface calculations;
2. limits cell thickness in thick layers;
3. uses one grid for roughness, fields, concentration/IMFP sampling, and RC integration;
4. keeps NumPy and JAX calculations physically consistent; and
5. keeps JAX shapes fixed throughout a fit.

All current functions and scalar/per-layer `field_step` and `roughness_step`
inputs remain available. The unified strategy is additive and initially opt-in.

## Confirmed design decisions

### Configurable maximum cell thickness

Users control `max_slice_thickness`, in Angstrom. Its default is 2.0 Angstrom.
The minimum number of cells per positive finite nominal layer is separately
configurable through `min_slices`, initially defaulting to 10.

For thickness `t`, minimum count `N_min`, and maximum cell thickness `dz_max`:

```text
N(t) = max(N_min, ceil(t / dz_max))
dz(t) = t / N(t)
```

Default examples:

- 4 Angstrom: 10 cells of 0.4 Angstrom;
- 16 Angstrom: 10 cells of 1.6 Angstrom;
- 160 Angstrom: 80 cells of 2.0 Angstrom.

`max_slice_thickness` is a maximum, not a minimum. Smaller user values increase
resolution and cost; larger values reduce cost but cannot reduce the count
below `min_slices`.

### One shared cell-centered grid

The new mode uses exactly one grid for roughness, field, concentration, IMFP,
attenuation, and RC calculations. Each finite nominal layer is partitioned into
equal-width cells. The grid stores:

- cell edges;
- cell centers;
- cell widths;
- nominal layer index for each cell; and
- effective optical-layer index for each cell.

No second `ceil(thickness / field_step)` grid is constructed in this mode.

## Physics background

At each cell center, the selected error-function or linear interface profile
determines graded optical constants, concentration, and attenuation
coefficient. One sharp effective optical layer represents each cell. Adjacent
cells with identical optical constants are physically equivalent to one thicker
layer apart from numerical roundoff.

The transfer matrix propagates fields through the effective cell layers. Field
intensity is evaluated at the same cell centers. For cell `j`, width `dz_j`,
and attenuation coefficient `mu_j = 1 / (lambda_j cos(alpha))`, optical depth
to the cell center is:

```text
tau_j = sum(mu_k * dz_k for k < j) + 0.5 * mu_j * dz_j
```

The raw RC intensity uses cell-centered midpoint quadrature:

```text
I_RC = sum(C_j * |E_j|^2 * exp(-tau_j) * dz_j)
```

This makes attenuation and XPS integration use the same cells as roughness and
fields. The existing endpoint/trapezoid integration remains unchanged on the
legacy path; convergence against it at fine resolution is required.

## JAX fixed-shape rule

Adaptive counts from every trial thickness would still cross integer
thresholds and recompile JAX. Fitting therefore uses a fixed-capacity plan.

For capacity thickness `t_cap`:

```text
N_fixed = max(N_min, ceil(t_cap / dz_max))
dz_trial = t_trial / N_fixed
```

Counts and array mappings are created once and reused. Each evaluation updates
only widths, centers, graded properties, and layer thickness arrays. A trial
thickness above its declared capacity is rejected because it would violate the
configured maximum thickness.

The recommended capacity stack is built from fitting upper bounds. Capacity
cannot be inferred reliably from arbitrary `stack_builder` callables, so the
first implementation requires an explicit capacity/reference stack or an
explicit fixed plan.

## Proposed package design

### `src/swxps/slicing.py`

Tentative immutable public types:

```python
LayerSlicingPolicy(
    min_slices: int = 10,
    max_slice_thickness: float = 2.0,
)

FixedLayerGridPlan(
    slice_counts: tuple[int, ...],
    capacity_thicknesses: tuple[float, ...],
)
```

Tentative public factory:

```python
plan = fixed_layer_grid_plan(capacity_stack.optical_layers, policy)
```

An internal or public `LayerGrid` materializes edges, centers, widths, and
mappings for a trial stack. Planning uses Python/NumPy only and does not import
JAX or table databases.

### High-level integration

Add an optional field such as `slicing` or `grid_plan` to
`ReflectivityRequest`, `RockingCurveRequest`, and `FittingProblem`. Existing
constructors remain valid. When absent, current `field_step` and
`roughness_step` behavior is unchanged. When present, the unified grid controls
all discretized forward calculations; the legacy step values are not used.

Use explicit validation and documentation so callers cannot accidentally mix
the two modes. The exact keyword name will be finalized after the pure planner
prototype, without renaming any current keyword.

### Effective optics and fields

Build one effective optical layer per grid cell, using graded `delta` and
`beta` at its center. Preserve vacuum first and the semi-infinite substrate
last. The field kernel evaluates one value at every cell center and uses the
grid's fixed effective-layer mapping.

### Concentration, attenuation, and RCs

Sample concentration and inverse IMFP at the same centers using the same
roughness profile. Add a cell-weighted integration path using widths and the
cell-centered cumulative attenuation formula. Do not change the signature or
behavior of existing `integrate_xps_intensity`; use a new helper or optional
grid-aware path.

### JAX arrays

Pass fixed-length widths, centers, nominal/effective indices, optical values,
concentration, and attenuation arrays into JIT kernels. Trial thickness changes
values but not shapes. NumPy and JAX must consume grids materialized from the
same plan rather than implementing count logic twice.

## Files to create or modify

- new `src/swxps/slicing.py`;
- `src/swxps/fields.py`;
- `src/swxps/xps.py`;
- `src/swxps/simulation.py`;
- `src/swxps/simulation_jax.py`;
- `src/swxps/fitting.py` for optional plan propagation;
- `src/swxps/__init__.py` for selected public exports;
- new `tests/test_slicing.py`;
- focused additions to fields, XPS, simulation, JAX, and fitting tests;
- new `benchmarks/performance/benchmark_slicing_strategy.py`;
- documentation updates after validation.

Migrate one maintained case-study runner only after package-level validation.
Historical and legacy runners do not drive the initial API.

## Implementation steps

1. Record legacy outputs and array shapes for thin, ordinary, and thick stacks.
2. Implement `LayerSlicingPolicy`, fixed-plan construction, validation, and
   grid materialization with no physics integration.
3. Test user changes to `max_slice_thickness` and `min_slices`.
4. Build effective optical cell layers from the unified grid in an opt-in NumPy path.
5. Evaluate fields at the same centers without constructing a second depth grid.
6. Add cell-centered concentration/IMFP sampling, attenuation, and midpoint RC integration.
7. Propagate the optional grid through high-level NumPy requests.
8. Reuse the same plan and mappings in JAX and verify shape stability.
9. Propagate the optional fixed plan through `FittingProblem`.
10. Benchmark compilation, repeated calls, memory, and thickness sweeps.
11. Migrate one maintained case-study runner in a separate reviewable change.

## Tests

### Policy and grid

- Default 4, 16, and 160 Angstrom examples produce 10, 10, and 80 cells.
- User values such as `max_slice_thickness=0.5` and `5.0` change counts correctly.
- Invalid count, thickness, capacity, and topology inputs are rejected.
- Cell widths are positive and sum to each nominal thickness.
- Centers lie inside their cells and nominal/effective mappings are correct.

### Shared-grid physics

- Roughness optical constants, field depths, concentration, and IMFP arrays
  have exactly one entry per cell and share identical centers.
- Cell-centered attenuation matches an analytic constant-IMFP slab.
- Midpoint RC integration converges to a fine legacy trapezoid reference.
- Thin-layer accuracy improves relative to an intentionally coarse legacy grid.
- A sharp stack subdivided into identical cells preserves reflectivity.

### JAX and fitting

- Adaptive NumPy and fixed-plan NumPy agree when counts match.
- NumPy and JAX reflectivity and RCs agree for the same fixed plan.
- Thickness trials spanning multiple 1 Angstrom boundaries retain identical shapes.
- Trials above plan capacity fail clearly rather than silently violating `dz_max`.
- Compile logging/benchmarking shows one initial compile per planned shape.

### Regression

- Existing scalar and sequence step tests remain passing without modification.
- Fresnel, identical-index, Bragg-peak, and reflectivity-bound tests remain passing.
- Full `python -B -m pytest -q -p no:cacheprovider` passes.

## Validation

Implementation is accepted only if:

1. all current APIs and legacy results remain intact;
2. users can change `max_slice_thickness`, defaulting to 2 Angstrom;
3. one grid is demonstrably shared by roughness, fields, attenuation, and RCs;
4. counts respect `min_slices` and the maximum cell thickness;
5. cell widths conserve fitted layer thickness exactly;
6. fixed plans keep JAX shapes constant over declared fitting bounds;
7. NumPy/JAX parity remains within current tolerances; and
8. accuracy, runtime, memory, and compilation behavior are reported for thin
   and 160 Angstrom cases.

## Implementation decisions

- The high-level keyword is `slicing` on reflectivity/rocking-curve requests and `FittingProblem`.
- The first release uses one global user-configurable `max_slice_thickness`.
- `LayerGrid`, `LayerSlicingPolicy`, `FixedLayerGridPlan`, and their factories are public.
- A per-layer maximum can be added later without changing the existing API if a maintained case study demonstrates the need.

## Progress log

- 2026-06-22: Confirmed that both roughness and field grid lengths currently
  depend on trial thickness through `ceil`.
- 2026-06-22: Confirmed user-configurable maximum slice thickness with a 2
  Angstrom default.
- 2026-06-22: Confirmed one shared cell-centered grid for roughness, fields,
  concentration/IMFP, attenuation, and RC integration.
- Implemented policy, adaptive/fixed plans, shared grid, graded optical cells, cell-centered fields, attenuation, midpoint RC integration, NumPy/JAX dispatch, and fitting propagation.
- Focused unified-grid suite: 22 passed.
- Full regression suite: 113 passed with 46 pre-existing `np.trapz` warnings.
- Thin-surface benchmark: one shape across a 2-6 Angstrom sweep, one new JAX compilation, maximum reflectivity absolute error `1.31e-5` versus a 0.1 Angstrom reference, and finite normalized RCs.
- The 160 Angstrom film case used 80 film cells (90 total with the surface layer), returned finite reflectivity, and took about 0.0032 seconds on this machine.
