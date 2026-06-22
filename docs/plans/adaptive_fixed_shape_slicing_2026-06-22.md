# Adaptive and fixed-shape layer slicing plan

## Status

Design only. No package code or public API behavior has been changed.

## Goal

Replace reliance on a globally fixed 1 Angstrom discretization with a reusable
layer-aware strategy that:

1. gives thin finite layers enough samples for accurate interface and field calculations;
2. prevents thick layers from being represented by slices that are too coarse;
3. keeps NumPy and JAX calculations physically consistent; and
4. keeps JAX shapes fixed throughout a fit so thickness changes do not trigger compilation.

All current functions, request classes, and scalar or per-layer `field_step`
and `roughness_step` inputs must remain available. The new strategy must be
additive and opt-in until parity and convergence are demonstrated.

## Terminology decision needed before implementation

The proposed 2 Angstrom criterion is interpreted as a **maximum permitted slice
thickness**, not a lower limit. This is required for the 160 Angstrom example:
ten slices would be 16 Angstrom thick, while a 2 Angstrom maximum requires 80
slices and protects resolution.

For finite-layer thickness `t`, minimum count `N_min`, and maximum slice
thickness `dz_max`:

```text
N(t) = max(N_min, ceil(t / dz_max))
dz(t) = t / N(t)
```

With `N_min = 10` and `dz_max = 2 Angstrom`:

- 4 Angstrom layer: 10 slices of 0.4 Angstrom;
- 16 Angstrom layer: 10 slices of 1.6 Angstrom;
- 160 Angstrom layer: 80 slices of 2.0 Angstrom.

If 2 Angstrom was intended as a minimum slice thickness, the two rules conflict
for layers thinner than `N_min * dz_min` and would cap rather than protect thick
layer resolution. Do not implement that alternative without a revised rule.

## Physics background

The roughness model replaces nominal finite layers with graded,
sharp-interface effective slices. Optical constants at slice centers follow
the selected error-function or linear profile. Effective slice thicknesses for
each nominal layer must sum exactly to that nominal thickness.

Electric fields and XPS attenuation are evaluated on a depth grid. The current
code contains two independent thickness-dependent shapes:

1. `effective_layers_with_roughness` uses approximately
   `ceil(t / roughness_step)` effective slices per finite layer;
2. `depth_grid` uses approximately `ceil(t / field_step) + 1` points per
   effective layer.

Changing a fitted thickness can therefore change both the effective-layer
length and depth-grid length. JAX compiles on shape. Merely applying the new
adaptive formula to each trial thickness would still cross integer thresholds
and trigger recompilation, so fitting needs fixed counts determined before the
first objective evaluation.

## Proposed design

### Pure planning objects

Add a backend-independent module such as `src/swxps/slicing.py` with tentative
immutable types:

- `LayerSlicingPolicy(min_slices=10, max_slice_thickness=2.0)`;
- `FixedLayerSlicePlan(slice_counts=...)`.

Names remain provisional. The planner uses Python and NumPy only, validates
positive finite values, and stores one count per nominal finite layer. It must
not depend on JAX or optical-constant databases.

### Adaptive NumPy mode

For one-off NumPy simulations, calculate counts from actual thickness using
`N(t)`. Thin layers receive at least ten slices and thick layers respect the
maximum width.

### Fixed-capacity fitting/JAX mode

For a fit, calculate counts once from capacity thickness `t_cap`:

```text
N_fixed = max(N_min, ceil(t_cap / dz_max))
dz_trial = t_trial / N_fixed
```

Reuse `N_fixed` for every evaluation. Thickness remains continuous because only
`dz_trial` changes, while all effective-layer and depth-grid shapes stay fixed.

The recommended capacity is the maximum thickness allowed by fitting bounds.
It cannot always be inferred from `FittingProblem` because `stack_builder` is
an arbitrary callable and one parameter may control several layers. Initially
require either a capacity/reference stack built at upper bounds or explicit
per-layer counts. Do not silently use only the initial stack.

### Shared nominal-layer plan

Roughness slicing and field/XPS depth sampling must use one nominal-layer plan,
or two explicitly related fixed plans. Do not independently resample every
roughness slice with a trial-dependent `ceil`, which would restore shape changes.

Preserve mappings among nominal material layer, effective optical slice,
sampled depth, concentration, and IMFP. Decide after a small prototype whether
field sampling can use exactly the planned intervals or needs a second fixed
resolution.

### Backward-compatible integration

Do not remove, rename, or reinterpret existing inputs:

- scalar and per-layer `roughness_step` retain legacy behavior;
- scalar `field_step` retains legacy behavior;
- existing `ReflectivityRequest`, `RockingCurveRequest`, and `FittingProblem`
  calls remain valid;
- NumPy remains the default backend.

Add the policy/plan through optional keyword fields or an additional accepted
input type. Choose the exact public form only after the prototype shows that a
single plan can serve reflectivity, fields, and XPS without duplicated grids.
Existing examples need not be rewritten in the first milestone.

## Files to create or modify

- new `src/swxps/slicing.py` or equivalent;
- `src/swxps/fields.py`;
- `src/swxps/simulation.py`;
- `src/swxps/simulation_jax.py`;
- `src/swxps/fitting.py` if high-level propagation is needed;
- `src/swxps/__init__.py` for selected public exports;
- new `tests/test_slicing.py`;
- focused additions to `tests/test_fields.py`, `tests/test_simulation.py`,
  `tests/test_simulation_jax.py`, and possibly `tests/test_fitting.py`;
- new `benchmarks/performance/benchmark_slicing_strategy.py`;
- documentation updates after API validation.

Migrate maintained case-study runners only after package-level validation.
Historical and legacy runners should not drive the first API design.

## Implementation steps

1. Record legacy NumPy/JAX results and shapes for thin, ordinary, and thick stacks.
2. Implement and test the pure count planner without physics-kernel integration.
3. Add optional fixed-count edge generation to roughness discretization while
   preserving the current `step` path exactly.
4. Add a fixed-shape depth-grid path without a trial-dependent `ceil` on
   effective slices.
5. Integrate the optional plan into NumPy requests and verify convergence
   against a fine legacy reference.
6. Reuse the identical plan in JAX; verify every allowed trial thickness gives
   identical effective-layer and depth-grid shapes.
7. Add optional propagation through `FittingProblem` without changing existing calls.
8. Benchmark initial compilation, repeated calls, and a thickness sweep crossing
   multiple 1 Angstrom boundaries.
9. Migrate one maintained case-study runner as a separate step.
10. Update only examples where the strategy materially improves the lesson.

## Tests

### Planner

- 4, 16, and 160 Angstrom layers produce 10, 10, and 80 slices with proposed defaults.
- Counts are positive integers and invalid policy values are rejected.
- Fixed plans reject changed topology or a wrong count length.
- Reconstructed thickness equals each nominal thickness within tolerance.

### Physics and parity

- A thin layer agrees with a fine legacy reference more closely than a coarse grid.
- Adaptive NumPy and fixed-plan NumPy agree when counts match.
- NumPy and JAX reflectivity and rocking curves agree for the same fixed plan.
- Existing scalar and sequence step tests remain unchanged and passing.
- Fresnel, identical-index, Bragg-peak, and reflectivity-bound tests remain passing.
- Concentration and attenuation remain aligned with nominal materials.

### Shape and compilation

- Trials spanning more than 1 Angstrom retain identical effective-layer and
  depth-grid shapes under one fixed plan.
- A thick-layer sweep retains shape while slice widths change continuously.
- Check JAX behavior with a benchmark or compile logging; tests assert stable
  shapes rather than depend on private JAX cache APIs.

### Regression

- Full `python -B -m pytest -q -p no:cacheprovider` passes.
- Legacy calls with no new policy reproduce previous numerical results.

## Validation

Implementation is acceptable only if:

1. existing APIs and legacy discretization behavior remain intact;
2. thin layers receive at least the configured minimum count;
3. adaptive slices do not exceed the configured maximum thickness;
4. fixed-capacity plans keep shapes constant over the declared fit range;
5. slice thicknesses sum exactly to fitted layer thickness;
6. NumPy/JAX parity stays within current tolerances;
7. a sweep shows one initial JAX compile per planned shape rather than a compile
   at every crossed step boundary; and
8. runtime and memory costs for a 160 Angstrom layer are reported.

## Decisions required before coding

1. Confirm that 2 Angstrom means **maximum** slice thickness.
2. Confirm defaults: `min_slices=10`, `max_slice_thickness=2.0` Angstrom.
3. Confirm legacy step behavior remains the initial default and the new mode is opt-in.
4. Confirm that fitting runners provide a capacity stack built at upper bounds
   rather than manually maintained counts where possible.
5. Decide whether roughness and field sampling use one grid or two related fixed
   resolutions after the prototype.

## Progress log

- 2026-06-22: Inspected NumPy and JAX paths. Both roughness slice count and
  field depth-grid count depend on trial thickness through `ceil`, so both must
  be addressed to prevent recompilation.
- 2026-06-22: Created this design plan. No source code changed.
