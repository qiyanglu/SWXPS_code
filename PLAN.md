# PLAN.md

## Goal

Wrap the validated reflectivity forward problem in one differentiable JAX function placed in a separate `swxps` module.

## Physics background

### Forward problem

The forward problem maps a fully specified physical model to predicted observables.

For this project, the currently validated forward core is specular x-ray reflectivity from a multilayer stack. Inputs are grazing-incidence angle, photon energy, layer thicknesses, interface roughnesses, and complex refractive-index parameters `n = 1 - delta + i beta`. The output is reflectivity `R = |r|^2`.

In compact form:

```text
angles, energy, thickness, roughness, delta, beta -> reflectivity
```

The existing NumPy implementation is the reference implementation. The new JAX function should reproduce the same Parratt recursion behavior while exposing gradients with respect to continuous real-valued stack parameters.

### Inverse problem

The inverse problem maps measured observables back to unknown model parameters.

For this project, measured reflectivity or SW-XPS rocking curves could be used to infer thicknesses, roughnesses, angle offset, optical constants, composition-profile parameters, IMFP-related parameters, and nuisance terms such as scale or background. This is normally done by minimizing a loss or maximizing a likelihood comparing measured curves to forward-model predictions.

The repository does not yet implement inverse fitting or optimization. This task only adds a differentiable forward function that future inverse methods can use.

## Differentiable JAX Forward Function

Create `src/swxps/jax_forward.py` with one public forward wrapper:

```python
reflectivity_forward_jax(angles_deg, energy_ev, thickness, delta, beta, roughness)
```

Design choices:

1. Use `jax.numpy` throughout the differentiable calculation.
2. Accept arrays instead of `Layer` dataclasses so JAX can trace numeric inputs directly.
3. Keep all optimizable parameters real-valued; construct complex refractive indices internally.
4. Use the same Parratt recursion equations as `src/swxps/reflectivity.py`.
5. Vectorize over input angles so the function returns one reflectivity value per angle.
6. Keep validation checks minimal because JAX-traced values cannot use ordinary Python value checks safely inside transformed functions.
7. Export the function from `swxps.__init__` for convenient use.
8. Add tests comparing JAX output with the NumPy Parratt reference.
9. Add an autograd test proving `jax.grad` returns finite gradients for a scalar loss.


## JIT Refactor Plan

The inverse problem will repeatedly evaluate an objective function and its gradient through `reflectivity_forward_jax(...)`. To support that use case, the JAX forward function should be directly compatible with `jax.jit` and efficient under tracing.

Planned refactor:

1. Decorate `reflectivity_forward_jax(...)` with `jax.jit` so callers can use it directly in objective functions.
2. Remove the scalar-angle `if` branch by flattening input angles and reshaping the output back to the original angle shape.
3. Replace the Python reverse loop over interfaces with `jax.lax.scan`.
4. Precompute all interface Fresnel amplitudes, roughness factors, and phase factors as arrays before the scan.
5. Keep vectorization over all angles in the array dimensions; do not loop over angles.
6. Add tests proving the function works through explicit `jax.jit` and that gradients through a JIT-compiled objective are finite.

## Files to create or modify

- `PLAN.md`
- `src/swxps/jax_forward.py`
- `src/swxps/__init__.py`
- `tests/test_jax_forward.py`

## Implementation steps

1. Update this plan before code changes.
2. Implement the JAX Parratt-style forward reflectivity function in a new file.
3. Export the new public function from the package root.
4. Add tests for NumPy/JAX numerical agreement.
5. Add a gradient smoke test for thickness, delta, beta, and roughness sensitivity.
6. Run the targeted JAX tests and the full test suite.
7. Refactor the JAX forward function for repeated JIT objective/gradient evaluation using `jax.lax.scan`.

## Tests

- `python -m pytest tests/test_jax_forward.py`
- `python -m pytest`

Expected tests:

- Two-layer JAX reflectivity matches NumPy Fresnel/Parratt reflectivity.
- Multilayer JAX reflectivity matches NumPy Parratt reflectivity.
- Identical refractive indices give near-zero reflectivity.
- A scalar loss differentiated with `jax.grad` returns finite gradients.
- Explicit `jax.jit` evaluation returns the same reflectivity as the reference.
- A JIT-compiled scalar objective differentiated with `jax.grad` returns finite gradients.

## Validation

The JAX forward function is valid for this stage if it reproduces the existing NumPy Parratt reflectivity within numerical tolerance and supports finite autograd gradients for continuous real-valued parameters.

## Progress log

- Planned a separate JAX reflectivity forward module before implementation.
- Implemented `reflectivity_forward_jax(...)` in `src/swxps/jax_forward.py`.
- Exported the function from `swxps`.
- Added JAX forward tests for NumPy agreement, identical-index behavior, and finite autograd gradients.
- Ran `python -m pytest tests/test_jax_forward.py -q`: 4 passed.
- Ran `python -m pytest`: 40 passed.
- Planned JIT refactor for inverse-problem objective and gradient use.
- Refactored `reflectivity_forward_jax(...)` with `@jax.jit`, vectorized interface precomputation, `jax.lax.scan` for reverse Parratt recursion, and shape restoration without a scalar branch.
- Added tests for explicit JIT use, gradients through a JIT-compiled objective, and scalar-angle shape preservation.
- Ran `python -m pytest tests/test_jax_forward.py -q`: 7 passed.
- Ran `python -m pytest`: 43 passed.
