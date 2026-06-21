# X-Ray Reflectivity Milestone Plan

## Goal

Implement a small Python package for simulating specular x-ray reflectivity from multilayer thin films using the Parratt recursion.

The first milestone will support:

- Python with numpy only.
- Grazing incidence angle in degrees.
- Photon energy in eV.
- Wavelength in Angstrom.
- Layer thickness in Angstrom.
- Manually supplied complex refractive indices:

  n = 1 - delta + i beta

- Vacuum as the first layer.
- A semi-infinite substrate as the last layer.
- s-polarization only.

This first milestone did not implement fitting, XPS intensities, p-polarization, or online optical-constant databases.

A later fitting milestone will add Bayesian optimization for experimental reflectivity and SW-XPS rocking-curve data. The first optimizer backend will use `scikit-optimize`, but the fitting layer should be written so that the Bayesian-optimization backend can be replaced later without changing the forward simulation API.

## Physics background

For photon energy E in eV, the x-ray wavelength is:

lambda = hc / E

using:

hc = 12398.419843320026 eV Angstrom

The incident angle theta is the grazing angle relative to the sample surface. For a layer j with complex refractive index n_j, define the z component of the wavevector as:

kz_j = k0 sqrt(n_j^2 - cos(theta)^2)

where:

k0 = 2 pi / lambda

and theta is converted from degrees to radians.

For s-polarization, the Fresnel reflection amplitude between layers j and j + 1 is:

r_j,j+1 = (kz_j - kz_j+1) / (kz_j + kz_j+1)

Interface roughness is included with the Nevot-Croce correction:

r_j,j+1,rough = r_j,j+1 exp(-2 kz_j kz_j+1 sigma_j,j+1^2)

where sigma_j,j+1 is the RMS roughness of the interface in Angstrom. In the code, each `Layer` stores the roughness of its upper interface. Therefore, the roughness of the interface between layer j and j + 1 is stored on layer j + 1. The vacuum layer roughness is unused.

The Parratt recursion starts from the bottom semi-infinite substrate, where the reflected amplitude is zero:

R_N = 0

Then it proceeds upward:

R_j = (r_j,j+1 + R_j+1 exp(2 i kz_j+1 d_j+1)) /
      (1 + r_j,j+1 R_j+1 exp(2 i kz_j+1 d_j+1))

Here d_j+1 is the thickness of the layer below the current interface. The first layer, vacuum, and the last layer, substrate, are semi-infinite and their thickness values should not affect the result.

The measured reflectivity is:

reflectivity = |R_0|^2

For standing-wave calculations, the local s-polarized electric field in layer j is written as the sum of downward and upward waves:

E_j(z) = A_j exp(i kz_j z) + B_j exp(-i kz_j z)

where z is the local depth measured downward from the top of layer j. The field intensity used for visualization is:

I_j(z) = |E_j(z)|^2

The forward and backward amplitudes at each layer top are propagated from the vacuum surface using the Parratt reflection amplitudes R_j. Starting with:

A_0 = 1
B_0 = R_0

the field at the bottom of layer j is:

E_j(d_j) = A_j exp(i kz_j d_j) + B_j exp(-i kz_j d_j)

and the next layer amplitude is set by tangential electric-field continuity:

A_j+1 = E_j(d_j) / (1 + R_j+1)
B_j+1 = R_j+1 A_j+1

This provides a transparent first implementation of the depth-dependent field profile. Roughness still affects the Parratt reflection amplitudes through the same Nevot-Croce correction used for reflectivity.

The field-profile implementation must be checked against an independent transfer-matrix calculation. For sharp interfaces, propagation through a layer and matching at each interface can be written with 2 by 2 matrices. At interface j to j + 1:

A_j + B_j = A_j+1 + B_j+1

kz_j (A_j - B_j) = kz_j+1 (A_j+1 - B_j+1)

The transfer-matrix solution uses the boundary conditions A_0 = 1 and B_N = 0. The resulting reflectivity is:

R = |B_0 / A_0|^2

For zero roughness, this must match the Parratt reflectivity. This comparison is required before trusting field profiles quantitatively.

Roughness in transfer-matrix calculations will be represented by replacing each rough interface with a set of thin effective layers. The optical constants across an interface are interpolated with an error-function profile:

f(z) = 0.5 [1 + erf((z - z_i) / (sqrt(2) sigma_i))]

delta(z) = [1 - f(z)] delta_above + f(z) delta_below

beta(z) = [1 - f(z)] beta_above + f(z) beta_below

where z_i is the nominal interface depth and sigma_i is the RMS roughness. The resulting discretized graded stack has only sharp interfaces, so it can be propagated by the transfer-matrix solver. This is an approximate roughness model and should be checked for convergence with the slicing step.

The first standing-wave XPS rocking-curve calculation will simulate normalized core-level intensity versus incidence angle. Cross sections are treated as constants, so the calculated intensity is:

I(theta) = integral C(z) |E(z, theta)|^2 A(z) dz

where C(z) is the emitting-species concentration profile and A(z) is the electron attenuation factor. With a depth-dependent electron IMFP lambda_e(z) and emission angle alpha relative to the surface normal:

A(z) = exp[- integral_0^z dz' / (lambda_e(z') cos(alpha))]

For normalized rocking curves:

I_norm(theta) = I(theta) / mean(I(theta_off_peak))

For the first La 4d example, the kinetic energy is:

E_kin = h nu - E_B = 1000 eV - 105 eV = 895 eV

The La concentration is assigned to LaNiO3 layers and not SrTiO3 layers. IMFP values are read from tabulated files in `IMFP`.

For a periodic multilayer with bilayer period d, constructive interference should appear near the Bragg condition:

m lambda = 2 d sin(theta)

for integer order m.

For fitting experimental data, the forward model should remain separate from the optimizer. A trial parameter vector x is mapped to physical stack and instrument parameters, then passed to the existing simulation API. The optimizer only sees a scalar objective:

F(x) = sum_j w_j chi_j^2(x) + P(x)

where j runs over datasets such as reflectivity and one or more SW-XPS rocking curves, w_j are user-selected dataset weights, and P(x) is an optional penalty for invalid or discouraged parameter combinations. Reflectivity residuals should support logarithmic comparison because the signal spans many orders of magnitude:

chi_R^2 = mean([log10(R_exp + floor) - log10(R_sim + floor)]^2)

Normalized SW-XPS rocking-curve residuals can initially use weighted mean-squared differences:

chi_RC,j^2 = mean([(I_exp,j - I_sim,j) / sigma_j]^2)

if experimental uncertainties are available, or unweighted mean-squared differences otherwise. Parameters should be physically bounded. For LNO/STO superlattice fitting, an initial small parameter set may include common LNO thickness, common STO thickness, internal superlattice roughness, substrate/interface roughness, top surface roughness, and incidence-angle offset.

The initial Bayesian optimizer will use Gaussian-process Bayesian optimization through `scikit-optimize`. The fitting code must isolate optimizer-specific concepts such as search dimensions, acquisition function, random initialization, and optimization state from reusable concepts such as parameter definitions, stack construction, objective evaluation, dataset weighting, and result records.

## Files to create or modify

Expected package layout:

- `pyproject.toml`
  - Minimal package metadata.
  - Test configuration if useful.
- `src/swxps/__init__.py`
  - Public package exports.
- `src/swxps/constants.py`
  - Physical constants such as hc in eV Angstrom.
- `src/swxps/layers.py`
  - A small `Layer` data structure.
  - Helpers for constructing complex refractive index from delta and beta.
  - RMS upper-interface roughness for each layer.
- `src/swxps/reflectivity.py`
  - Energy-to-wavelength conversion.
  - kz calculation.
  - s-polarized Fresnel amplitude.
  - Nevot-Croce roughness correction.
  - Parratt reflectivity calculation.
- `src/swxps/fields.py`
  - Parratt reflection amplitudes at each layer top.
  - Forward and backward field amplitudes.
  - Sharp-interface transfer-matrix field amplitudes.
  - Rough-interface transfer-matrix approximation using graded effective layers.
  - Transfer-matrix reflectivity for comparison against Parratt.
  - Depth-grid generation through finite layers.
  - Electric field and field intensity as a function of depth.
- `src/swxps/optical_constants.py`
  - Reader for Henke-style optical-constant tables stored in `OPC`.
  - Linear interpolation of delta and beta at requested photon energy.
- `src/swxps/imfp.py`
  - Reader for tabulated IMFP files stored in `IMFP`.
  - Linear interpolation of IMFP at requested electron kinetic energy.
- `src/swxps/xps.py`
  - Depth-dependent concentration and attenuation helpers.
  - Normalized standing-wave XPS rocking-curve simulation with constant cross sections.
- `src/swxps/simulation.py`
  - Clean high-level input and output dataclasses for reflectivity and SW-XPS RC simulations.
  - Angle-offset handling for future fitting workflows.
  - Material-labeled layer stacks and core-level definitions.
- `src/swxps/profiles.py`
  - Sample material or element concentration profiles versus depth.
  - Use the same error-function roughness grading as the XPS model.
- `src/swxps/fitting.py`
  - Optimizer-independent fitting dataclasses and objective helpers.
  - Parameter definitions with names, bounds, initial values if supplied, and physical units.
  - Mapping from trial vectors to simulation requests.
  - Residual/objective functions for reflectivity and normalized SW-XPS rocking curves.
  - Fit history records that do not depend on a specific optimizer package.
- `src/swxps/optimizers.py` or `src/swxps/bo.py`
  - First Bayesian-optimization backend using `scikit-optimize`.
  - Adapter from generic fitting parameter definitions to `skopt.space` dimensions.
  - Optimizer result conversion back to package-native result dataclasses.
  - Keep this module thin so another optimizer can replace `scikit-optimize` later.
- `tests/test_reflectivity.py`
  - Unit tests for the required physical checks.
- `tests/test_optical_constants.py`
  - Unit tests for table parsing, interpolation, exact lookup, and out-of-range handling.
- `tests/test_fields.py`
  - Unit tests for depth-grid construction and simple field-profile limits.
- `tests/test_imfp.py`
  - Unit tests for IMFP parsing and interpolation.
- `tests/test_xps.py`
  - Unit tests for attenuation and normalized rocking-curve behavior.
- `tests/test_simulation.py`
  - Unit tests for the high-level simulation API, angle offsets, and multi-core-level RC outputs.
- `tests/test_profiles.py`
  - Unit tests for stack profile sampling and rough-interface smoothing.
- `tests/test_fitting.py`
  - Unit tests for parameter-vector mapping, objective calculation, and fit history records.
- `tests/test_bo.py`
  - Unit tests for the `scikit-optimize` adapter using a cheap synthetic objective or monkeypatched simulation.
- `examples/reflectivity/plot_lno_sto_reflectivity.py`
  - Example visualization for a LaNiO3/SrTiO3 multilayer using manually supplied optical constants.
- `examples/fields/plot_lno_sto_field_profile.py`
  - Example visualization of standing-wave field intensity through a LaNiO3/SrTiO3 multilayer.
- `examples/xps/plot_lno_la4d_rocking_curve.py`
  - Example normalized La 4d, O 1s, and Ti 2p rocking curves for the LaNiO3/SrTiO3 multilayer.
- `examples/profiles/plot_lno_sto_stack_profile.py`
  - Example visualization of La, Ti, and O concentration profiles versus depth.
- `examples/README.md`
  - Short instructions for running the examples and interpreting generated outputs.
- `examples/synthetic_c_lno_sto/generate_lno_sto_c_synthetic_data.py`
  - Synthetic C/LNO/STO data generator with reflectivity and La 4d, O 1s, Ti 2p, and C 1s RCs.
- `examples/synthetic_c_lno_sto/fit_lno_sto_c_synthetic_bo.py`
  - Normal BO fitting example using the synthetic C/LNO/STO data.
- `examples/synthetic_c_lno_sto/fit_lno_sto_c_synthetic_staged_bo.py`
  - Staged multi-start BO fitting example using the same synthetic data.
- `examples/synthetic_c_lno_sto/plot_fitted_stack_schematic.py`
  - Schematic visualization of the best-fit C/LNO/STO stack.

## Implementation steps

1. Create the package skeleton under `src/swxps`.
2. Add a simple `Layer` dataclass with:
   - `thickness`
   - `delta`
   - `beta`
   - a property or helper returning `n = 1 - delta + i beta`
3. Add `energy_to_wavelength(energy_ev)`.
4. Add `kz_in_layers(angle_deg, wavelength, refractive_indices)`.
5. Add `fresnel_r_s(kz_top, kz_bottom)`.
6. Add `parratt_reflectivity(angle_deg, energy_ev, layers)`.
7. Ensure the implementation accepts scalar or numpy-array angles.
8. Validate input lengths and layer ordering:
   - At least two layers are required.
   - First layer is expected to be vacuum.
   - Last layer is treated as semi-infinite.
9. Keep the core calculation independent of any optical-constant database.
10. Add tests before expanding beyond this milestone.
11. Add optional optical-constant file lookup as a helper outside the Parratt core.
12. Use linear interpolation for requested energies that fall between tabulated energies.
13. Add `roughness` to `Layer`, defaulting to zero for backward compatibility.
14. Apply the Nevot-Croce roughness correction to each interface Fresnel amplitude.
15. Add functions for calculating field amplitudes inside the multilayer at one incidence angle.
16. Add functions for sampling electric field and field intensity on a depth grid.
17. Add an LNO/STO example showing the field-strength distribution versus depth.
18. Add an independent sharp-interface transfer-matrix field solver.
19. Compare transfer-matrix reflectivity with Parratt reflectivity for zero-roughness stacks.
20. Add graded effective-layer construction for rough interfaces.
21. Use the graded effective stack in transfer-matrix reflectivity and field-profile calculations when layer roughness is nonzero.
22. Add convergence-oriented controls for rough-interface slicing step and roughness width.
23. Add IMFP table lookup from `IMFP`.
24. Add normalized SW-XPS rocking-curve integration with constant photoionization cross section.
25. Add an LNO/STO La 4d example at 1000 eV photon energy and 895 eV kinetic energy.
26. Add a stable high-level simulation API for reflectivity and normalized SW-XPS RCs.
27. Represent simulation stacks with material-labeled layers so optimization code can update thickness, roughness, and other physical parameters cleanly.
28. Include angle offset as an explicit simulation input.
29. Add stack profile sampling for visualizing element concentration versus depth.
30. Add an LNO/STO example showing roughness-broadened La, Ti, and O concentration profiles.
31. Add an optimizer-independent fitting layer that wraps the existing high-level simulation API.
32. Define fitting parameters as named bounded quantities, with a clean conversion from vector values to physical simulation requests.
33. Add dataset containers for experimental reflectivity and normalized SW-XPS rocking curves, including optional uncertainties and dataset weights.
34. Add residual functions for logarithmic reflectivity comparison and normalized rocking-curve comparison.
35. Add a scalar joint objective that can combine reflectivity and multiple SW-XPS rocking curves.
36. Add fit-history records storing parameter values, objective values, and per-dataset residual contributions.
37. Add a thin `scikit-optimize` Bayesian-optimization backend for the first implementation.
38. Keep BO-specific details behind a small adapter so later optimizers can be added without changing objective or simulation code.
39. Validate BO first on synthetic data generated from known parameters, starting with fewer than 10 parameters.
40. Only after synthetic recovery works, use the same fitting API on experimental datasets.

## Minimal JAX backend experiment

### Goal

Add a small optional JAX backend for the low-level Parratt reflectivity calculation so it can be compared against the existing NumPy implementation for numerical agreement, JIT timing, and differentiability. This is an experiment only: it must coexist with the NumPy backend and must not replace the high-level simulation API, the transfer-matrix roughness path, or the Bayesian-optimization workflow.

### Physics background

Use the same s-polarized Parratt recursion and Nevot-Croce roughness convention as `src/swxps/reflectivity.py`: angles are grazing angles in degrees, energy is in eV, thickness/roughness are in Angstrom, and each layer has `n = 1 - delta + i beta`. The first JAX loss for gradient checks should be a simple scalar mean-squared difference between simulated reflectivity and a fixed target curve, not a fitting objective.

### Files to create or modify

- `src/swxps/reflectivity_jax.py`
  - Optional JAX implementation of the low-level Parratt core.
  - JAX 64-bit enablement.
  - Jitted reflectivity and value-and-gradient helpers.
- `tests/test_reflectivity_jax.py`
  - Skip cleanly when JAX is not installed.
  - Compare the JAX reflectivity result with `parratt_reflectivity` for one representative fixed-shape input.
  - Check the JAX scalar loss gradient against a finite-difference derivative.
- `examples/benchmarks/benchmark_jax_reflectivity.py`
  - Compare NumPy wall time, first JAX JIT call including compilation, repeated JAX JIT calls, and JAX value-and-gradient timing.
- `pyproject.toml`
  - Add an optional `jax` extra only if useful for documenting the dependency.

### Implementation steps

1. Identify the smallest NumPy reflectivity core to port; the first target is `parratt_reflectivity` / `parratt_amplitude`, not full SW-XPS rocking curves.
2. Add a separate JAX module that accepts fixed-shape arrays for angles, layer thicknesses, deltas, betas, and roughnesses.
3. Use `jax.numpy` only inside traced functions and enable 64-bit precision.
4. Implement the layer recursion with JAX-compatible control flow such as `jax.lax.scan`.
5. Add a jitted reflectivity wrapper for fixed-shape inputs.
6. Add a scalar test loss and a jitted `jax.value_and_grad` wrapper for one differentiable parameter vector.
7. Add focused tests for NumPy/JAX agreement and a finite-difference gradient check.
8. Add a benchmark script for local timing diagnostics.
9. Add a second narrow JAX milestone for Parratt-based electric-field intensities on a precomputed fixed depth grid.
10. Add JAX attenuation and trapezoidal XPS integration helpers that accept precomputed concentration and IMFP/attenuation arrays.
11. Add a JAX normalized rocking-curve helper for one fixed core level and fixed off-peak mask.
12. Compare JAX field intensities and normalized RCs against the existing sharp-interface NumPy Parratt/XPS helpers before attempting transfer-matrix roughness discretization.

### Tests

- Existing tests must continue to pass.
- New JAX tests should skip when JAX is not installed.
- For an installed JAX environment, JAX reflectivity should agree with NumPy Parratt reflectivity within tight floating-point tolerances for a fixed representative multilayer.
- The JAX loss gradient should agree with a central finite-difference derivative for a small perturbation.

### Validation

The JAX backend is useful only if it reproduces the existing NumPy Parratt curve, its compiled repeated-call timing is competitive for fixed-shape arrays, and gradients are finite and numerically plausible. Gradient-based fitting, L-BFGS-B, replacement optimizers, and BO integration are explicitly out of scope for this milestone.

The second milestone is acceptable when JAX field intensities match the existing Parratt field profile for a sharp-interface stack and JAX normalized RCs match the existing `normalized_rocking_curve` behavior for zero roughness. Full transfer-matrix roughness discretization, graded effective-layer construction inside JAX, and multi-core high-level simulation integration remain out of scope until the narrow optical/XPS kernels are validated.

### Transfer-matrix JAX backend milestone

Goal: add an optional JAX/JIT backend that mirrors the current NumPy high-level simulation path closely enough to be selected by fitting or BO drivers without replacing the existing backend. The first practical version should precompute the same roughness-discretized effective stack with the existing NumPy helper, then pass fixed-shape arrays into JAX transfer-matrix kernels for reflectivity, field intensities, and normalized RC integration.

Physics background: use the same sharp-interface transfer-matrix equations currently implemented in `src/swxps/fields.py`, including the same effective-layer roughness approximation from `effective_layers_with_roughness`. Interface and propagation matrices remain the optical core; the JAX implementation should only change the array backend and loop machinery, not the physical model.

Files to create or modify:

- `src/swxps/simulation_jax.py`
  - Optional high-level JAX simulation entry points returning the same result dataclasses as `simulation.py`.
  - Conversion from `SimulationStack` / `Layer` objects to fixed-shape arrays outside JIT.
  - Reuse of existing NumPy effective-layer generation for 1:1 roughness behavior in the first milestone.
- `src/swxps/reflectivity_jax.py`
  - Add transfer-matrix batched amplitude, reflectivity, field-intensity, and RC kernels if a separate JAX kernel module is not introduced.
- `tests/test_simulation_jax.py`
  - Skip when JAX is unavailable.
  - Compare JAX reflectivity against `simulate_reflectivity` for rough stacks.
  - Compare JAX normalized RCs against `simulate_rocking_curves` for one or more core levels.
- `examples/benchmarks/benchmark_jax_reflectivity.py`
  - Add timing for the high-level transfer-matrix JAX reflectivity/RC wrappers.
- `src/swxps/fitting.py`
  - Add an opt-in simulation backend selector for fitting/BO while keeping NumPy as the default.

Implementation steps:

1. Port `_transfer_matrix_field_amplitudes_sharp_batched` to JAX using fixed-shape arrays and `lax.scan`.
2. Add JAX transfer-matrix reflectivity for arrays of angles.
3. Add JAX electric-field intensity sampling on the same fixed depth grid produced from the effective stack.
4. Add a high-level JAX reflectivity wrapper that calls `effective_layers_with_roughness` before entering JIT.
5. Add a high-level JAX rocking-curve wrapper that reuses one JAX field calculation for all requested core levels, matching the current NumPy `simulate_rocking_curves` structure.
6. Keep concentration grading, attenuation arrays, and emitting-layer selection outside JIT initially, using existing NumPy helpers for 1:1 behavior.
7. Return existing result dataclasses so fitting code can switch backend with minimal adapter code later.
8. Add tests against the existing NumPy transfer-matrix simulation path, including roughness.
9. Benchmark NumPy versus first JAX call, repeated JAX call, and JAX RC simulation with multiple core levels.
10. Add an optional `simulation_backend="jax"` path to `FittingProblem` only after the high-level JAX wrappers return the same result dataclasses.

Validation: JAX reflectivity and RC outputs must match the current NumPy transfer-matrix high-level simulation within floating-point tolerances for fixed representative rough stacks. The first call may be slower because it includes compilation; repeated calls should be measured separately. BO integration should remain opt-in until numerical agreement and repeated-call speed are demonstrated.

### Standalone JAX gradient optimizer milestone

Goal: add a small local optimizer that uses JAX/JIT value-and-gradient callbacks with SciPy `L-BFGS-B`. This optimizer must be separate from the existing Bayesian optimization backend and must not replace the NumPy simulation backend or the BO workflow.

Physics/numerics background: the optimizer sees only a scalar loss and gradient for a fixed differentiable simulation setup. The first implementation should optimize a bounded physical parameter vector, internally scaled to `[0, 1]` so thickness, roughness, and angle-offset parameters have comparable optimizer scales. Gradients returned by JAX with respect to physical parameters must be chain-rule scaled before passing to SciPy in scaled coordinates.

Files to create or modify:

- `src/swxps/jax_gradient.py`
  - Dataclasses for optimizer settings, iteration history, and result.
  - `optimize_with_jax_gradient` using SciPy `minimize(..., method="L-BFGS-B")`.
  - Parameter scaling helpers from physical bounds to `[0, 1]` and back.
  - Lazy SciPy import so the module can be imported without SciPy installed.
- `tests/test_jax_gradient.py`
  - Smoke test for one bounded initial vector.
  - Verify final loss is lower than initial loss on a simple deterministic objective.
  - Verify optimized parameters remain within bounds.
  - Add a JAX-specific test that skips when JAX is unavailable and uses an existing JAX value-and-gradient loss.
- `scripts/run_jax_gradient_fit.py`
  - Small representative script that constructs a JAX differentiable reflectivity loss, runs L-BFGS-B, and prints initial loss, final loss, best parameters, status, and wall time.
- `README.md`
  - Short note that the JAX gradient optimizer is local, needs a good initial guess, can get trapped in local minima, is separate from BO, and is best used from a physically reasonable starting structure.
- `pyproject.toml`
  - Add an optional extra for JAX gradient optimization if useful, including SciPy and JAX.

Implementation steps:

1. Add scaling helpers and result/history dataclasses.
2. Implement an optimizer function that accepts `FitParameter` definitions and a physical-space value-and-gradient callable.
3. Convert SciPy scaled vectors to physical vectors before evaluating the JAX loss.
4. Convert physical gradients back to scaled gradients using the parameter ranges.
5. Record callback history with iteration, loss, parameter vector, and gradient norm.
6. Return best parameter dictionary, best loss, SciPy status/message, iteration/evaluation counts, and history.
7. Keep BO untouched and expose the new optimizer independently.
8. Add tests for scaling, loss decrease, bounds, and optional JAX value-and-gradient usage.
9. Add a small script for a one-parameter synthetic reflectivity fit.

Validation: the first optimizer is acceptable when it can reduce a simple bounded loss from one initial vector, keeps all fitted values inside bounds, records a useful history, and can call an existing JAX value-and-gradient function when JAX is installed. Multi-start, BO+gradient hybrids, Adam, Optax, and production experimental fitting scripts are out of scope for this milestone.

## Tests

Required tests for this milestone:

1. Single interface / Fresnel limit
   - Use a two-layer stack: vacuum / substrate.
   - Compare Parratt reflectivity with the direct s-polarized Fresnel result.

2. Identical-index stack
   - Use several layers with identical refractive indices.
   - Reflectivity should be near zero across a representative angle grid.

3. Periodic multilayer Bragg peak
   - Build a repeated A/B multilayer between vacuum and substrate.
   - Use a bilayer period d.
   - Check that the strongest peak appears near:

     theta = arcsin(lambda / (2 d))

   - Use a tolerance broad enough for finite-stack effects.

4. Reflectivity upper bound
   - For physically reasonable absorbing layers, verify reflectivity does not exceed 1 except for small numerical tolerance.

5. Optical-constant table lookup
   - Parse fixed-format Henke/LBNL files with two header rows.
   - Return exact tabulated values when the requested energy exists.
   - Interpolate delta and beta for energies between table rows.
   - Raise a clear error for energies outside the table range.

6. Roughness
   - Zero roughness must reproduce the previous sharp-interface behavior.
   - Positive roughness should reduce high-angle reflectivity/fringe contrast.
   - Negative or non-finite roughness values should be rejected.

7. Electric field profile
   - Identical-index stacks should give uniform unit field intensity when there is no reflection.
   - The depth grid should cover finite layers but not the semi-infinite vacuum.
   - The returned field and intensity arrays should match the requested depth grid.
   - Transfer-matrix reflectivity should match Parratt reflectivity for sharp interfaces.
   - Transfer-matrix reflectivity with zero roughness should be unchanged by graded-stack machinery.
   - Positive roughness should damp high-angle multilayer reflectivity in the graded-stack model.
   - Rough-interface field profiles should be finite and generally attenuate into absorbing multilayers.

8. Normalized SW-XPS rocking curves
   - IMFP tables should parse headers and interpolate values.
   - Attenuation should decrease monotonically with depth.
   - A constant field and uniform concentration should give a flat normalized rocking curve.
   - The La 4d example should produce finite normalized intensities versus incidence angle.

9. High-level simulation API
   - Reflectivity simulation should return user angles and angle-offset-corrected calculation angles.
   - RC simulation should return named core-level outputs with normalized and raw intensities.
   - Multiple core levels should be simulated from the same stack and angle grid.

10. Stack profile visualization
   - Concentration profiles should be sampled on a depth grid through finite layers.
   - Rough interfaces should appear as smooth concentration transitions.
   - Profiles should support multiple elements/material properties on the same depth grid.

11. Roughness discretization controls
   - Keep the default graded-interface slice thickness at 1 Angstrom for user-facing simulations.
   - Allow users to provide either one global roughness slice thickness or one slice thickness per finite layer.
   - Allow roughness grading shape to be selected explicitly, initially `erf` or `linear`, while keeping `erf` as the default.
   - Split the error-function truncation factor from the linear ramp width factor, because they represent different things.
   - Match the default linear ramp to the same RMS roughness by treating it as the CDF of a uniform height distribution, so the ramp half-width is `sqrt(3) * roughness`.
   - Add tests that verify per-layer slice counts and linear roughness grading.

12. Fitting objective and parameter mapping
   - Parameter definitions should preserve names, bounds, units, and vector order.
   - Vector-to-stack mapping should update only the intended thicknesses, roughnesses, and angle offset.
   - Invalid parameter combinations should return a clear error or a finite penalty, according to the chosen objective policy.
   - Reflectivity residuals should support logarithmic comparison with a configurable positive floor.
   - Rocking-curve residuals should support optional experimental uncertainties and dataset weights.
   - Joint objectives should report both the total scalar objective and per-dataset contributions.

13. Bayesian-optimization backend
   - The first BO backend should use `scikit-optimize`.
   - Tests should use a cheap synthetic objective rather than expensive full simulations.
   - The BO adapter should not import or modify low-level physics functions.
   - The optimizer result should include best parameters, best objective, and evaluation history.
   - A synthetic LNO/STO-like example should recover known parameters within physically reasonable tolerance.

Suggested command:

`python -m pytest`

## Validation

The implementation is acceptable when:

- A two-layer stack reproduces the Fresnel reflectivity.
- No interfaces between identical refractive indices produce near-zero reflectivity.
- A periodic multilayer produces a Bragg-like maximum near the expected angle.
- Reflectivity remains finite, non-negative, and not unphysically above 1 for the tested physical cases.
- The electric-field profile is finite on the finite layer stack and reduces to the expected no-reflection limit for identical media.
- The normalized SW-XPS rocking curve is finite, positive, and normalized by explicitly selected off-peak angles.
- High-level simulation functions have explicit dataclass inputs and outputs suitable for later optimization loops.
- Stack concentration profiles can be inspected before RC simulation to verify layer sequence and roughness behavior.
- Graded roughness slicing is user-controllable and reports predictable slice counts for a known stack.
- Fitting objectives can combine reflectivity and SW-XPS rocking curves while keeping the forward simulation API unchanged.
- Bayesian optimization through `scikit-optimize` can recover a small synthetic parameter set before being trusted on experimental data.
- Optimizer-specific code is isolated enough that a different backend can replace `scikit-optimize` later.
- The code remains transparent enough that each equation maps clearly to the physics above.

## Standalone JAX nonlinear least-squares optimizer

### Goal

Add a third, independent optimizer backend that minimizes a concatenated
reflectivity and SW-XPS rocking-curve residual vector with
`scipy.optimize.least_squares(method="trf")`. Keep BO, L-BFGS-B, the NumPy
backend, and the validated JAX forward physics unchanged.

### Physics background

For reflectivity, use a configurable log residual with a positive floor. For
each normalized rocking curve, divide intensity differences by a stable curve
scale and multiply every block by `sqrt(weight / number_of_points)`. The total
SciPy cost is therefore `0.5 * sum(residuals**2)` without favoring a curve only
because it has more samples or a larger absolute signal.

### Files to create or modify

- Create `src/swxps/jax_least_squares.py` for residual construction, JAX
  Jacobians, scaled bounded TRF optimization, history, and result records.
- Create `tests/test_jax_least_squares.py` and
  `scripts/run_jax_least_squares_fit.py`.
- Update package exports, optional dependencies, and README documentation.

### Implementation steps

1. Build fixed-shape JAX residuals from a JAX-traceable simulator callback and
   existing reflectivity/rocking-curve dataset objects.
2. JIT the residual and `jax.jacfwd` Jacobian, with thin NumPy adapters for
   SciPy.
3. Run TRF in scaled `[0, 1]` coordinates and return physical parameters,
   residuals, Jacobian, status, evaluation counts, history, and covariance when
   estimable.
4. Add one small synthetic example using the existing JAX reflectivity kernel.

### Tests

- Check concatenated reflectivity plus two-RC residual length and Jacobian
  shape.
- Check a bounded synthetic fit runs, lowers its initial cost, and remains
  inside physical bounds.
- Run the existing JAX/NumPy physics comparison tests unchanged.

### Validation

The milestone is complete when TRF consumes JIT-compiled JAX residuals and
Jacobians, returns a lower finite cost for a representative fixed-shape
problem, and requires no changes to either existing optimizer workflow.

### Progress

- 2026-06-20: Inspected the BO, L-BFGS-B, fitting, JAX forward/loss, test, and
  documentation structure; selected a fixed-shape simulator callback as the
  differentiable boundary for the first least-squares backend.
- 2026-06-20: Implemented the standalone TRF backend with concatenated weighted
  reflectivity/RC residuals, configurable normalization, JIT-compiled
  `jax.jacfwd` Jacobians, scaled bounds, history, covariance estimation, tests,
  exports, documentation, and a reflectivity example. Verified with
  `python -m pytest`; all 83 tests passed.

## Progress log

- 2026-06-09: Created the execution plan for the first x-ray reflectivity milestone.
- 2026-06-09: Implemented package skeleton, layer model, energy-to-wavelength conversion, s-polarized Fresnel amplitudes, Parratt recursion, and the required tests.
- 2026-06-09: Verified with `python -m pytest`; all 4 tests passed.
- 2026-06-09: Added a plotting example for a vacuum / [LaNiO3 / SrTiO3] multilayer / SrTiO3 substrate stack.
- 2026-06-09: Planned optical-constant lookup from Henke/LBNL `.dat` files in the `OPC` folder, with interpolation.
- 2026-06-09: Planned RMS interface roughness support using a Nevot-Croce correction.
- 2026-06-09: Implemented layer roughness, Nevot-Croce interface damping, roughness tests, and roughness support in file-derived layers.
- 2026-06-09: Verified with `python -m pytest`; all 12 tests passed.
- 2026-06-10: Planned electric-field profile support as a precursor to later standing-wave XPS intensity calculations.
- 2026-06-10: Implemented depth-dependent electric-field profiles, tests, and an LNO/STO field-intensity example.
- 2026-06-10: Verified with `python -m pytest`; all 16 tests passed.
- 2026-06-10: Planned independent transfer-matrix field solver and reflectivity comparison against Parratt.
- 2026-06-10: Implemented a sharp-interface transfer-matrix field solver; transfer-matrix reflectivity matches Parratt for zero roughness in tests.
- 2026-06-10: Updated field-profile examples to use transfer-matrix fields with zero roughness.
- 2026-06-10: Planned transfer-matrix roughness support using graded effective layers made from many thin sharp-interface slices.
- 2026-06-10: Implemented graded-interface transfer-matrix roughness support for reflectivity and electric-field profiles.
- 2026-06-10: Updated LNO/STO examples to use transfer-matrix reflectivity and graded-roughness field profiles.
- 2026-06-10: Verified with `python -m pytest`; all 21 tests passed.
- 2026-06-11: Updated the LNO/STO electric-field example to plot a depth-angle contour map of `|E|^2`.
- 2026-06-11: Planned first normalized SW-XPS rocking-curve milestone with constant cross sections and IMFP table lookup.
- 2026-06-11: Implemented IMFP table parsing and interpolation from `IMFP`.
- 2026-06-11: Implemented normalized SW-XPS rocking-curve integration with constant cross sections, depth-dependent field intensity, concentration, and electron attenuation.
- 2026-06-11: Added a normalized La 4d rocking-curve example for LNO/STO at 1000 eV photon energy and 895 eV electron kinetic energy.
- 2026-06-11: Verified with `python -m pytest`; all 30 tests passed.
- 2026-06-11: Expanded the rocking-curve angle range and updated the example to compare La 4d and O 1s normalized RCs.
- 2026-06-11: Added Ti 2p to the normalized rocking-curve example and split La 4d, O 1s, and Ti 2p RCs into vertically aligned subplots.
- 2026-06-12: Planned a high-level simulation API with material-labeled stacks, explicit angle offsets, and clean dataclass results for future fitting workflows.
- 2026-06-12: Implemented `simulation.py` with material-labeled stacks, reflectivity requests/results, core-level requests/results, angle-offset handling, and multi-core RC simulation.
- 2026-06-12: Updated the La 4d/O 1s/Ti 2p example to use the high-level simulation API and reuse field profiles across core levels.
- 2026-06-12: Verified with `python -m pytest`; all 33 tests passed.
- 2026-06-12: Planned stack profile visualization for checking concentration distributions and roughness before RC simulation.
- 2026-06-12: Implemented stack concentration profile sampling and an LNO/STO La/Ti/O concentration-versus-depth visualization example.
- 2026-06-12: Verified with `python -m pytest`; all 36 tests passed.
- 2026-06-12: Planned user-controllable roughness discretization with scalar or per-layer slice thickness and selectable grading shape.
- 2026-06-12: Planned separate roughness-width controls and RMS-matched linear ramp defaults for comparing error-function and linear roughness descriptions.
- 2026-06-15: Planned an initial fitting milestone using `scikit-optimize` Bayesian optimization, with optimizer-independent parameter mapping, objective evaluation, dataset weighting, and fit-history records.
- 2026-06-15: Implemented optimizer-independent fitting helpers for bounded parameters, stack updates, reflectivity and SW-XPS residuals, joint objectives, and fit-history records.
- 2026-06-15: Implemented a thin `scikit-optimize` Bayesian-optimization backend with package-native settings and result dataclasses.
- 2026-06-15: Added fitting and BO adapter tests; verified with `python -m pytest`; all 47 tests passed.
- 2026-06-15: Added a synthetic C/LNO/STO dataset generator with reflectivity plus normalized La 4d, O 1s, Ti 2p, and C 1s rocking curves for later BO fitting tests.
- 2026-06-15: Added a finite-layer fitting constraint so roughness cannot exceed layer thickness while still allowing semi-infinite substrate interface roughness.
- 2026-06-15: Reorganized examples into per-example folders and updated scripts to write outputs next to each example.
- 2026-06-15: Added a BO fitting example for the synthetic C/LNO/STO dataset with history CSV, convergence plot, and best-fit comparison plot outputs.
- 2026-06-15: Added GP surrogate-slice diagnostic plots that show objective mean and standard deviation versus each fit parameter.
- 2026-06-15: Added a high-level `FittingProblem` API and `run_bayesian_fit` wrapper so users can provide datasets, stack builders, parameter bounds, initial guesses, and independent dataset weights without hand-writing objective plumbing.
- 2026-06-15: Added declarative stack templates, including `SuperlatticeTemplate`, so common stacks can be described without manual layer-append loops.
- 2026-06-15: Updated examples to use declarative stack templates and added StackTemplate boundary validation for vacuum first layer and zero-thickness semi-infinite substrate last layer.
- 2026-06-15: Added reusable fitting diagnostics utilities for history CSV export, convergence plots, best-fit plots, and GP surrogate-slice plots.
- 2026-06-15: Added a staged multi-start BO fitting driver that fits selected parameter groups first, carries best values forward as fixed values, repeats each stage with independent seeds, and exports a per-stage summary.
- 2026-06-15: Planned a reusable sample-stack schematic visualization utility that can draw fitted multilayer stacks, collapse repeated interior layers, annotate layer thicknesses, and show incident/diffracted x-rays plus a stylized standing wave.
- 2026-06-15: Implemented stack schematic plotting utilities and a fitted C/LNO/STO schematic example based on saved BO best-fit parameters.
- 2026-06-16: Updated README descriptions for the current fitting workflow and planned tracking representative `.png` and `.csv` example outputs in GitHub.
- 2026-06-16: Planned an experimental Sample#13 preparation/fitting script that reads unnormalized reflectivity/RC data, keeps reflectivity raw, edge-normalizes RCs for inspection, reports the proposed stack and fit ranges, and only runs BO when explicitly requested.
- 2026-06-16: Planned explicit Sample#13 fit modes for reflectivity-only, RC-only, and joint BO fitting, with live objective progress reporting for diagnosing slow experimental fits.
- 2026-06-16: Planned reusable RC background subtraction utilities that fit polynomial backgrounds to user-selected edge percentages while leaving reflectivity preprocessing untouched.
- 2026-06-16: Planned layer-selective SW-XPS emission controls as the recommended/default fitting setup for short-IMFP core levels, so RCs can be integrated over selected near-surface layers without changing the optical stack.
- 2026-06-16: Planned timing diagnostics for BO fitting to separate forward objective time, reflectivity simulation time, RC simulation time, scoring time, and BO/GP optimizer overhead.
- 2026-06-16: Planned Sample#13 reflectivity-fit refinement using prior fit results to tighten LNO/STO superlattice thickness ranges, correct the superlattice ordering to STO/LNO above the STO substrate, and fit separate STO and LNO superlattice roughnesses.
- 2026-06-16: Planned separate Sample#13 stack builders for reflectivity and RC fitting: reflectivity omits the surface C layer and uses one continuous LNO cap, while RC fitting keeps the C layer and split LNO cap for layer-selective emission.
- 2026-06-16: Planned Sample#13 superlattice update from 20 to 40 STO/LNO repeats based on the corrected experimental README before rerunning reflectivity-only BO.
- 2026-06-16: Planned a Sample#13 reflectivity-only comparison with the surface C layer included as fitted thickness/roughness parameters while keeping a single continuous LNO cap.
- 2026-06-16: Planned Sample#13 non-ideal superlattice fitting based on the legacy MATLAB model, using repeat-dependent STO/LNO thicknesses and linearly varying STO/LNO roughnesses while omitting C from reflectivity fitting.
- 2026-06-16: Planned BO-friendly Sample#13 graded-superlattice parameters that enforce valid thickness ordering and roughness profiles by construction, replacing invalid-point penalties with positive thickness deltas and first/last roughness values.
- 2026-06-16: Planned Sample#13 graded-superlattice refinement using an explicitly error-function-like thickness transition and tighter 2-5 Angstrom roughness bounds for physically plausible reflectivity fits.
- 2026-06-16: Planned Sample#13 RC-only BO fitting initialized from the reflectivity-only BO best result, reusing the erf-like graded superlattice, fitting the surface C layer over tighter bounds, and weighting C 1s/La 4d/Ni 3p as 0.5/3/3.
- 2026-06-16: Planned a constrained Sample#13 RC-only BO rerun that fixes the superlattice thickness profile to the reflectivity best result, uses a 50 Angstrom top LNO emitting slab, and widens the RC angle-offset range to +/-0.3 degrees.
- 2026-06-16: Planned a full Sample#13 RC-only BO rerun using the 50 Angstrom top LNO emitting slab while allowing the erf-like superlattice thickness and roughness profile parameters to vary again.
- 2026-06-17: Planned a Sample#13 optical-constant update to use the newly supplied 800-900 eV LNO/STO OPC tables near 815 eV, then rerun reflectivity-only BO before revisiting RC fitting.
- 2026-06-17: Planned a Sample#13 reflectivity-only BO rerun using only the 12-15 degree reflectivity window to avoid low-angle points steering BO into an unphysical basin.
- 2026-06-17: Planned a simplified Sample#13 reflectivity-only BO rerun with roughness values fixed and only cap thickness, erf-like superlattice thickness parameters, and a widened +/-0.2 degree angle offset active.
- 2026-06-17: Planned a follow-up Sample#13 thickness-only reflectivity BO rerun with the same optional erf-like thickness gradient but a wider +/-0.3 degree angle-offset range.
- 2026-06-17: Planned a staged Sample#13 reflectivity roughness-only BO rerun that fixes the thickness profile and angle offset from the previous best reflectivity history row.
- 2026-06-17: Planned a Sample#13 three-RC BO rerun initialized from the staged reflectivity baseline, with the 50 Angstrom top LNO signal slab and layer-selective La/Ni emission from that slab only.
- 2026-06-17: Planned and ran a Sample#12 reflectivity-only BO setup from `examples/LNO_STO_LNO_case_Sample#12/Readme.md`, reusing the C/LNO/[STO/LNO]x40/STO stack, 815 eV reflectivity data, and the erf-like superlattice thickness gradient profile before later RC fitting; the 80-call run saved history and diagnostic plots in the Sample#12 folder.
- 2026-06-17: Planned and ran a Sample#12 RC simulation from the best reflectivity BO stack, with a fixed 2 Angstrom surface C layer and a split top LNO cap composed of a 50 Angstrom rough emitting slab plus a zero-roughness buried remainder, comparing simulated C 1s, Ni 3p, and La 4d rocking curves against the experimental RCs.
- 2026-06-17: Planned and ran a Sample#12 three-RC Bayesian-optimization fit initialized from the reflectivity BO best stack, keeping the 2 Angstrom C layer and 50 Angstrom top LNO emitting slab fixed while fitting cap thickness, erf-like superlattice thickness profile, roughness profile, and RC angle offset against C 1s, Ni 3p, and La 4d.
- 2026-06-18: Planned an `OPC/C.dat` access repair after Windows reported access denied for the working-tree file while the tracked Git blob remained readable; replace only the broken OneDrive reparse-point placeholder with the tracked optical-constant table and verify sandbox reads.
- 2026-06-18: Planned a forward-simulation efficiency pass before additional fitting runs. Goal: preserve reflectivity and SW-XPS physics while reducing repeated work in the transfer-matrix forward model. Physics background: roughness-discretized effective layers, nominal depth grids, concentration grading, and electron attenuation are independent of incidence angle for a fixed stack, while transfer-matrix products remain sequential over layers but can be batched over angles. Files to modify: `src/swxps/fields.py`, `src/swxps/simulation.py`, possibly `src/swxps/xps.py`, plus focused tests if behavior changes. Implementation steps: measure a representative reflectivity/RC baseline, precompute effective roughness stacks once per request, add batched transfer-matrix amplitude/profile helpers that keep layer recursion explicit but vectorize over angle/depth arrays, reuse concentration and attenuation arrays per core level, then compare pre/post outputs. Tests/validation: existing reflectivity/field/XPS/simulation tests must pass; representative reflectivity and RC arrays should match baseline within floating-point tolerance and report improved timing.
- 2026-06-18: Implemented the forward-simulation efficiency pass. For a 121-angle rough C/LNO/[LNO/STO]x20/STO benchmark at 1000 eV, best-of-three reflectivity timing improved from 2.04 s to 0.031 s and four-core RC timing improved from 11.03 s to 0.070 s. Reflectivity and RC outputs matched the pre-change baseline to roundoff-level differences, and the full `python -m pytest` suite passed with 70 tests.
- 2026-06-18: Planned a Sample#12 joint reflectivity plus three-RC BO script with a revised cap stack: fitted C thickness/roughness, two fitted-total top LNO slabs over a fixed 160 Angstrom zero-roughness LNO bottom cap, layer-selective C/Ni/La emission, RC weights C 1s:La 4d:Ni 3p = 0.5:3:3, and an explicit reflectivity log-MSE weight reported before running BO.
- 2026-06-18: Archived older Sample#12 generated BO fitting artifacts into `examples/LNO_STO_LNO_case_Sample#12/previous_BO_fitting` and ran the new joint cap3 reflectivity-plus-RC BO for 80 calls with auto reflectivity weighting. The best finite result was evaluation 15 with objective 0.0230232; 69 of 80 evaluations were finite because independent carbon thickness/roughness bounds can produce invalid roughness > thickness candidates.
- 2026-06-18: Implemented a Sample#12 joint cap3 BO cleanup: reparameterized carbon roughness through a fitted fraction so the derived roughness is always between 1 Angstrom and min(8 Angstrom, carbon thickness), cropped reflectivity to the RC-angle range by default, and switched the joint best-fit plot to distinct data-point colors for reflectivity and each RC panel.
- 2026-06-18: Ran the updated Sample#12 joint cap3 BO with 240 calls and 80 initial points. The best objective improved to 0.0118544 at evaluation 42 using the cropped reflectivity window and constrained carbon roughness; 229 of 240 evaluations were finite, with remaining invalid candidates attributable to the independently fitted first LNO cap thickness and cap roughness.
- 2026-06-18: Implemented a follow-up Sample#12 joint cap3 parameter cleanup to reparameterize the first LNO cap roughness through a fitted fraction so the derived roughness is always between 1 Angstrom and min(5 Angstrom, first LNO cap thickness), with the first LNO cap thickness lower bound raised to 1.01 Angstrom to keep a nonzero roughness physically valid.
- 2026-06-18: Changed Sample#12 joint cap3 La 4d emission from only the second top LNO slab to both top LNO slabs, matching the Ni 3p emitting-layer selection, then reran the 240-call BO. The best objective improved to 0.00630033 at evaluation 178, with all 240 evaluations finite after the carbon and cap-roughness reparameterizations.
- 2026-06-18: Ran a small multi-seed Sample#12 joint cap3 BO batch with stronger fixed reflectivity weights, using separate output prefixes for each run. Tested reflectivity weights 0.10 and 0.20 with random seeds 12 and 112 at 120 calls / 50 initial points. The fixed-weight runs showed optimizer variability and improved reflectivity raw error in the best 0.20/seed12 case, but the previous 240-call auto-weight run still had the best combined objective and RC residuals.
- 2026-06-18: Ran a longer Sample#12 seeded BO batch after archiving older fitting artifacts into `previous_fitting`. Intermediate fixed-weight runs were written to `seeded_BO_runs` for reflectivity weights 0.09, 0.075, and 0.06 with seed 12 at 180 calls / 60 initial points. The best new seeded result was the 0.06-weight run with objective 0.00967951; its main artifacts plus a summary CSV/plot were exported back into the main Sample#12 folder. The archived auto-weight 240-call result remains the best overall objective at 0.00630033.
- 2026-06-18: Implemented and ran a Sample#12 staged multi-start BO driver for the joint cap3 model. It fits angle/superlattice geometry, cap/C parameters, roughness parameters, then all parameters; writes one subfolder per stage/start; records `summary.csv` and `final_best_parameters.csv`; and promotes the best final fit/stack plots to the run root. Existing generated Sample#12 fitting outputs and previous run folders were archived into `archive_before_staged_multistart`. Verified with `python -m py_compile` and a one-start/four-call staged smoke run, then removed smoke artifacts. The full 3-start, 4-stage, 120-call-per-start run reached best objective 0.0062494 in `04_final_all/start_02_seed_3014`, slightly improving on the previous 0.00630033 single-run best.
- 2026-06-18: Planned a refresh of `examples/synthetic_c_lno_sto` after the fitting and simulation APIs changed substantially. Goal: archive old synthetic C/LNO/STO BO scripts/artifacts and write a current joint reflectivity-plus-RC BO driver for studying robust BO settings. Physics background: the synthetic stack is vacuum/C/[LNO/STO]x20/STO at 1000 eV, with reflectivity fitted in log space and normalized La 4d, O 1s, Ti 2p, and C 1s rocking curves fitted in linear intensity after off-peak normalization. Files to modify: `examples/synthetic_c_lno_sto/*` and this plan. Implementation steps: move old example files into a subfolder, make a self-contained stack/data/problem builder, add BO controls for staged versus joint fitting and dataset weighting, save history/convergence/best-fit/schematic diagnostics, and verify with a short smoke run. Tests/validation: the script should construct the joint fitting problem, run a small BO job, recover physically plausible parameters near the synthetic truth, and keep reflectivity plus RC simulations finite.
- 2026-06-18: Refreshed `examples/synthetic_c_lno_sto`: archived previous scripts/artifacts into `previous_fitting`, added `fit_reflectivity_rc_bo.py` with self-contained synthetic data generation, auto reflectivity-weight balancing, direct and staged BO modes, and diagnostic outputs. Regenerated the 161-point synthetic CSV/preview and verified the script with `python -m py_compile` plus a 4-call end-to-end BO smoke run.
- 2026-06-18: Ran synthetic C/LNO/STO BO comparisons. A 200-call direct single-seed run reached objective `1.931e-4` with near-zero angle offset and was archived under `single_run_outputs`. A staged 2-start run with 100 calls per stage/start reached `5.549e-4`, getting locked into a positive angle-offset/thickness basin; an alternate direct seed-112 run reached `1.147e-3`. Distinct data colors were added for each RC in the synthetic best-fit plot.
- 2026-06-18: Cleaned the synthetic example directory by moving staged/multi-start comparison outputs into `staged_multistart_outputs`. Ran another direct single BO with 240 calls, 70 initial points, and seed 24; it reached objective `3.919e-4` at evaluation 142, better than staged/multi-start but not better than the archived 200-call seed-12 direct run (`1.931e-4`).
- 2026-06-18: Planned an automated multi-seed synthetic C/LNO/STO BO driver. Goal: reduce seed luck in the noiseless synthetic benchmark by running repeated direct BO fits, saving per-seed artifacts, writing a summary CSV, and promoting the best run to stable output names. Files to modify: `examples/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py`, synthetic example output folders, and this plan. Validation: compile the script and run a small multi-seed smoke test that creates per-seed histories plus a summary and promoted best-fit artifacts.
- 2026-06-18: Implemented the synthetic multi-seed BO driver with `--multi-seed`, `--seeds`, `--seed-start`, `--seed-count`, and `--multi-seed-dir`. Each seed writes artifacts in its own subfolder, a summary CSV records objective/contribution/parameter/timing columns, and the best seed is promoted to stable `*_best_*` artifacts in the multi-seed output folder. Verified with `python -m py_compile` and a two-seed four-call smoke run, then removed smoke artifacts.
- 2026-06-19: Planned and implemented the first minimal JAX backend experiment for the low-level Parratt reflectivity core. Added optional `src/swxps/reflectivity_jax.py` with 64-bit JAX, fixed-shape array inputs, a `lax.scan` Parratt recursion, jitted reflectivity, scalar MSE loss, and jitted value-and-gradient wrapper. Added JAX tests that skip when JAX is unavailable, plus a benchmark script for NumPy/JAX timing. Verified with `python -m pytest`; all 70 existing tests passed and the JAX test module skipped because JAX is not installed in this environment.
- 2026-06-19: Expanded the minimal JAX experiment beyond reflectivity to include Parratt reflection amplitudes, layer field amplitudes, fixed-depth-grid electric-field intensities, attenuation, raw XPS intensity integration, normalized rocking curves, and a normalized-RC value-and-gradient wrapper. Added skipped-when-unavailable JAX tests for field intensity agreement, normalized RC agreement, and finite-difference RC gradients, and expanded the benchmark script to time field and RC kernels. Verified the non-JAX suite with `python -m pytest`; all 70 existing tests passed and the JAX test module skipped because JAX is still not installed in this environment.
- 2026-06-19: Added an optional transfer-matrix JAX high-level backend for reflectivity and SW-XPS RCs. The wrapper reuses the existing NumPy effective-layer roughness construction for 1:1 physics, then runs fixed-shape JAX transfer-matrix reflectivity, field-intensity, and normalized-RC kernels. Added skipped-when-unavailable comparison tests against `simulate_reflectivity` and `simulate_rocking_curves`, expanded the benchmark for high-level backend timing, and added `simulation_backend="jax"` to `FittingProblem` while preserving the default NumPy path. Verified with `python -m pytest`; all 70 existing tests passed and the two JAX test modules skipped because JAX is not installed in this environment.
- 2026-06-19: Planned a synthetic C/LNO/STO BO backend comparison now that JAX is installed locally. Goal: run one direct BO fit with the existing NumPy forward backend and one direct BO fit with the JAX high-level forward backend using identical 40-initial/200-total settings, seed, data stride, parameter bounds, and objective weights. Files to create: a new timestamped output folder under `examples/synthetic_c_lno_sto` containing a comparison runner, per-backend histories/plots, and summary CSV/text files. Validation: both BO runs should complete, save diagnostics, report timing fields from `BayesianOptimizationResult`, and compare best objectives/parameters against the synthetic truth.
- 2026-06-19: Planned and implemented a standalone JAX-gradient optimizer using SciPy L-BFGS-B in `src/swxps/jax_gradient.py`. The optimizer accepts existing `FitParameter` bounds, optimizes scaled `[0, 1]` variables, calls a physical-space value-and-gradient callback, chain-rule scales gradients for SciPy, records iteration/loss/parameter/gradient-norm history, and returns package-native result dataclasses. Added exports, a `gradient` optional dependency extra, tests for scaling/loss reduction/bounds/existing JAX value-and-gradient callbacks, a small `scripts/run_jax_gradient_fit.py` example, and README documentation that this is a local optimizer separate from BO. Corrected `normalized_rocking_curve_jax` to use the transfer-matrix JAX field kernel so it matches the validated NumPy RC path. Verified with `python -m pytest`; all 81 tests passed with JAX installed.
- 2026-06-19: Planned a synthetic C/LNO/STO JAX-gradient fitting example. Goal: create a new output folder that runs the standalone `jax_gradient.py` L-BFGS-B optimizer on the synthetic benchmark, saves gradient history and diagnostic plots, evaluates the gradient result with the existing high-level NumPy objective, and compares objective/parameters/wall time against existing BO summaries. Physics note: the current differentiable RC loss uses fixed-shape JAX arrays and a fixed depth grid, so this is an experimental local optimizer comparison rather than an exact replacement for the high-level roughness-discretized BO objective.
- 2026-06-19: Planned a repair of the synthetic JAX-gradient fitting example after auditing the code. The high-level JAX simulation path is already a 1:1 transfer-matrix counterpart to NumPy, but the first gradient helper bypassed it by using nominal-layer Parratt/RC kernels and fixed concentration grids. Fix: add an exact-objective gradient run that calls `FittingProblem.evaluate` with `simulation_backend="jax"` for every parameter vector so roughness discretization, depth grids, concentration grading, attenuation, and scoring match BO. Since this high-level path has parameter-dependent effective-stack shapes, use finite-difference gradients as a practical bridge inside `optimize_with_jax_gradient`, then compare against BO and the previous approximate-gradient run.
- 2026-06-19: Planned a Sample#12 JAX-gradient replacement for the joint cap3 BO workflow. Goal: tidy the Sample#12 directory by archiving old `.py` BO scripts, create a new gradient-fitting folder, reuse `fit_sample12_joint_cap3_bo.py` for the data preparation, cap3 stack, dataset weights, and plotting context, then run the high-level JAX simulation objective through the standalone L-BFGS-B gradient optimizer. Physics background: keep the same 815 eV C/LNO/LNO/STO/STO cap3 model, reflectivity log-MSE, and normalized C 1s/Ni 3p/La 4d rocking-curve MSE terms; only replace the optimizer path. Files to modify: `examples/LNO_STO_LNO_case_Sample#12/*`, this plan. Validation: the gradient runner should compile, evaluate the initial objective, write history/contribution/fit/stack artifacts, and print fitted parameters plus derived C and cap roughnesses.
- 2026-06-19: Implemented and ran the Sample#12 JAX-gradient replacement. Moved the old top-level Sample#12 `.py` scripts into `old_bo_python_scripts`, added `jax_gradient_fit/fit_sample12_joint_cap3_jax_gradient.py`, verified it with `python -m py_compile`, and ran the high-level JAX objective from the staged BO best starting point. The initial NumPy/JAX objective was `0.0062494`; L-BFGS-B converged successfully in 25 iterations and 75 function evaluations, with re-evaluated NumPy/JAX objective `0.00390254`. Artifacts were written under `examples/LNO_STO_LNO_case_Sample#12/jax_gradient_fit`.
- 2026-06-19: Planned a follow-up Sample#12 gradient multi-attempt run. Goal: run several L-BFGS-B attempts from the staged BO best plus small bounded perturbations, save each attempt in a new subfolder, promote the best attempt to run-root artifacts, and include the old BO superlattice thickness/roughness gradient plot for the fitted repeat stack. Validation: compile the updated runner, run the multi-attempt fit, check the attempt summary, and confirm the promoted best artifacts include best fit, stack schematic, convergence, JAX contributions, NumPy validation contributions, and superlattice profile.
- 2026-06-19: Implemented the Sample#12 multi-attempt gradient runner update. The script now supports `--attempts`, bounded random perturbations around the selected start, `--output-dir`, per-attempt folders, a promoted best result, clearer `validation_numpy_contributions` naming, and `sample12.save_superlattice_profile_plot` output for the fitted superlattice thickness/roughness profile. Verified with `python -m py_compile`. A first 4-attempt command timed out after completing two attempts; the better completed attempt reached JAX/NumPy objective `0.00368157`. A second completed-output run in `jax_gradient_fit/attempts_run_more_2` used two attempts and promoted attempt 2, reaching JAX/NumPy objective `0.00365685`; it hit the L-BFGS-B iteration limit at 35 iterations, indicating possible room for further local refinement.
- 2026-06-19: Planned a cleanup of the Sample#12 JAX-gradient output folder followed by one clean longer single-attempt run. Goal: delete old generated run files/subfolders from `examples/LNO_STO_LNO_case_Sample#12/jax_gradient_fit` while preserving `fit_sample12_joint_cap3_jax_gradient.py`, then run one 50-iteration staged-best JAX-gradient refinement into a single new subfolder. Validation: confirm the folder contains only the script plus the new run folder, and report the new objective/contributions/artifact paths.
- 2026-06-19: Completed the Sample#12 JAX-gradient output cleanup and clean single run. Deleted old generated artifacts/subfolders from `jax_gradient_fit`, preserving only `fit_sample12_joint_cap3_jax_gradient.py`, then ran one staged-best attempt with `--maxiter 50` into `clean_single_50iter`. L-BFGS-B converged before the limit in 25 iterations / 75 function evaluations. Initial JAX objective was `0.0062494`; final JAX and NumPy-validation objectives matched at `0.00390254`. The cleaned folder now contains only the script and `clean_single_50iter`.
- 2026-06-19: Planned a Sample#13 cleanup plus JAX-gradient cap3 fitting workflow. Goal: reorganize the messy Sample#13 directory into data files, archived BO scripts/artifacts, and a new `jax_gradient_fit` folder mirroring Sample#12; then add a high-level JAX-gradient runner that keeps Sample#13 angle windows, photon energy, RC weights, and old BO geometry while splitting the LNO cap into LNO-1/LNO-2/LNO-bottom. Physics background: the cap stack is `vacuum/C/LNO-1/LNO-2/LNO-bottom/[STO/LNO]x40/STO`; LNO-1 is constrained to 2-20 Angstrom and represents the thin Ni-free top LNO region, so La 4d emits from LNO-1 and LNO-2 while Ni 3p emits only from LNO-2. Files to create/modify: `examples/LNO_STO_LNO_case_Sample#13/old_bo_python_scripts`, `examples/LNO_STO_LNO_case_Sample#13/bo_outputs`, `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit/fit_sample13_joint_cap3_jax_gradient.py`, and this plan. Validation: compile the new runner, execute setup evaluation without `--run-fit`, and confirm initial NumPy/JAX objectives are finite with the intended layer-selective emission.
- 2026-06-19: Updated the Sample#13 JAX-gradient cap3 runner carbon-thickness lower bound from 5 Angstrom to 2 Angstrom, then ran one `--attempts 1 --maxiter 50` fit in `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit/clean_single_50iter`. The initial NumPy/JAX objective was `0.036289`; the best NumPy/JAX validation objective improved to `0.0203535`. SciPy terminated with status 2 (`ABNORMAL`) after 43 iterations and 170 function evaluations, so the run improved the fit but did not report clean optimizer convergence.
- 2026-06-19: Planned a Sample#13 follow-up gradient multi-start after the best single run put `top_lno_layer1_thickness` on its 2 Angstrom lower bound. Goal: relax the LNO-1 lower bound to 1 Angstrom, keep the Ni-free LNO-1 / La-emitting LNO-1+LNO-2 RC selection, and run three 35-iteration bounded perturbation starts in a new output folder. Validation: compile the updated runner, run `--attempts 3 --maxiter 35`, compare the attempt summary, and check whether the relaxed LNO-1 thickness or other parameters remain bound-limited.
- 2026-06-19: Completed the Sample#13 `top_lno_layer1_thickness >= 1 Angstrom` three-start, 35-iteration gradient run in `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit/layer1_1A_multistart_35iter`. The best run was attempt 2 with NumPy/JAX validation objective `0.010005`, improving from the previous single-run objective `0.0203535`; attempts 1 and 2 reached the iteration limit, while attempt 3 converged to a worse local minimum. Best attempt 2 placed `carbon_thickness` at its 15 Angstrom upper bound and `sto_thickness_delta`/`lno_thickness_delta` at their 0 Angstrom lower bounds; `top_lno_layer1_thickness` relaxed to `1.47249 Angstrom`, no longer exactly at the lower bound.
- 2026-06-19: Planned a new Sample#13 gradient-fitting variant with the first LNO layer upper-interface roughness fixed to zero and the second LNO layer upper-interface roughness fitted. Goal: create a separate folder under `examples/LNO_STO_LNO_case_Sample#13` for this variant, preserve the same cap stack and layer-selective RC emission, and reparameterize the LNO-2 roughness as a fitted fraction of `top_lno_layer1_thickness` so it cannot exceed the thin Ni-free layer thickness. Physics background: in the stack convention, a layer roughness is the RMS roughness of that layer's upper interface; therefore LNO-1 roughness controls the C/LNO-1 interface and LNO-2 roughness controls the LNO-1/LNO-2 interface. Files to create/modify: `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit_lno2_roughness/fit_sample13_joint_cap3_jax_gradient_lno2_roughness.py` and this plan. Validation: compile the new runner and execute a no-fit setup evaluation to confirm finite matching NumPy/JAX objectives and the intended derived roughness values.
- 2026-06-19: Implemented the Sample#13 LNO-2 roughness variant in `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit_lno2_roughness`. The runner fixes the C/LNO-1 interface roughness to `0 A`, fits `lno2_roughness_fraction` in `[0, 1]`, and derives `lno2_roughness = lno2_roughness_fraction * top_lno_layer1_thickness`, enforcing the requested roughness constraint. Verified with `python -m py_compile` and a no-fit setup run; initial NumPy/JAX objectives matched at `0.0233661`, with auto reflectivity weight `0.0901584`.
- 2026-06-19: Ran the Sample#13 LNO-2 roughness variant with one start and a 50-iteration limit. The first launch exposed an invalid-candidate path where a trial stack had non-positive RC normalization; patched the finite-difference objective to return a large penalty for invalid `ValueError` candidates, then reran into `jax_gradient_fit_lno2_roughness/single_50iter_penalty`. The optimizer reported convergence after 1 iteration / 4 function evaluations and stayed at the initial objective `0.0233661`, with derived roughnesses `lno1_roughness = 0 A` and `lno2_roughness = 1.5 A`. Generated roughness-aware top-100-A C/La/Ni concentration CSV and a shaded layer/profile plot in the same output folder.
- 2026-06-19: Reset `examples/LNO_STO_LNO_case_Sample#13/jax_gradient_fit_lno2_roughness` by deleting failed run folders, generated plots/data, cache files, and helper scripts, preserving only the fitting script. Updated the variant so both LNO-1 and LNO-2 upper-interface roughnesses are fitted as fractions of `top_lno_layer1_thickness`: `lno1_roughness = lno1_roughness_fraction * top_lno_layer1_thickness` and `lno2_roughness = lno2_roughness_fraction * top_lno_layer1_thickness`, enforcing both requested roughness constraints. Verified with `python -m py_compile` and a no-fit setup run; initial NumPy/JAX objectives matched at `0.0233697`, with auto reflectivity weight `0.0906933`.
- 2026-06-19: Ran the constrained LNO-1/LNO-2 roughness Sample#13 variant with one start and a 60-iteration limit in `jax_gradient_fit_lno2_roughness/single_60iter`. The initial NumPy/JAX objective was `0.0233697`; the best NumPy/JAX validation objective improved to `0.012808`. SciPy terminated with status 2 (`ABNORMAL`) after 56 iterations and 227 function evaluations. Best parameters pinned `top_lno_layer1_thickness` at its 1 Angstrom lower bound, `sto_thickness_delta` and `lno_thickness_delta` at 0 Angstrom, and `thickness_transition_repeat` at 39; derived roughnesses were `lno1_roughness = 0.506911 A` and `lno2_roughness = 0.516942 A`, both satisfying the thickness constraint.
- 2026-06-19: Added reusable concentration-depth plotting helpers to `src/swxps/profiles.py`: `sample_layer_concentration_profiles` for per-layer chemistry such as Ni-free LNO-1, and `plot_vertical_concentration_profiles` for roughness-graded shaded profiles with depth on the y-axis. Exported both from `swxps`. Added `plot_single_60iter_vertical_concentration.py` under the Sample#13 constrained-roughness variant and generated `sample13_single60_vertical_concentration_profiles.png/csv` in `jax_gradient_fit_lno2_roughness/single_60iter`. The final plot uses separate shaded La/Ni/C tracks with layer labels only in the right-side strip to avoid text/profile overlap.
- 2026-06-19: Tweaked the vertical concentration plot styling after review. Added low-saturation default element colors, a dashed layer-box mode, and a categorical-strip single-panel mode to `plot_vertical_concentration_profiles`. Regenerated the Sample#13 constrained-roughness `single_60iter` concentration map for the top 50 Angstrom only as `sample13_single60_top50_vertical_concentration_profiles.png/csv`; the plot now uses one panel with La/Ni/C categorical shade strips, dashed layer interface boxes, depth increasing downward, and no grey overlap from stacked transparent profiles.
- 2026-06-19: Fixed a profile-grading artifact after the Sample#13 top-50-A plot showed apparent C inside LNO. Cause: `graded_layer_property_at_depth` allowed a rough interface window to overwrite values outside the two adjacent nominal layers, so the large C-surface roughness tail reappeared deeper in the LNO region. Updated the function to apply each interface grading only to depths nominally belonging to the adjacent above/below layers, then regenerated `sample13_single60_top50_vertical_concentration_profiles.png/csv` with low-saturation categorical strips and color-coded dashed layer boxes. Verified with `python -m pytest tests/test_xps.py tests/test_profiles.py` (10 passed).
- 2026-06-19: Added `src/swxps/result_exports.py` with utility functions `save_fit_curve_data_csv` and `save_optimized_stack_csv` for durable fitted-curve and optimized-stack exports. Added `examples/LNO_STO_LNO_case_Sample#13/collect_best_results_so_far.py`, created `examples/LNO_STO_LNO_case_Sample#13/best_results_so_far`, and collected the current best Sample#13 constrained-roughness artifacts there. The folder now contains `best_fit_experiment_and_simulation.csv`, `optimized_stack_layers.csv`, the best fit/stack/convergence/superlattice/profile figures, summary files, and validation contribution CSVs.
- 2026-06-19: Added and ran `examples/LNO_STO_LNO_case_Sample#12/collect_best_results_so_far.py`, following the Sample#13 collector structure. It rebuilds the clean single-run best simulation, exports `best_fit_experiment_and_simulation.csv` and `optimized_stack_layers.csv`, and copies the current best Sample#12 fit plots, summaries, JAX contributions, NumPy validation contributions, and superlattice profile into `examples/LNO_STO_LNO_case_Sample#12/best_results_so_far`.
- 2026-06-19: Updated the Sample#12 best-results collector to use the shared `plot_vertical_concentration_profiles` utility. The collector now exports `sample12_clean50_top30_vertical_concentration_profiles.png` and `.csv` in `examples/LNO_STO_LNO_case_Sample#12/best_results_so_far` with C, La, and Ni depth profiles from the optimized top-30-A stack region; Sr and Ti are intentionally omitted because STO is below this depth window.
- 2026-06-20: Planned a separate Sample#13 gradient-fit experiment excluding La 4d. Goal: fit only reflectivity, C 1s, and Ni 3p in a new `jax_gradient_fit_without_la4d` folder while fixing the C/LNO-1 roughness to zero and fitting the LNO-1/LNO-2 roughness as a fraction of LNO-1 thickness. Physics background: C 1s emits from the surface C layer, Ni 3p emits only from LNO-2 because LNO-1 is treated as Ni-free, and La 4d is omitted from both the objective and diagnostics. Files to create/modify: the new Sample#13 fitting folder/script and this plan. Validation: compile and run the setup check, verify matching finite NumPy/JAX objectives with exactly three contributions, then run one 60-iteration start and inspect the saved summary and constraints.
- 2026-06-20: Implemented and ran the Sample#13 reflectivity + C 1s + Ni 3p gradient variant in jax_gradient_fit_without_la4d. The model fixes LNO-1 upper-interface roughness at 0 A, omits La 4d data and simulation entirely, emits Ni 3p from LNO-2 only, and constrains LNO-2 roughness to fraction times LNO-1 thickness. A first boundary-start diagnostic stopped before iteration 1 and was preserved as boundary_start_failed; an interior-start single run completed all 60 iterations in single_60iter, improving the NumPy/JAX objective from 0.00813137 to 0.00699815. It stopped at the iteration limit with LNO-1 thickness 1 A, LNO-2 roughness 0.562491 A, and matching NumPy/JAX validation.
- 2026-06-20: Planned a Sample#13 forward-only depth-sensitivity comparison using the latest reflectivity + C 1s + Ni 3p fitted stack. Goal: compare the current surface-selected La 4d and Ni 3p RCs with RCs integrated over every chemically eligible LNO layer, without changing or refitting the stack. Physics background: La emits from all LNO layers; Ni emits from every LNO layer except the Ni-free LNO-1 cap. The same electric field, IMFP attenuation, roughness grading, angle offset, and normalization are used in both calculations. Files to create: a comparison script plus overlay, difference, curve CSV, and summary CSV outputs under the latest Sample#13 fit folder. Validation: verify layer-index selections, finite normalized/raw curves, quantify RMS/max RC differences and deeper-layer raw-signal fractions, and inspect both figures.
- 2026-06-20: Completed the Sample#13 surface-selected versus whole-LNO-stack RC forward comparison using the latest fitted stack. La 4d used 2 surface LNO layers versus all 43 LNO layers; Ni 3p used LNO-2 only versus 42 LNO layers, excluding the Ni-free LNO-1 layer. Buried layers contributed 2.868% of mean raw La signal and 3.735% of mean raw Ni signal, but separate normalization reduced the RC-shape changes to RMS 0.001302 and 0.001582. Whole-stack predictions increased experimental MSE by 7.18% for La and 4.91% for Ni. Saved reproducible overlay/difference PNGs and curve/summary CSVs in the latest single_60iter folder.
- 2026-06-20: Planned a Sample#13 bounded TRF least-squares fit in a new jax_least_squares_without_la4d folder. Goal: fit the latest reflectivity + C 1s + Ni 3p model with the newly added least-squares optimizer while preserving LNO-1 roughness fixed at 0 A, LNO-2 roughness constrained by LNO-1 thickness, and near-surface C/Ni emission selection. Physics background: the concatenated residual vector uses weighted pointwise log10 reflectivity residuals and weighted normalized RC residuals, so its squared norm equals the existing joint objective. Adaptive roughness changes effective-stack shapes, therefore use the validated exact high-level JAX forward model with a cached central finite-difference residual Jacobian. Files to create: a dedicated runner and single_run outputs including history/convergence, contributions, summary, fitted curves, optimized stack, covariance/uncertainty, best-fit, stack, and superlattice figures. Validation: compile, confirm initial residual norm matches the high-level objective, run bounded TRF, compare JAX/NumPy objectives, verify roughness constraints and finite outputs, and inspect generated artifacts.
- 2026-06-20: Implemented and ran the Sample#13 bounded TRF least-squares workflow in jax_least_squares_without_la4d/single_run. The exact high-level residual norm matched the existing objective within 2.2e-11 at setup. Starting from the latest gradient result nudged inside bounds, TRF converged by xtol in 31 residual and 13 Jacobian evaluations, improving the NumPy/JAX objective from 0.00699926 to 0.00672052; NumPy and JAX validation agreed to numerical precision. The result keeps LNO-1 roughness fixed at 0 A and gives LNO-2 roughness 0.418447 A for a 1.00007 A LNO-1 layer. Generated best-fit, convergence, stack, and superlattice PNGs plus history, contributions, summary, fitted curves, optimized stack, covariance, and parameter-uncertainty CSVs. All PNGs passed Pillow verification and tests/test_jax_least_squares.py passed (2 tests). Covariance uncertainties are large for several correlated stack parameters and multiple parameters remain effectively bound-limited.
- 2026-06-20: Planned a clean rerun of the Sample#13 TRF least-squares workflow with La 4d added to reflectivity, C 1s, and Ni 3p. Preserve the Ni-free LNO-1 chemistry: La 4d emits from LNO-1 and LNO-2, while Ni 3p emits only from LNO-2. Recompute the automatic reflectivity weight against all three RC residual blocks. Replace the ambiguous twin-axis superlattice plot with a three-panel profile read directly from the optimized stack: STO/LNO layer thicknesses, period thickness, and roughness. Add parameter_positions.png/csv showing every fitted parameter normalized to its allowed range and highlighting the lower/upper 1% zones. Clear the previous single_run generated files before execution, then regenerate and validate the complete artifact set.
- 2026-06-20: Completed the clean Sample#13 all-RC TRF rerun in jax_least_squares_all_rcs/single_run after clearing prior generated outputs and renaming the formerly without_la4d workflow. The fit includes reflectivity, C 1s, Ni 3p, and La 4d; La emits from LNO-1/LNO-2 and Ni from LNO-2 only. TRF converged by xtol in 24 residual and 10 Jacobian evaluations, improving objective 0.00884637 to 0.00837042 with NumPy/JAX agreement. Replaced the twin-axis profile with stack-derived superlattice_profile.png/csv; the optimized period is about 32.09575 A and nearly constant because both thickness deltas fitted to zero. Added parameter_positions.png/csv with normalized range positions and 1% bound highlighting. Near-bound parameters are lno2_roughness_fraction, both thickness deltas, both first-repeat roughnesses, and both angle offsets. All five PNGs passed Pillow verification; focused least-squares/XPS/profile tests passed (12 tests, 3 existing trapz warnings).
- 2026-06-20: Planned a tighter long Sample#13 all-RC TRF refinement and conditional best-result promotion. Restart from the current single_run optimum, increase max_nfev from 80 to 160, and tighten ftol/xtol to 1e-12 and gtol to 1e-9. Remove the superlattice periodicity plot and CSV from the runner and generated outputs. Compare the longer candidate against the existing best_results_so_far using the existing best reflectivity weight and the same reflectivity/C 1s/Ni 3p/La 4d raw residuals. Only if the common weighted objective improves, delete the old best-results files and rebuild the folder with fitted curves, optimized stack, best-fit/convergence/stack/parameter-range figures, summaries, contributions, covariance/uncertainty/history CSVs, and a fresh top-30-A concentration plot; omit the periodicity plot.
- 2026-06-20: Completed the tighter Sample#13 all-RC TRF refinement in jax_least_squares_all_rcs/long_run, restarting from single_run with max_nfev 160, ftol/xtol 1e-12, and gtol 1e-9. It converged by xtol in 24 residual and 8 Jacobian evaluations. Under the original saved-best reflectivity weight, the long-run common objective is 0.01034516 versus 0.01280799 for the old best, a 19.2288% improvement, and 0.0346% better than the shorter TRF candidate. Removed superlattice/periodicity plot generation and files. Because the candidate improved, cleared best_results_so_far and rebuilt it with stable fitted-curve/stack CSVs, prefixed best-fit/convergence/stack/parameter-position figures, summaries, contributions, history, covariance/uncertainty/position CSVs, selection_comparison.csv, and a fresh top-30-A concentration PNG/CSV. Updated the top-level collector to use conditional TRF promotion. Verified no periodicity artifacts remain; all promoted PNGs are valid and 12 focused tests passed with 3 existing trapz warnings.
- 2026-06-20: Planned a Sample#13 all-RC TRF expanded-bounds rerun. Widen reflectivity_angle_offset and rc_angle_offset from [-0.30, 0.30] to [-0.35, 0.35] degrees. Widen sto_thickness_start and lno_thickness_start from [13.0, 18.5] to [12.5, 19.0] Angstrom, leaving thickness-delta bounds unchanged. Restart from the promoted long_run parameters, use the tighter 160-evaluation TRF settings, write to expanded_bounds_run, preserve the no-periodicity-output decision, and compare residuals/bound positions against the promoted best.
- 2026-06-20: Completed the Sample#13 expanded-bounds all-RC TRF run in jax_least_squares_all_rcs/expanded_bounds_run. Local least-squares bounds are now [-0.35, 0.35] degrees for both angle offsets and [12.5, 19.0] A for both STO/LNO starting thicknesses; shared earlier models remain unchanged. Restarted from long_run with max_nfev 160 and tight tolerances. TRF converged by xtol in 25 residual and 3 Jacobian evaluations at objective 0.00832842. Fitted offsets stayed at 0.299437 and 0.297433 degrees, and STO/LNO starts at 13.70064 and 18.39748 A, so widening removed their bound flags but did not materially move the solution. Under the common saved-best weighting, expanded_bounds_run scored 0.01034562 versus promoted best 0.01034516, 0.0044% worse; best_results_so_far was therefore left unchanged. Verified outputs contain no periodicity artifacts, all four PNGs are valid, and focused least-squares tests passed (2 tests).
- Remaining: Review normalized XPS rocking curves against experimental examples before adding cross sections, p-polarization, or online optical-constant database features. Continue validating BO recovery quality before moving to experimental data.
- 2026-06-20: Planned a synthetic C/LNO/STO nonlinear least-squares versus L-BFGS-B comparison. Goal: run the new bounded TRF optimizer and the corrected gradient optimizer from the same default initial values against identical stride-2 synthetic reflectivity plus four-RC data, parameter bounds, auto dataset weights, and exact high-level JAX-backed physics. Because adaptive roughness slicing changes effective-stack shapes, both methods will use the same central finite-difference scale: TRF differentiates the exact residual vector, while L-BFGS-B differentiates its squared norm. Files to create: `examples/synthetic_c_lno_sto/jax_least_squares_vs_gradient_20260620/*` and this plan. Validation: run each method in a fresh Python process, confirm its JAX and NumPy validation objectives agree, save histories/contributions/best-fit/stack figures, and report objective, parameter error, evaluation counts, and wall time in a combined summary.
- 2026-06-20: Completed the synthetic C/LNO/STO exact high-level least-squares versus gradient comparison in `jax_least_squares_vs_gradient_20260620`. Each optimizer ran in a fresh Python process from the same default start, with stride 2, auto dataset weights, the JAX high-level forward backend, and central finite differences over the adaptive roughness model. TRF converged in 45.248 s to NumPy/JAX objective `3.42758e-7` with normalized parameter RMSE `0.008723`; L-BFGS-B converged in 72.818 s to objective `4.50866e-7` with normalized parameter RMSE `0.075660`. NumPy and JAX validation objectives agreed to numerical precision. Focused optimizer tests passed (6 tests), and the folder contains the runner, per-method summaries/histories/contributions/fit figures/stack figures, and combined CSV/text/PNG comparisons.
- 2026-06-20: Planned and completed a Sample#12 bounded TRF least-squares refit in jax_least_squares_fit. The exact 227-point high-level JAX residual objective matched NumPy within 1.3e-11. TRF converged by xtol in 25 residual and 7 Jacobian evaluations, improving the common objective from 0.00390254275 to 0.00321527040 (17.6109%). The improvement guard promoted the candidate and replaced best_results_so_far with TRF fit, stack, convergence, superlattice, contribution, covariance, uncertainty, selection, and top-30-A C/La/Ni outputs. Focused least-squares/profile tests passed (5 tests).
- 2026-06-20: Planned a Sample#12 thin-carbon bounded TRF rerun. Override only carbon_thickness from its previous 2-15 A range to 1-5 A, retain the promoted cap3 model, data, reflectivity weight, RC weights, JAX forward backend, and all other bounds, and restart from the promoted result clipped just inside the new range. Save outputs in a separate carbon_1_5A_run folder, compare against the promoted objective under identical weighting, and promote only if the constrained candidate improves. Validate matching residual/JAX/NumPy objectives, carbon thickness within 1-5 A, finite outputs, and focused tests.
- 2026-06-20: Completed the Sample#12 thin-carbon TRF rerun in jax_least_squares_fit/carbon_1_5A_run. With carbon_thickness constrained to 1-5 A, residual/JAX/NumPy setup objectives agreed within 2.2e-11. TRF converged by xtol in 23 residual and 4 Jacobian evaluations to objective 0.00352130612 with carbon_thickness 4.95462 A and derived carbon roughness 3.18635 A. The C 1s residual improved, but reflectivity worsened enough that the common objective remained 9.52% above the promoted 0.00321527040 result; the candidate was therefore not promoted and best_results_so_far remained unchanged. All 4 PNGs verified and focused least-squares/profile tests passed (5 tests).
## Sample#12 matched RC normalization experiment (2026-06-21)

### Goal
Add an optional edge-polynomial normalization for simulated rocking curves and refit Sample#12 using the same 10%-per-edge quadratic procedure as the experimental RCs.

### Physics background
The current simulation divides each raw RC by one scalar mean over all angles, while experimental data are divided pointwise by a quadratic background fitted to both angle-window edges. Including the standing-wave peak in the simulated mean can reduce apparent modulation. The new mode divides simulated raw intensity by its own edge-fitted polynomial background without changing electric fields, attenuation, concentrations, or raw XPS integration.

### Files to create or modify
Modify src/swxps/preprocessing.py, simulation.py, simulation_jax.py, fitting.py, focused tests, and the Sample#12 TRF runner. Save the experiment in a separate edge-normalization run folder.

### Implementation steps
1. Add a shared mean/edge-polynomial RC normalization helper.
2. Thread the normalization choice through high-level NumPy/JAX requests and fitting problems while preserving mean normalization by default.
3. Add NumPy/JAX parity and preprocessing tests.
4. Refit Sample#12 from the promoted parameters with original carbon bounds and no automatic promotion.

### Tests and validation
Existing behavior must remain unchanged by default. Edge-normalized NumPy and JAX curves must agree, the experimental and simulated procedures must use the same edge fraction/order, residual/JAX/NumPy objectives must match, and generated outputs must remain finite.
- 2026-06-21: Completed the Sample#12 matched RC-normalization experiment. Added optional mean versus edge_polynomial normalization to the shared preprocessing helper, RockingCurveRequest, FittingProblem, and both NumPy/JAX high-level simulation paths; mean remains the default. Added helper/parity tests. Refit Sample#12 in jax_least_squares_fit/edge_polynomial_normalization_run with the original 2-15 A carbon range, 10%-per-edge quadratic simulated backgrounds, and promotion disabled because the objective convention changed. TRF converged by xtol in 22 residual and 4 Jacobian evaluations from 0.00342457446 to 0.00332310482. At identical promoted parameters, edge normalization worsened C1s MSE from 0.00113038 to 0.00172009 and reduced the fitted maximum from 1.12855 to 1.10760 versus experimental 1.21499; after refitting C1s MSE was 0.00165397. Therefore the amplitude mismatch is not caused by mean-versus-edge normalization. The promoted best remained unchanged. Full pytest passed (85 tests); all 4 run PNGs verified.
