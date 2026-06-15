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
- `examples/plot_lno_sto_reflectivity.py`
  - Example visualization for a LaNiO3/SrTiO3 multilayer using manually supplied optical constants.
- `examples/plot_lno_sto_field_profile.py`
  - Example visualization of standing-wave field intensity through a LaNiO3/SrTiO3 multilayer.
- `examples/plot_lno4d_rocking_curve.py`
  - Example normalized La 4d rocking curve for the LaNiO3/SrTiO3 multilayer.
- `examples/plot_lno_sto_stack_profile.py`
  - Example visualization of La, Ti, and O concentration profiles versus depth.
- `examples/README.md`
  - Short instructions for running the example.
- `examples/fit_lno_sto_synthetic_bo.py`
  - Synthetic Bayesian-optimization example using known LNO/STO-like parameters.
  - Recover a small parameter set from simulated reflectivity and/or SW-XPS rocking curves before using experimental data.

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
- Remaining: Review normalized XPS rocking curves against experimental examples before adding cross sections, p-polarization, or online optical-constant database features. Implement fitting first on synthetic data, then move to experimental data only after the objective and BO adapter are validated.
