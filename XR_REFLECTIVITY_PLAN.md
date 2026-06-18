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
- 2026-06-18: Planned a refresh of `examples/synthetic_c_lno_sto` after the fitting and simulation APIs changed substantially. Goal: archive old synthetic C/LNO/STO BO scripts/artifacts and write a current joint reflectivity-plus-RC BO driver for studying robust BO settings. Physics background: the synthetic stack is vacuum/C/[LNO/STO]x20/STO at 1000 eV, with reflectivity fitted in log space and normalized La 4d, O 1s, Ti 2p, and C 1s rocking curves fitted in linear intensity after off-peak normalization. Files to modify: `examples/synthetic_c_lno_sto/*` and this plan. Implementation steps: move old example files into a subfolder, make a self-contained stack/data/problem builder, add BO controls for staged versus joint fitting and dataset weighting, save history/convergence/best-fit/schematic diagnostics, and verify with a short smoke run. Tests/validation: the script should construct the joint fitting problem, run a small BO job, recover physically plausible parameters near the synthetic truth, and keep reflectivity plus RC simulations finite.
- 2026-06-18: Refreshed `examples/synthetic_c_lno_sto`: archived previous scripts/artifacts into `previous_fitting`, added `fit_reflectivity_rc_bo.py` with self-contained synthetic data generation, auto reflectivity-weight balancing, direct and staged BO modes, and diagnostic outputs. Regenerated the 161-point synthetic CSV/preview and verified the script with `python -m py_compile` plus a 4-call end-to-end BO smoke run.
- 2026-06-18: Ran synthetic C/LNO/STO BO comparisons. A 200-call direct single-seed run reached objective `1.931e-4` with near-zero angle offset and was archived under `single_run_outputs`. A staged 2-start run with 100 calls per stage/start reached `5.549e-4`, getting locked into a positive angle-offset/thickness basin; an alternate direct seed-112 run reached `1.147e-3`. Distinct data colors were added for each RC in the synthetic best-fit plot.
- 2026-06-18: Cleaned the synthetic example directory by moving staged/multi-start comparison outputs into `staged_multistart_outputs`. Ran another direct single BO with 240 calls, 70 initial points, and seed 24; it reached objective `3.919e-4` at evaluation 142, better than staged/multi-start but not better than the archived 200-call seed-12 direct run (`1.931e-4`).
- 2026-06-18: Planned an automated multi-seed synthetic C/LNO/STO BO driver. Goal: reduce seed luck in the noiseless synthetic benchmark by running repeated direct BO fits, saving per-seed artifacts, writing a summary CSV, and promoting the best run to stable output names. Files to modify: `examples/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py`, synthetic example output folders, and this plan. Validation: compile the script and run a small multi-seed smoke test that creates per-seed histories plus a summary and promoted best-fit artifacts.
- 2026-06-18: Implemented the synthetic multi-seed BO driver with `--multi-seed`, `--seeds`, `--seed-start`, `--seed-count`, and `--multi-seed-dir`. Each seed writes artifacts in its own subfolder, a summary CSV records objective/contribution/parameter/timing columns, and the best seed is promoted to stable `*_best_*` artifacts in the multi-seed output folder. Verified with `python -m py_compile` and a two-seed four-call smoke run, then removed smoke artifacts.
- Remaining: Review normalized XPS rocking curves against experimental examples before adding cross sections, p-polarization, or online optical-constant database features. Continue validating BO recovery quality before moving to experimental data.
