# Examples

Each example lives in its own subfolder. New examples should follow the same
pattern and write generated files next to the script that creates them.

In every stack, the first layer is vacuum and the last layer is the
semi-infinite substrate. In `StackTemplate`, the last layer must therefore have
zero thickness; its roughness still describes the roughness of the substrate's
upper interface.

## LaNiO3/SrTiO3 Reflectivity

Run:

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
```

This creates:

```text
examples/reflectivity/lno_sto_reflectivity.png
```

The example stack is:

```text
vacuum / [LaNiO3 / SrTiO3] x 20 / SrTiO3 substrate
```

The script uses:

- Energy: 3000 eV
- LaNiO3 thickness: 20 Angstrom
- SrTiO3 thickness: 20 Angstrom
- Bilayer period: 40 Angstrom
- Interface roughness: 3 Angstrom
- Grazing angle scan: 0.05 to 5.0 degrees

The optical constants are read from `OPC/LaNiO3.dat` and `OPC/SrTiO3.dat`.
The requested energy must lie within the table range.

## Roughness Comparison

Run:

```powershell
python examples/roughness/compare_lno_sto_roughness.py
```

This creates:

```text
examples/roughness/lno_sto_roughness_comparison.png
```

It plots the same LaNiO3/SrTiO3 stack with sharp interfaces and with 3 Angstrom
RMS roughness, so the damping of fringes and Bragg features can be seen
directly.

## Roughness Profile Shape Comparison

Run:

```powershell
python examples/roughness/compare_lno_sto_roughness_profiles.py
```

This creates:

```text
examples/roughness/lno_sto_roughness_profile_comparison.png
```

It calculates the same 3 Angstrom rough LNO/STO stack twice, once with
error-function roughness grading and once with RMS-matched linear roughness
grading. The top panel compares reflectivity, and the lower panel plots the
fractional difference between the two models.

## Electric Field Profile

Run:

```powershell
python examples/fields/plot_lno_sto_field_profile.py
```

This creates:

```text
examples/fields/lno_sto_field_profile.png
```

It uses 1000 eV photons, scans around the first-order Bragg condition, finds
the local reflectivity maximum, and plots a contour map of graded-roughness
transfer-matrix electric-field intensity `|E|^2` as a function of depth and
grazing incidence angle.

## Top 100 Angstrom Off-Peak Fields

Run:

```powershell
python examples/fields/plot_lno_sto_top100_offpeak_fields.py
```

This creates:

```text
examples/fields/lno_sto_top100_offpeak_fields.png
```

It uses the same 20 Angstrom / 20 Angstrom LNO/STO stack at 1000 eV, but plots
only the top 100 Angstrom and excludes the peak angle from the graded-roughness
transfer-matrix field profiles.

## La 4d, O 1s, and Ti 2p Rocking Curves

Run:

```powershell
python examples/xps/plot_lno_la4d_rocking_curve.py
```

This creates:

```text
examples/xps/lno_la4d_o1s_ti2p_rocking_curves.png
```

It simulates normalized La 4d, O 1s, and Ti 2p standing-wave XPS rocking curves
for the LaNiO3/SrTiO3 stack. The photon energy is 1000 eV. La 4d uses a 105 eV
binding energy, O 1s uses a 530 eV binding energy, and Ti 2p uses a 460 eV
binding energy. Cross sections are treated as constants; IMFP values are
interpolated from `IMFP`. The RCs are separated into vertically aligned
subplots so their angular phases can be compared directly.

The example uses the high-level simulation API:

```python
simulate_reflectivity(ReflectivityRequest(...))
simulate_rocking_curves(RockingCurveRequest(...))
```

Those request/result objects are intended to be the stable interface for later
optimization or fitting code.

## Synthetic C/LNO/STO Data for Fitting

Run:

```powershell
python examples/synthetic_c_lno_sto/generate_lno_sto_c_synthetic_data.py
```

This creates:

```text
examples/synthetic_c_lno_sto/lno_sto_c_synthetic_data.csv
examples/synthetic_c_lno_sto/lno_sto_c_synthetic_data.png
```

The synthetic stack is:

```text
vacuum / C / [LaNiO3 / SrTiO3] x 20 / SrTiO3 substrate
```

The C layer is 10 Angstrom thick with 2 Angstrom RMS top-surface roughness.
The LNO and STO layers are 20 Angstrom thick. The CSV contains grazing angle,
reflectivity, and normalized La 4d, O 1s, Ti 2p, and C 1s rocking curves. This
file is intended as a deterministic synthetic dataset for testing later
Bayesian-optimization fitting workflows.

To run a small Bayesian-optimization fit against this synthetic dataset:

```powershell
python examples/synthetic_c_lno_sto/fit_lno_sto_c_synthetic_bo.py --n-calls 30 --n-initial-points 8 --stride 8
```

This creates:

```text
examples/synthetic_c_lno_sto/lno_sto_c_bo_history.csv
examples/synthetic_c_lno_sto/lno_sto_c_bo_convergence.png
examples/synthetic_c_lno_sto/lno_sto_c_bo_best_fit.png
examples/synthetic_c_lno_sto/lno_sto_c_bo_surrogate_slices.png
```

The fit script defines bounded parameters with initial guesses for C thickness,
C roughness, common LNO thickness, common STO thickness, common superlattice
roughness, substrate/interface roughness, and angle offset. Increase `--n-calls`
and decrease `--stride` for a more expensive but more complete fit.
The surrogate-slice plot shows the Gaussian-process objective mean and standard
deviation for each parameter while holding the other parameters at the BO best
values.

To run the staged multi-start version:

```powershell
python examples/synthetic_c_lno_sto/fit_lno_sto_c_synthetic_staged_bo.py --n-calls 20 --n-initial-points 6 --n-starts 3 --stride 4
```

This creates:

```text
examples/synthetic_c_lno_sto/lno_sto_c_staged_bo_summary.csv
examples/synthetic_c_lno_sto/lno_sto_c_staged_bo_final_history.csv
examples/synthetic_c_lno_sto/lno_sto_c_staged_bo_final_convergence.png
examples/synthetic_c_lno_sto/lno_sto_c_staged_bo_best_fit.png
```

The staged driver fits smaller parameter groups first, carries the best values
forward as fixed values, and repeats each stage from independent BO random
seeds. This is intended for RC-heavy fits where a single seven-parameter GP run
can spend too much effort exploring unhelpful combinations early on.

To draw a schematic of the fitted stack from the saved BO history:

```powershell
python examples/synthetic_c_lno_sto/plot_fitted_stack_schematic.py
```

This creates:

```text
examples/synthetic_c_lno_sto/lno_sto_c_bo_stack_schematic.png
```

The schematic shows the top layers, the bottom substrate region, an ellipsis for
collapsed repeated layers, annotated thicknesses/roughnesses, incident and
diffracted x-ray waves, and a stylized standing-wave pattern.

The intended fitting API is:

```python
stack_template = StackTemplate(
    energy_ev=1000.0,
    base_dir=repo_root,
    parts=(
        LayerTemplate.vacuum(),
        LayerTemplate.from_file("C", "OPC/C.dat", "carbon_thickness", "carbon_roughness"),
        SuperlatticeTemplate(
            repeats=20,
            period=(
                LayerTemplate.from_file("LNO", "OPC/LaNiO3.dat", "lno_thickness", "roughness"),
                LayerTemplate.from_file("STO", "OPC/SrTiO3.dat", "sto_thickness", "roughness"),
            ),
        ),
        LayerTemplate.from_file("STO", "OPC/SrTiO3.dat", 0.0, "substrate_roughness"),
    ),
)

parameters = (
    FitParameter("carbon_thickness", 0.0, 20.0, "Angstrom", initial=5.0),
    FitParameter("carbon_roughness", 0.0, 5.0, "Angstrom", initial=1.0),
    FitParameter("lno_thickness", 15.0, 25.0, "Angstrom", initial=20.0),
    FitParameter("sto_thickness", 15.0, 25.0, "Angstrom", initial=20.0),
    FitParameter("roughness", 0.0, 5.0, "Angstrom", initial=2.0),
    FitParameter("substrate_roughness", 0.0, 5.0, "Angstrom", initial=2.0),
    FitParameter("angle_offset", -0.2, 0.2, "deg", initial=0.0),
)

reflectivity = ReflectivityData(
    "reflectivity",
    angles,
    measured_reflectivity,
    weight=1.0,
)
rocking_curves = (
    RockingCurveData("C 1s", angles, measured_c1s_rc, weight=5.0),
)

problem = FittingProblem(
    parameters=parameters,
    stack_builder=stack_template.builder(),
    photon_energy_ev=1000.0,
    reflectivity=reflectivity,
    rocking_curves=rocking_curves,
    core_levels=core_level_requests,
    angle_offset_parameter="angle_offset",
)

result = run_bayesian_fit(problem, BayesianOptimizationSettings(n_calls=50))
save_fit_history_csv("history.csv", result.history, parameters)
plot_fit_convergence("convergence.png", result.history)
```

A staged run uses the same `FittingProblem` and `FitParameter` objects:

```python
parameter_by_name = {parameter.name: parameter for parameter in parameters}

stages = (
    FitStage(
        "period_and_angle",
        (
            parameter_by_name["lno_thickness"],
            parameter_by_name["sto_thickness"],
            parameter_by_name["angle_offset"],
        ),
    ),
    FitStage(
        "surface",
        (
            parameter_by_name["carbon_thickness"],
            parameter_by_name["carbon_roughness"],
            parameter_by_name["angle_offset"],
        ),
    ),
    FitStage(
        "roughness",
        (
            parameter_by_name["superlattice_roughness"],
            parameter_by_name["substrate_roughness"],
            parameter_by_name["angle_offset"],
        ),
    ),
    FitStage("final_all", parameters),
)

staged = run_staged_multistart_bayesian_fit(
    problem,
    stages,
    BayesianOptimizationSettings(n_calls=30, n_initial_points=8),
    n_starts=3,
    random_seed=11,
)
save_staged_fit_summary_csv("staged_summary.csv", staged, parameters)
```

Each dataset has its own `weight`, so reflectivity and each rocking curve can be
weighted independently.

## Stack Concentration Profile

Run:

```powershell
python examples/profiles/plot_lno_sto_stack_profile.py
```

This creates:

```text
examples/profiles/lno_sto_stack_profile.png
```

It plots La, Ti, and O relative concentration versus depth for the 20 Angstrom /
20 Angstrom LNO/STO stack. The broadened transitions show the same
error-function roughness model used by the SW-XPS simulations.

## Roughness Profile Concentration Comparison

Run:

```powershell
python examples/profiles/compare_lno_sto_concentration_profiles.py
```

This creates:

```text
examples/profiles/lno_sto_concentration_profile_comparison.png
```

It compares La and Ti concentration profiles for the same 3 Angstrom rough
LNO/STO stack using error-function and RMS-matched linear roughness grading.
