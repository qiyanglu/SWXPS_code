# ProjectSpec JAX Least-Squares Synthetic C/LaNiO3/SrTiO3 Benchmark

This folder runs the existing synthetic C/LaNiO3/SrTiO3 (C/LNO/STO)
reflectivity plus SW-XPS rocking-curve benchmark through the YAML ProjectSpec
workflow.

It is intentionally a ProjectSpec YAML example that uses the internal
fixed-grid JAX residual builder:

- `project.yaml` owns the JAX least-squares variant: materials, OPC and IMFP
  paths, stack layer IDs/tags, core-level emitting layers, datasets, and fitting
  parameters.
- `project_simulate_only.yaml` uses fixed synthetic truth values and writes
  simulation/report outputs without fitting.
- `project_bo.yaml` uses the existing generic Bayesian-optimization backend as
  an optional black-box baseline.
- `run.optimizer.residual: "auto_fixed_grid"` builds the fixed-shape residual
  directly from the YAML stack, parameters, datasets, and slicing settings.
- `run_project.py` is the minimal script entry point.
- `quick_start.py` prints `swanx inspect`, validates, then runs the fit.

The YAML mirrors the existing synthetic case in
`benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py`:

- photon energy: `1000 eV`
- stack: `vacuum / C / [LaNiO3 / SrTiO3] x 20 / SrTiO3 substrate`
- OPC files: `data/OPC/C.dat`, `data/OPC/LaNiO3.dat`, `data/OPC/SrTiO3.dat`
- IMFP files: `data/IMFP/C.ANG`, `data/IMFP/LNO.ANG`, `data/IMFP/STO.ANG`
- datasets: local copy of `lno_sto_c_synthetic_data.csv`
- rocking-curve normalization: `edge_polynomial`, first/last 10 percent,
  polynomial order 2
- fitted parameters: carbon thickness, carbon roughness fraction, LNO/STO
  thicknesses, superlattice roughness, substrate roughness, and angle offset

ProjectSpec expressions support the exact original benchmark carbon-roughness
formula directly:

```text
carbon_roughness = 1 + carbon_roughness_fraction * (min(5, carbon_thickness) - 1)
```

## Install

From the repository root, install the optional pieces:

```bash
python -m pip install -e ".[project,least-squares,plot]"
```

## Run With The Script Interface

From anywhere:

```bash
python benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/run_project.py
```

For a more verbose first run:

```bash
python benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/quick_start.py
```

The ProjectSpec runner builds the fixed-grid residual internally from the YAML
stack, parameters, datasets, and slicing settings.

## Run With The CLI Interface

From the repository root:

```bash
swanx inspect benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/project.yaml
swanx validate benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/project.yaml
swanx run benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/project.yaml

swanx run benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/project_simulate_only.yaml
swanx run benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/project_bo.yaml
```

Or from this benchmark folder:

```bash
cd benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares
swanx inspect project.yaml
swanx validate project.yaml
swanx run project.yaml
swanx run project_simulate_only.yaml
swanx run project_bo.yaml
```

Equivalent module form is also available with `python -m swanx.cli ...`.

Outputs are written under:

```text
benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/runs/
```

Expected key files include:

```text
report.md
fit/fit_summary.json
fit/best_parameters.csv
fit/residuals.csv
optimizer/least_squares/status.json
optimizer/least_squares/convergence_history.csv
optimizer/least_squares/parameter_uncertainty.csv
plots/fit_overview.png
plots/reflectivity_fit.png
plots/rocking_curves_fit.png
plots/stack_schematic.png
plots/convergence.png
plots/parameter_uncertainty.png
plots/parameter_correlation.png
```

After a least-squares run, `run.outputs.identifiability: true` writes
`identifiability_analysis/` inside the run folder.

For `project_bo.yaml`, the BO-specific plot set includes:

```text
plots/convergence.png
plots/surrogate_slices.png
```

For `project_simulate_only.yaml`, the YAML omits datasets intentionally, so it
writes pure simulation outputs without `data/` or `fit/residuals.csv`; its
curve plots use `simulation_overview.png`, `reflectivity_simulation.png`, and
`rocking_curves_simulation.png`.
