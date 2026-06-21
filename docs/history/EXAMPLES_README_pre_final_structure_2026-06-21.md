# Examples

This directory contains maintained, reproducible scripts. Generated optimizer
runs are written under `artifacts/runs` and historical experiments are preserved
under `artifacts/archive`.

## Tutorial examples

```powershell
python examples/reflectivity/plot_lno_sto_reflectivity.py
python examples/roughness/compare_lno_sto_roughness.py
python examples/fields/plot_lno_sto_field_profile.py
python examples/xps/plot_lno_la4d_rocking_curve.py
python examples/profiles/plot_lno_sto_stack_profile.py
python benchmarks/performance/benchmark_jax_reflectivity.py
```

## Synthetic fitting benchmark

The current self-contained C/LNO/STO driver is:

```powershell
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

Fit outputs are written to `runs/synthetic_c_lno_sto/current`.

## Experimental case studies

- `LNO_STO_LNO_case_Sample#12`: raw data, current runners, support modules, and one canonical best-result export.
- `LNO_STO_LNO_case_Sample#13`: raw data, current runners, support modules, and one canonical best-result export.
- `sample12_sample13_comparison`: comparison of the two canonical results.

Each case directory keeps inputs and maintained scripts. Optimizer attempts,
failed runs, and superseded outputs no longer live in the active example tree.
