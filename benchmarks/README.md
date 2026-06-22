# Benchmarks

- `performance/`: focused backend timing scripts.
- `synthetic_c_lno_sto/`: deterministic synthetic reflectivity/SW-XPS fitting benchmark.

Benchmark-generated output belongs under `runs/synthetic_c_lno_sto`.

```powershell
python benchmarks/performance/profile_forward_workflow.py
python benchmarks/performance/benchmark_slicing_strategy.py
python benchmarks/performance/benchmark_jax_reflectivity.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```

`profile_forward_workflow.py` is the primary stage-by-stage NumPy baseline. It
reports table loading, stack construction, roughness discretization,
reflectivity, fields/SW-XPS, and the complete fitting objective separately.



`benchmark_slicing_strategy.py` checks thin-surface convergence against a dense 0.1 Angstrom reference and verifies fixed shape across a thickness sweep.
