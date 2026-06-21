# Benchmarks

- `performance/`: focused backend timing scripts.
- `synthetic_c_lno_sto/`: deterministic synthetic reflectivity/SW-XPS fitting benchmark.

Benchmark-generated output belongs under `runs/synthetic_c_lno_sto`.

```powershell
python benchmarks/performance/benchmark_jax_reflectivity.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
```
