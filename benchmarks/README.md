# Benchmarks

- `performance/`: focused backend timing scripts.
- `synthetic_c_lno_sto/`: deterministic synthetic C/LaNiO3/SrTiO3
  (C/LNO/STO) reflectivity/SW-XPS fitting benchmark.
- `synthetic_c_lno_sto/projectspec_jax_least_squares/`: the same synthetic case routed through the YAML ProjectSpec workflow with the internal fixed-grid JAX least-squares residual.

Maintained synthetic fitting benchmarks use the same rocking-curve convention
as ProjectSpec: edge-polynomial normalization with the first and last 10
percent of each RC and polynomial order 2. Peak-exclusion masks remain only for
legacy mean-normalized or field-inspection scripts.

Legacy benchmark-generated output belongs under `runs/synthetic_c_lno_sto`. The ProjectSpec benchmark writes project-local outputs under `benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/runs/`.

```powershell
python benchmarks/performance/profile_forward_workflow.py
python benchmarks/performance/benchmark_slicing_strategy.py
python benchmarks/performance/benchmark_jax_reflectivity.py
python benchmarks/synthetic_c_lno_sto/fit_reflectivity_rc_bo.py --generate-only
python benchmarks/synthetic_c_lno_sto/compare_slicing_strategies.py
python benchmarks/synthetic_c_lno_sto/projectspec_jax_least_squares/quick_start.py
```

`profile_forward_workflow.py` is the primary stage-by-stage NumPy baseline. It
reports table loading, stack construction, roughness discretization,
reflectivity, fields/SW-XPS, and the complete fitting objective separately.



`benchmark_slicing_strategy.py` checks thin-surface convergence against a dense 0.1 Angstrom reference and verifies fixed shape across a thickness sweep.
