# ProjectSpec Quickstart Examples

These YAML files are copy-pasteable starting points for SWANX ProjectSpec
workflows. Paths are written relative to this folder, so validation works from
any current working directory.

The examples use the same introductory case as the synthetic benchmark:
vacuum / C / [LaNiO3/SrTiO3 (LNO/STO)]x20 / SrTiO3 (STO) at 1000 eV.
Data-overlay and fitting examples read the benchmark CSV with reflectivity plus
La 4d, O 1s, Ti 2p, and C 1s rocking curves, so the beginner YAML workflow
matches the Python examples and benchmark scripts.

## Files

- `minimal_simulate_only.yaml`: C-capped 20-repeat LNO/STO superlattice,
  simulation and plots only.
- `multilayer_repeat.yaml`: the same C/LNO/STO stack with explicit repeat-block
  tags and `repeat_index`.
- `compare_with_data.yaml`: simulation-only project that overlays the synthetic
  reflectivity and four rocking-curve datasets and writes residuals.
- `fit_jax_least_squares_placeholder.yaml`: shows the JAX least-squares factory
  settings required for fitting. This file validates, but it is not runnable
  until you provide `fit_factory.py` with `build_residual` next to the YAML file.
- `bo_optional_baseline.yaml`: small Bayesian-optimization baseline config. BO
  is optional, not the recommended default and not a fallback.

## Useful Commands

```bash
swanx validate examples/01_quickstart_projectspec/minimal_simulate_only.yaml
swanx run examples/01_quickstart_projectspec/minimal_simulate_only.yaml
swanx validate examples/01_quickstart_projectspec/compare_with_data.yaml
```

For the best beginner experience, start with:

```bash
swanx init my_project
python my_project/run_project.py
```

That generated project is a self-contained JAX least-squares fitting starter.
Use the YAML files in this folder when you want smaller simulation-only or
configuration-focused examples.
