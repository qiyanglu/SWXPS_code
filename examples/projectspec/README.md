# ProjectSpec Examples

These YAML files are copy-pasteable starting points for SWANX ProjectSpec
workflows. Paths are written relative to this folder, so validation works from
any current working directory.

## Examples

- `minimal_simulate_only.yaml`: single LNO film on STO substrate, no datasets,
  simulation and plots only.
- `multilayer_repeat.yaml`: compact LNO/STO repeat block with layer tags and
  `repeat_index`.
- `compare_with_data.yaml`: simulation-only project that overlays tutorial
  reflectivity and La 4d rocking-curve data and writes residuals.
- `fit_jax_least_squares_placeholder.yaml`: shows the JAX least-squares factory
  settings required for fitting. This file validates, but it is not runnable
  until you provide `fit_factory.py` with `build_residual` next to the YAML file.
- `bo_optional_baseline.yaml`: small Bayesian-optimization baseline config. BO
  is optional, not the recommended default and not a fallback.

## Useful Commands

```bash
swanx validate examples/projectspec/minimal_simulate_only.yaml
swanx run examples/projectspec/minimal_simulate_only.yaml
swanx validate examples/projectspec/compare_with_data.yaml
```

For the best beginner experience, start with:

```bash
swanx init my_project
python my_project/run_project.py
```