# Roadmap

## Current focus

SWANX is stabilizing the user workflow around:

```text
data files -> swanx.io -> simulation requests / fitting data -> simulation + fitting + diagnostics
```

Unified slicing is the default high-level simulation path. JAX-based automatic
differentiation is the primary fitting strategy; Bayesian optimization remains
a baseline and robustness check.

## Near-term priorities

1. Keep README and maintained examples concise, executable, and based on
   `swanx`.
2. Validate the `data/OPC`, `data/IMFP`, and `data/curves` workflow on tutorial
   and representative case-study inputs.
3. Improve fixed-shape JAX least-squares fitting documentation.
4. Preserve NumPy/JAX numerical parity and source-location tests.
5. Continue validating RC preprocessing, weighting, angular offsets, parameter
   identifiability, and fitted structures.

## Future IO extensions

- richer experimental-data formats;
- export of fit results and best-fit curves;
- uncertainty/sigma propagation in downstream diagnostics and exports;
- optional adapters for additional optical-constant formats;
- eventual fitting source migration into `swanx.fitting.*` submodules.

## Deferred physics/features

- p-polarization;
- online optical-constant databases;
- expanded photoionization cross-section models;
- new optimizers without a demonstrated validation need;
- broad physical-kernel restructuring without a separate validated plan.
