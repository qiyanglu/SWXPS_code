# Roadmap

## Current focus

SWANX is stabilizing two complementary user workflows:

```text
swanx init my_project -> inspect/edit project.yaml -> run_project.py -> project-local report folder
```

and, for custom Python work:

```text
data files -> swanx.io -> simulation requests / fitting data -> simulation + fitting + diagnostics
```

The YAML ProjectSpec path is the main human-editable project input. Default
initializer projects are self-contained from packaged tutorial data, with
`minimal`, `multilayer`, and `fit-demo` initializer choices. The maintained
examples now include `examples/04_fitting/projectspec_jax_least_squares/`, a
runnable ProjectSpec JAX least-squares fitting project that mirrors the default
init tutorial. Direct Python APIs remain the right surface for custom
fixed-shape JAX fitting, new diagnostics, and lower-level simulation
experiments.

Unified slicing is the default high-level simulation path. JAX-based automatic
differentiation and least-squares are the recommended fitting strategy for
differentiable fixed-shape workflows; Bayesian optimization remains an optional
global black-box baseline/robustness check.

## Near-term priorities

1. Keep README, user guide, and maintained examples concise,
   executable, based on `swanx`, and aligned with the synthetic
   C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 teaching case.
2. Expand ProjectSpec examples only when they reuse existing IO, simulation,
   and fitting APIs without introducing new physics paths.
3. Keep fixed-shape JAX least-squares fitting docs and examples aligned with
   ProjectSpec callback factories.
4. Preserve NumPy/JAX numerical parity and source-location tests.
5. Continue validating RC preprocessing, shared experimental/simulated
   normalization, weighting, angular offsets, parameter identifiability, and
   fitted structures.
6. Keep s/p/mixed polarization examples and tests aligned with representative
   multilayer and superlattice cases.

## Future IO and report extensions

- richer experimental-data formats when real lab conventions require them;
- richer ProjectSpec examples and report exports beyond the current CSV, JSON,
  Markdown, and plot layout;
- uncertainty/sigma propagation in downstream diagnostics and exports;
- optional adapters for additional optical-constant formats;
- maintain the focused `swanx.fitting.*` backend layout and compatibility
  shims while reducing complexity only when there is a clear need.

## Deferred physics/features

- online optical-constant databases;
- Excel, GUI, JSON input, and HTML report frontends; the current ProjectSpec
  report is Markdown only;
- expanded photoionization cross-section models;
- new optimizers without a demonstrated validation need;
- broad physical-kernel restructuring without a separate validated plan.
