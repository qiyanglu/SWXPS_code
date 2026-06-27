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
`minimal`, `multilayer`, and `fit-demo` templates. Direct Python APIs remain the
right surface for custom fixed-shape JAX fitting, new diagnostics, and
lower-level simulation experiments.

Unified slicing is the default high-level simulation path. JAX-based automatic
differentiation and least-squares are the recommended fitting strategy for
differentiable fixed-shape workflows; Bayesian optimization remains an optional
global black-box baseline/robustness check.

## Near-term priorities

1. Keep README, user guide, templates, and maintained examples concise,
   executable, and based on `swanx`.
2. Expand ProjectSpec examples only when they reuse existing IO, simulation,
   and fitting APIs without introducing new physics paths.
3. Improve fixed-shape JAX least-squares fitting documentation and show how it
   relates to ProjectSpec callback factories.
4. Preserve NumPy/JAX numerical parity and source-location tests.
5. Continue validating RC preprocessing, weighting, angular offsets, parameter
   identifiability, and fitted structures.
6. Keep s/p/mixed polarization examples and tests aligned with representative
   multilayer and superlattice cases.

## Future IO and report extensions

- richer experimental-data formats when real lab conventions require them;
- richer ProjectSpec examples and report exports beyond the current CSV, JSON,
  Markdown, and plot layout;
- uncertainty/sigma propagation in downstream diagnostics and exports;
- optional adapters for additional optical-constant formats;
- continued cleanup of fitting internals into focused `swanx.fitting.*`
  submodules when it reduces complexity.

## Deferred physics/features

- online optical-constant databases;
- Excel, GUI, JSON input, and HTML report frontends; the current ProjectSpec
  report is Markdown only;
- expanded photoionization cross-section models;
- new optimizers without a demonstrated validation need;
- broad physical-kernel restructuring without a separate validated plan.
