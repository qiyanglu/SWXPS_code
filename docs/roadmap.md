# Roadmap

## Current stage

SWANX is preparing for API freeze before public release. The Python import name is `swanx`; the early development namespace `swxps` has been removed.

Unified slicing is the default for high-level simulation requests and fitting problems. Explicit `slicing=None` selects the legacy fixed-step path. JAX-based automatic differentiation is the recommended fitting strategy; Bayesian optimization remains available as a baseline and robustness comparison.

The current priority is API clarity, realistic file-based OPC/IMFP workflows, JAX fitting workflow documentation, maintained examples, and physical validation -- not adding more core physics.

## Near-term priorities

1. Keep README and maintained examples clear, compact, executable, and based on `swanx`.
2. Validate the file-based OPC/IMFP workflow on tutorial and real case-study inputs.
3. Document the maintained fixed-shape JAX least-squares/autodiff workflow.
4. Maintain a small set of representative simulation and fitting tutorials.
5. Validate experimental RC preprocessing, normalization, weighting, bounds, angular offsets, parameter identifiability, and fitted structures.
6. Preserve NumPy/JAX numerical parity and source-location tests.

## Future IO extensions

- Experimental reflectivity CSV reader.
- Experimental rocking-curve CSV reader.
- Fit-result export helpers for user workflows.
- Optional adapters for CXRO-style generated files.
- Eventual fitting source migration into `swanx.fitting.*` submodules.

## Deferred physics/features

- p-polarization.
- Online optical-constant databases.
- Expanded photoionization cross-section models.
- New optimizers without a demonstrated validation need.
- Broad physical-kernel restructuring without a separate validated plan.

Historical stage plans under `docs/plans/` record completed migrations. They are not the current roadmap and should retain status notes rather than be rewritten as current instructions.
