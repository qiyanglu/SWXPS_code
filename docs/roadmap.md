# Roadmap

## Current stage

The scientific tool is **SWANX**: **S**tanding-**W**ave **A**nalysis for
**N**anoscale **X**-ray spectroscopy. Its Python import name is `swanx`.

Namespace and source-structure migration Stages 1-6 are complete. Stack,
diagnostics, optics, XPS, simulation, fitting, and workflow implementations now
live in their maintained `swanx` locations. Flat historical paths and the old
`swxps` namespace remain temporary compatibility shims.

Unified slicing is the default for high-level simulation requests and fitting
problems. Explicit `slicing=None` selects the legacy fixed-step path. JAX-based
automatic differentiation is the recommended fitting strategy; Bayesian
optimization remains available as a baseline and robustness comparison.

Historical stage plans under `docs/plans/` record completed migrations. They
are not the current roadmap and should retain short status notes rather than be
rewritten as current instructions.

## Near-term priorities

1. Keep the public API and README examples clear, compact, and executable.
2. Document the maintained fixed-shape JAX least-squares/autodiff workflow.
3. Maintain a small set of representative simulation and fitting tutorials.
4. Validate experimental RC preprocessing, normalization, weighting, bounds,
   angular offsets, parameter identifiability, and fitted structures.
5. Preserve NumPy/JAX numerical parity and namespace compatibility tests.
6. Retire the `swxps` shim only in a separately planned breaking release.

Generated `runs/` and `archive/` content remains ignored. Experimental
`case_studies/` are intentionally retained locally; no Git-history rewrite is
planned as part of the namespace work.

The current priority is API clarity, JAX fitting workflow documentation,
maintained examples, and physical validation—not adding more core physics.

## Deferred features

- p-polarization.
- Online optical-constant databases.
- Expanded photoionization cross-section models.
- New optimizers without a demonstrated validation need.
- Broad physical-kernel restructuring without a separate validated plan.

Each substantial item requires a scoped plan under `docs/plans/`.
