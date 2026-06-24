# Roadmap

## Current stage

The project is now branded `swanx` (*standing-wave analysis for X-ray
spectroscopy*). Reflectivity, transfer-matrix fields, rough interfaces,
normalized SW-XPS, unified layer slicing, three fitting backends, and fitting
diagnostics are implemented and tested. The former `swxps` namespace remains a
temporary compatibility shim.

Stage 2 moved diagnostics and stack slicing/profile code into their public subpackages. Stage 3 has now moved Parratt, transfer-matrix/field, and unified-grid optics implementations into `swanx.optics`. Flat `swanx.*` and legacy `swxps.*` paths remain compatibility shims. XPS, simulation, fitting, and workflow implementation moves are intentionally deferred.

Unified slicing is the default high-level path; `slicing=None` selects the
legacy fixed-step path.

## Near-term priorities

1. Build clear user-facing simulation, fitting, and reporting workflows from
   the existing validated components.
2. Migrate a small set of maintained examples to `import swanx as sx` and use
   them as executable documentation.
3. Validate experimental RC preprocessing, weighting, bounds, and fitted
   structures against raw data and independent expectations.
4. Keep NumPy/JAX forward, residual, fixed-shape, and namespace compatibility
   behavior covered by tests.
5. Produce compact reproducible case-study summaries and diagnostics.
6. Retire the `swxps` shim only in a separately planned breaking release.

The two standalone fitting demonstrations now live under `examples/fitting/` and use `swanx`. Generated `runs/` and `archive/` content remains ignored. The experimental `case_studies/` tree is intentionally retained; no Git-history rewrite is planned.

The current priority is workflows, examples, validation, and repository
readability闁炽儲鏀簅t adding more core physics.

## Deferred features

- p-polarization.
- Online optical-constant databases.
- Expanded photoionization cross-section models.
- New optimizers without a demonstrated validation need.
- Broad physical-kernel restructuring during the namespace transition.

Each substantial item requires a scoped plan under `docs/plans/`.
