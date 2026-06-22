# Roadmap

## Current stage

Reflectivity, transfer-matrix fields, rough interfaces, normalized SW-XPS,
preprocessing, and three fitting backends are implemented and tested. A scoped
numerical-infrastructure milestone is planned to improve thin-layer resolution
and keep JAX shapes fixed during thickness fitting. The main scientific task
remains validation of experimental assumptions rather than adding optimizer
machinery.

## Near-term priorities

1. Validate and implement additive adaptive/fixed-shape layer slicing while
   preserving all legacy step-based APIs.
2. Review experimental RC preprocessing and normalization against raw data.
3. Quantify sensitivity to dataset weights, bounds, initialization, and minima.
4. Check fitted thickness, roughness, chemistry, and offsets independently.
5. Keep NumPy/JAX forward and residual parity covered by tests.
6. Produce compact, reproducible case-study summaries.

## Deferred features

- p-polarization.
- Online optical-constant databases.
- Expanded photoionization cross-section models.
- New optimizers without a demonstrated validation need.

Each substantial item requires a scoped plan under `docs/plans/`.
