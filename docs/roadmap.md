# Roadmap

## Current stage

Reflectivity, transfer-matrix fields, rough interfaces, normalized SW-XPS,
preprocessing, and three fitting backends are implemented and tested. The main
scientific task is validation of experimental assumptions rather than adding
more optimizer machinery.

## Near-term priorities

1. Review experimental RC preprocessing and normalization against raw data.
2. Quantify sensitivity to dataset weights, bounds, initialization, and minima.
3. Check fitted thickness, roughness, chemistry, and offsets independently.
4. Keep NumPy/JAX forward and residual parity covered by tests.
5. Produce compact, reproducible case-study summaries.

## Deferred features

- p-polarization.
- Online optical-constant databases.
- Expanded photoionization cross-section models.
- New optimizers without a demonstrated validation need.

Each substantial item requires a scoped plan under `docs/plans/`.
