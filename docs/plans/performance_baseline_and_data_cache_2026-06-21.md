# Performance baseline and scientific-table cache plan

## Goal

Measure the major stages of one representative SWXPS forward/fitting workflow
and remove repeated optical-constant and IMFP file parsing without changing the
existing public API or numerical results.

## Physics background

The benchmark uses the existing multilayer conventions: grazing angles in
degrees, photon energy in eV, lengths in Angstrom, refractive index
`n = 1 - delta + i beta`, vacuum first, and a semi-infinite substrate last.
Caching changes only when static tabulated data are parsed; interpolation,
roughness discretization, reflectivity, fields, attenuation, and SW-XPS
integration remain unchanged.

## Files to create or modify

- `benchmarks/performance/profile_forward_workflow.py`
- `src/swxps/optical_constants.py`
- `src/swxps/imfp.py`
- focused tests for cache identity, invalidation behavior, and numerical parity
- `docs/architecture.md`

Existing examples and public function signatures remain unchanged.

## Implementation steps

1. Commit the current reorganized repository as the milestone baseline.
2. Time table loading, stack construction, roughness discretization,
   reflectivity, fields/SW-XPS, and one fitting-objective evaluation separately.
3. Add bounded in-process caches for resolved optical-constant and IMFP files.
4. Preserve uncached behavior when a file changes by including file metadata in
   the cache key or otherwise providing explicit cache invalidation.
5. Document the data flow and static-versus-dynamic boundary.

## Tests

- Repeated loads return the cached parsed table.
- Rewriting a temporary table invalidates the cached value.
- Cached and freshly parsed interpolation results agree exactly.
- Existing optical-constant, IMFP, stack-builder, simulation, and fitting tests pass.
- Full `python -B -m pytest -q -p no:cacheprovider` passes.

## Validation

The benchmark must report all named stages and finite positive timings. Cached
loads must be measurably faster in repeated local execution, while simulated
reflectivity and rocking curves remain numerically unchanged.

## Progress log

- 2026-06-21: Created the milestone plan.
- 2026-06-21: Committed the reorganized repository as baseline `04da44a`.
- 2026-06-21: Added bounded, metadata-aware caches and focused parity/invalidation tests.
- 2026-06-21: Added the stage-by-stage C/[LNO/STO]x8/STO workflow benchmark.
- 2026-06-21: Default benchmark measured an 8.28x cached table-load speedup;
  the full objective took 0.023589 seconds for 61 angles on this machine.
- 2026-06-21: Full regression suite passed: 91 tests, with 46 existing warnings.
