# Architecture

SWANX means **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy. The supported Python namespace is `swanx`.

## Public API

Beginner users should use `import swanx as sx` for the compact top-level simulation API. Focused subpackages support explicit IO, fitting, diagnostics, and internal development.

## Maintained namespaces

- `swanx.stack`: layers, stacks, templates, slicing, and profiles.
- `swanx.optics`: Parratt, transfer-matrix, fields, and unified-grid optics.
- `swanx.xps`: attenuation, XPS intensity, and rocking curves.
- `swanx.fitting`: parameters, objectives, and maintained fitting backends.
- `swanx.diagnostics`: covariance, correlation, plots, reports, and result exports.
- `swanx.io`: OPC readers, IMFP readers, material-table loaders, stack/core-level builders, and experimental reflectivity/rocking-curve readers.
- `swanx.workflows`: high-level simulation, fitting, and reporting entry points.

## IO boundary

Tutorial inputs live under `data/OPC/`, `data/IMFP/`, and `data/curves/`.
OPC, IMFP, and experimental curve files are read before simulation/fitting.
CXRO optical-constant tables are interpolated at photon energy, IMFP tables are
interpolated at core-level kinetic energy, and the resulting numbers are used
to construct explicit `SimulationStack`, `CoreLevelRequest`,
`ReflectivityData`, and `RockingCurveData` objects.

`RockingCurveRequest` does not read files internally. JAX-traced fitting functions receive fixed numerical arrays or fixed-shape model inputs, not file paths.

Rocking-curve normalization algorithms live in `swanx.preprocessing`.
`swanx.io.read_rocking_curve_data(...)` may call those algorithms when a
normalization mode is requested, but preprocessing functions are not exported
from `swanx.io`.

## Core physics

- `optics/parratt.py`: Parratt amplitudes and reflectivity.
- `optics/fields.py`: transfer-matrix fields and rough-interface effective layers.
- `polarization.py`: shared validation and weighting for s, p, and mixed
  polarization requests.
- `xps/attenuation.py`: electron attenuation through depth-dependent IMFPs.
- `xps/intensity.py`: continuous XPS integration and graded property sampling.
- `xps/rocking_curve.py`: normalized rocking-curve construction.
- `xps/grid.py`: cell-centered attenuation and midpoint XPS integration.
- `stack/slicing.py`: adaptive and fixed-plan unified layer grids.

The core uses grazing angles in degrees, photon energy in eV, lengths in Angstrom, and `n = 1 - delta + i beta`. Vacuum is first and the semi-infinite substrate last.

## Fitting

Unified slicing is the default for high-level simulation and fitting. `slicing=None` explicitly selects the legacy fixed-step path. JAX-based least-squares/autodiff is the recommended fitting path for differentiable fixed-shape workflows; Bayesian optimization remains available as a baseline and robustness check.
