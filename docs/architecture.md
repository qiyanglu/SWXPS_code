# Architecture

SWANX means **S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy. The supported Python namespace is `swanx`.

## Public API

Beginner users who want a whole project run should start with `swanx init my_project` and `python my_project/run_project.py`; default init projects copy packaged tutorial data and are runnable from any process current working directory. Advanced project scripts can call `swanx.project.run_project("project.yaml")`. Users writing custom simulation scripts should use `import swanx as sx` for the compact top-level simulation API. Focused subpackages support explicit IO, fitting, diagnostics, and internal development.

## Maintained namespaces

- `swanx.stack`: layers, stacks, templates, slicing, and profiles.
- `swanx.optics`: Parratt, transfer-matrix, fields, and unified-grid optics.
- `swanx.xps`: attenuation, XPS intensity, and rocking curves.
- `swanx.fitting`: parameters, objectives, and maintained fitting backends. Optimizer-independent helpers live in `swanx.fitting.core`; backends live in `swanx.fitting.bo`, `swanx.fitting.jax_gradient`, and `swanx.fitting.jax_least_squares`.
- `swanx.diagnostics`: covariance, correlation, plots, reports, and result exports.
- `swanx.io`: OPC readers, IMFP readers, material-table loaders, stack/core-level builders, and experimental reflectivity/rocking-curve readers.
- `swanx.workflows`: high-level simulation, fitting, and reporting entry points.
- `swanx.project`: YAML ProjectSpec parsing, validation, object building, project execution, and report-file writing. Report implementation modules live under `swanx.project.reporting`; `swanx.project.reports` is the compatibility facade.

## Compatibility shims

Several root modules remain only to keep older documented imports working. New
internal code should use the maintained subpackages instead:

- `swanx.bo` -> `swanx.fitting.bo`
- `swanx.jax_gradient` -> `swanx.fitting.jax_gradient`
- `swanx.jax_least_squares` -> `swanx.fitting.jax_least_squares`
- `swanx.reflectivity` -> `swanx.optics.parratt`
- `swanx._fitting` -> `swanx.fitting.core`

## IO boundary

Starter inputs live under `data/OPC/`, `data/IMFP/`, and `data/curves/` for
`swanx init`. Maintained examples use those OPC/IMFP tables plus the synthetic
C/LNO/STO benchmark CSV when they need reflectivity and rocking-curve data.
OPC, IMFP, and experimental curve files are read before simulation/fitting.
CXRO optical-constant tables are interpolated at photon energy, IMFP tables are
interpolated at core-level kinetic energy, and the resulting numbers are used
to construct explicit `SimulationStack`, `CoreLevelRequest`,
`ReflectivityData`, and `RockingCurveData` objects.

The YAML ProjectSpec workflow is the main human-editable wrapper over the same IO and simulation/fitting objects. It resolves materials, stack layers, explicit core-level emitting layers, datasets, and report output before calling existing SWANX APIs. `swanx inspect` summarizes a ProjectSpec without running simulation or fitting. Default outputs are written under the project YAML directory, and every run writes a simple Markdown `report.md`. Thickness and roughness values are Angstrom; `roughness_A` is the upper-interface roughness of that layer, and `repeat_index` is 1-based in repeat blocks.

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

Unified slicing is the default for high-level simulation and fitting. `slicing=None` explicitly selects the legacy fixed-step path. `simulate_only` is fully supported by the YAML ProjectSpec workflow. JAX-based least-squares/autodiff is the recommended fitting path for differentiable fixed-shape workflows, with explicit factory callbacks still required by YAML projects; Bayesian optimization remains available as an optional global black-box baseline/robustness check, not the default.
