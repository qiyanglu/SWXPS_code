# User guide

SWANX (**S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy) is imported as `swanx`.

Beginner scripts should start with:

```python
import swanx as sx
```

The recommended simulation entry pattern is `import swanx as sx`. File readers,
fitting data classes, and diagnostics live in focused namespaces for explicit
workflow steps.

## Optical constants (OPC)

SWANX expects optical constants to be resolved before simulation. CXRO-style OPC files use the fixed column meaning:

```text
Energy(eV), Delta, Beta
```

The values are interpolated at photon energy:

```text
E = h nu
```

SWANX keeps the refractive-index convention:

```text
n = 1 - delta + i beta
```

Use:

```python
from swanx.io import read_optical_constants

table = read_optical_constants("data/OPC/LaNiO3.dat")
delta, beta = table.at_energy(900.0)
```

## IMFP

Electron IMFP files are interpolated at photoelectron kinetic energy:

```text
E_kin = h nu - E_B
```

Use:

```python
from swanx.io import read_imfp

table = read_imfp("data/IMFP/LNO.ANG")
lambda_angstrom = table.at_kinetic_energy(795.0)
```

## Material labels

Material labels must match across layer specs, OPC mappings, IMFP mappings, and concentration dictionaries. Common labels in examples are:

```text
LNO
STO
vacuum
```

## Loading material tables

Users may place files wherever convenient and pass explicit paths. The tutorial
files live under `data/OPC/` and `data/IMFP/`:

```python
from swanx.io import load_material_tables

tables = load_material_tables(
    opc_files={"LNO": "data/OPC/LaNiO3.dat", "STO": "data/OPC/SrTiO3.dat"},
    imfp_files={"LNO": "data/IMFP/LNO.ANG", "STO": "data/IMFP/STO.ANG"},
)
```

Directory mode is also available:

```python
tables = load_material_tables(opc_dir="data/OPC", imfp_dir="data/IMFP", materials=["LNO", "STO"])
```

The repository examples use `data/OPC/` and `data/IMFP/`. These are tutorial/example data, not a complete built-in materials database.

## Building simulation requests

`RockingCurveRequest` receives already-resolved stack optical constants, core-level IMFP dictionaries, and material concentration dictionaries. It does not read OPC or IMFP files internally.

```python
from swanx.io import stack_from_layer_specs, core_level_from_tables

stack = stack_from_layer_specs(
    [
        {"material": "vacuum", "thickness": 0.0},
        {"material": "LNO", "thickness": 40.0, "roughness": 3.0},
        {"material": "STO", "thickness": 0.0},
    ],
    optical_constants=tables.optical_constants,
    energy_ev=900.0,
)

la4d = core_level_from_tables(
    name="La 4d",
    binding_energy_ev=105.0,
    photon_energy_ev=900.0,
    concentration_by_material={"LNO": 1.0},
    imfp_tables=tables.imfp,
)
```

## Polarization

High-level reflectivity and rocking-curve requests use s-polarization by
default:

```python
request = sx.ReflectivityRequest(angles=angles, energy_ev=900.0, stack=stack)
```

Set `polarization="p"` for p-polarized optics, or pass a mixed raw weighting:

```python
request = sx.RockingCurveRequest(
    angles=angles,
    photon_energy_ev=900.0,
    stack=stack,
    core_levels=(la4d,),
    polarization={"s": 0.7, "p": 0.3},
)
```

For mixed polarization, SWANX combines raw reflectivity or raw SW-XPS intensity
as `fs * s + fp * p` before rocking-curve normalization.

## Loading experimental reflectivity data

Experimental reflectivity files can be CSV or whitespace-separated tables with
headers:

```text
angle_deg,reflectivity
5.0,0.010
5.1,0.012
```

Use:

```python
from swanx.io import read_reflectivity_data

reflectivity_exp = read_reflectivity_data("data/curves/lno_sto_reflectivity.csv")
```

Headerless files are supported when explicit column indices are provided:

```python
reflectivity_exp = read_reflectivity_data(
    "reflectivity.dat",
    angle_column=0,
    intensity_column=1,
)
```

The loader sorts rows by angle, rejects duplicate angles, rejects non-finite
angles/intensities/sigma values, and rejects negative sigma. Negative intensity
is allowed for background-subtracted data.

## Loading experimental rocking curves

Rocking-curve files use the same table conventions:

```python
from swanx.io import read_rocking_curve_data

la4d_exp = read_rocking_curve_data("data/curves/la4d_rocking_curve.csv")
```

Common intensity headers include `intensity`, `I`, `counts`, and `signal`.
Headerless files require explicit column indices.

## Rocking-curve normalization

`swanx.io` reads external files. `swanx.preprocessing` owns normalization
algorithms. `swanx.fitting` consumes `ReflectivityData` and `RockingCurveData`
objects.

To normalize while loading a rocking curve, pass a preprocessing mode through
the loader:

```python
la4d_exp = read_rocking_curve_data(
    "data/curves/la4d_rocking_curve.csv",
    normalization_mode="mean",
)
```

Supported modes match `swanx.preprocessing.normalize_rocking_curve`, including
`mean` and `edge_polynomial`. With `normalization_mode=None`, raw intensity is
returned.


## YAML ProjectSpec workflow

The main human-editable workflow is a YAML `ProjectSpec`. Start with:

```bash
swanx init my_project
python my_project/run_project.py
```

Use the CLI for automation when needed:

```bash
swanx validate my_project/project.yaml
swanx run my_project/project.yaml
```

Use `swanx init my_project --copy-example-data` for a self-contained starter,
or `swanx init my_project --data-root /path/to/data` to point at another data
root. Default outputs are written under `my_project/runs/`, and every run writes
`report.md`. YAML support is optional via `python -m pip install -e ".[project]"`.
Thickness and roughness fields use Angstrom: `thickness_A` and `roughness_A`.
`roughness_A` on layer j means roughness/interdiffusion at the upper interface
of that layer, the interface between layer j-1 and layer j. In repeat blocks,
`repeat_index` is 1-based. Core levels must explicitly select emitting layers
with `emit_from.layer_ids`, `emit_from.tags`, or `emit_from.all: true`.

## Full fitting input workflow

A realistic file-based fitting setup is:

1. Read OPC files with `load_material_tables(...)`.
2. Read IMFP files with `load_material_tables(...)`.
3. Build a `SimulationStack` with `stack_from_layer_specs(...)`.
4. Build core-level requests with `core_level_from_tables(...)`.
5. Load experimental reflectivity with `read_reflectivity_data(...)`.
6. Load experimental rocking curves with `read_rocking_curve_data(...)`.
7. Pass those objects into `swanx.fitting.FittingProblem`.

## Optimization

JAX-based least-squares fitting is the recommended path for differentiable fixed-shape SWANX workflows. Bayesian optimization remains available as an optional global black-box baseline/robustness check, not the default.

OPC and IMFP files are read outside the JAX-traced residual function. JAX fitting receives fixed numerical arrays or fixed-shape model inputs.
