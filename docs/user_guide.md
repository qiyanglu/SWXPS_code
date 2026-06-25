# User guide

SWANX (**S**tanding-**W**ave **A**nalysis for **N**anoscale **X**-ray spectroscopy) is imported as `swanx`.

Beginner scripts should start with:

```python
import swanx as sx
```

Advanced users can import focused APIs from `swanx.stack`, `swanx.optics`, `swanx.xps`, `swanx.fitting`, `swanx.diagnostics`, and `swanx.io`.

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

table = read_optical_constants("OPC/LaNiO3.dat")
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

table = read_imfp("IMFP/LNO.ANG")
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

Users may place files in project folders such as `OPC/` and `IMFP/`, or pass explicit paths:

```python
from swanx.io import load_material_tables

tables = load_material_tables(
    opc_files={"LNO": "OPC/LaNiO3.dat", "STO": "OPC/SrTiO3.dat"},
    imfp_files={"LNO": "IMFP/LNO.ANG", "STO": "IMFP/STO.ANG"},
)
```

Directory mode is also available:

```python
tables = load_material_tables(opc_dir="OPC", imfp_dir="IMFP", materials=["LNO", "STO"])
```

The repository examples use `examples/data/OPC/` and `examples/data/IMFP/`. These are tutorial/example data, not a complete built-in materials database.

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

## Optimization

JAX-based least-squares fitting is the recommended path for differentiable fixed-shape SWANX workflows. Bayesian optimization remains available as a baseline and robustness check.

OPC and IMFP files are read outside the JAX-traced residual function. JAX fitting receives fixed numerical arrays or fixed-shape model inputs.
