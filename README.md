# SWXPS

Transparent Python tools for simulating x-ray reflectivity and standing-wave
XPS from multilayer thin films.

The current code focuses on readable, testable simulations for:

- Parratt-recursion x-ray reflectivity for multilayers
- Transfer-matrix reflectivity and electric-field profiles
- Error-function interface roughness through graded effective slices
- Optical constants loaded from Henke-style `.dat` files in `OPC`
- IMFP values loaded from tabulated files in `IMFP`
- Normalized standing-wave XPS rocking curves with constant cross sections
- Stack/concentration profile visualization utilities

## Install

From the repository root:

```powershell
python -m pip install -e .
```

For plotting examples:

```powershell
python -m pip install -e ".[plot]"
```

## Run Tests

```powershell
python -m pytest
```

## Examples

Example scripts live in `examples`.

```powershell
python examples/plot_lno_sto_reflectivity.py
python examples/compare_lno_sto_roughness.py
python examples/plot_lno_sto_field_profile.py
python examples/plot_lno_la4d_rocking_curve.py
python examples/plot_lno_sto_stack_profile.py
```

The generated figures are intentionally ignored by Git. Re-run the examples to
regenerate them locally.

## Current Scope

The package is intended as a transparent simulation backend. Fitting and
optimization are not implemented yet, but the high-level request/result objects
in `swxps.simulation` are designed so future fitting code can call the
simulation functions directly.
