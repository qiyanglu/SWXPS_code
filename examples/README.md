# Examples

## LaNiO3/SrTiO3 Reflectivity

Run:

```powershell
python examples/plot_lno_sto_reflectivity.py
```

This creates:

```text
examples/lno_sto_reflectivity.png
```

The example stack is:

```text
vacuum / [LaNiO3 / SrTiO3] x 20 / SrTiO3 substrate
```

The script uses:

- Energy: 3000 eV
- LaNiO3 thickness: 20 Angstrom
- SrTiO3 thickness: 20 Angstrom
- Bilayer period: 40 Angstrom
- Interface roughness: 3 Angstrom
- Grazing angle scan: 0.05 to 5.0 degrees

The optical constants are read from `OPC/LaNiO3.dat` and `OPC/SrTiO3.dat`.
The requested energy must lie within the table range.

## Roughness Comparison

Run:

```powershell
python examples/compare_lno_sto_roughness.py
```

This creates:

```text
examples/lno_sto_roughness_comparison.png
```

It plots the same LaNiO3/SrTiO3 stack with sharp interfaces and with 3 Angstrom
RMS roughness, so the damping of fringes and Bragg features can be seen
directly.

## Electric Field Profile

Run:

```powershell
python examples/plot_lno_sto_field_profile.py
```

This creates:

```text
examples/lno_sto_field_profile.png
```

It uses 1000 eV photons, scans around the first-order Bragg condition, finds
the local reflectivity maximum, and plots a contour map of graded-roughness
transfer-matrix electric-field intensity `|E|^2` as a function of depth and
grazing incidence angle.

## Top 100 Angstrom Off-Peak Fields

Run:

```powershell
python examples/plot_lno_sto_top100_offpeak_fields.py
```

This creates:

```text
examples/lno_sto_top100_offpeak_fields.png
```

It uses the same 20 Angstrom / 20 Angstrom LNO/STO stack at 1000 eV, but plots
only the top 100 Angstrom and excludes the peak angle from the graded-roughness
transfer-matrix field profiles.

## La 4d, O 1s, and Ti 2p Rocking Curves

Run:

```powershell
python examples/plot_lno_la4d_rocking_curve.py
```

This creates:

```text
examples/lno_la4d_o1s_ti2p_rocking_curves.png
```

It simulates normalized La 4d, O 1s, and Ti 2p standing-wave XPS rocking curves
for the LaNiO3/SrTiO3 stack. The photon energy is 1000 eV. La 4d uses a 105 eV
binding energy, O 1s uses a 530 eV binding energy, and Ti 2p uses a 460 eV
binding energy. Cross sections are treated as constants; IMFP values are
interpolated from `IMFP`. The RCs are separated into vertically aligned
subplots so their angular phases can be compared directly.

The example uses the high-level simulation API:

```python
simulate_reflectivity(ReflectivityRequest(...))
simulate_rocking_curves(RockingCurveRequest(...))
```

Those request/result objects are intended to be the stable interface for later
optimization or fitting code.

## Stack Concentration Profile

Run:

```powershell
python examples/plot_lno_sto_stack_profile.py
```

This creates:

```text
examples/lno_sto_stack_profile.png
```

It plots La, Ti, and O relative concentration versus depth for the 20 Angstrom /
20 Angstrom LNO/STO stack. The broadened transitions show the same
error-function roughness model used by the SW-XPS simulations.
