# SWANX Examples

These examples are organized as a user learning path. Start with the YAML
ProjectSpec workflow unless you need a custom Python script.

All maintained examples use the same introductory synthetic case as the main
benchmark: a carbon cap on a 20-repeat LaNiO3/SrTiO3 (LNO/STO) superlattice on
a SrTiO3 (STO) substrate at 1000 eV. The shared case includes reflectivity plus
La 4d, O 1s, Ti 2p, and C 1s rocking curves, so stack building, simulation,
data overlay, and fitting examples all teach the same workflow.

## Recommended Path

1. `01_quickstart_projectspec/`: copy-pasteable ProjectSpec YAML files for
   simulation, data overlays, repeats, and optional fitting settings.
2. `02_experimental_data/`: a small Python example that loads the synthetic
   benchmark CSV as data and evaluates the matching overlay.
3. `03_python_api/`: compact Python API examples for users who need scripting
   instead of YAML.
4. `04_fitting/`: standalone fitting examples and notes for JAX-based fitting.
5. `advanced/`: lower-level optics, field, roughness, XPS, and profile
   visualizations for users who want to inspect the model internals.

## Quick Commands

```powershell
swanx validate examples/01_quickstart_projectspec/minimal_simulate_only.yaml
swanx run examples/01_quickstart_projectspec/minimal_simulate_only.yaml
python examples/03_python_api/build_from_opc_imfp.py
python examples/02_experimental_data/load_and_overlay_curves.py
```
