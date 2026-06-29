# SWANX Examples

These examples are organized as a user learning path. Start with the YAML
ProjectSpec workflow unless you need a custom Python script.

Most examples use the same compact LNO/STO tutorial system: LaNiO3 layers on
SrTiO3 with tutorial OPC, IMFP, reflectivity, and rocking-curve data under
`data/`. The larger C/LNO/STO synthetic case is used repeatedly in
`benchmarks/` for fitting, slicing, and optimizer comparisons.

## Recommended Path

1. `01_quickstart_projectspec/`: copy-pasteable ProjectSpec YAML files for
   simulation, data overlays, repeats, and optional fitting settings.
2. `02_experimental_data/`: a small Python example that loads tutorial
   experimental curves and evaluates residuals.
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

Private experimental fitting belongs in local ignored `case_studies/`.
Synthetic fitting and timing studies belong in `benchmarks/`. Generated run
output belongs in local ignored `runs/` or project-local `runs/` folders.
