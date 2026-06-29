# ProjectSpec v1.2 UX Polish Plan

## Goal

Make the YAML ProjectSpec workflow easier for beginners without changing optics,
XPS, reflectivity, fitting algorithms, or numerical behavior.

## Scope

- Package C/LaNiO3/SrTiO3 starter OPC, IMFP, and curve files with
  `swanx.project`.
- Make default `swanx init my_project` self-contained and independent of process
  current working directory.
- Add `--template minimal`, `--template multilayer`, and `--template fit-demo`.
- Add `swanx inspect project.yaml` for non-simulation project summaries.
- Validate missing `jax_gradient` callback factories like least-squares.
- Improve `report.md` with per-plot notes, overlay status, and simulation-only
  parameter values.
- Add visible progress messages for `swanx run` and generated beginner scripts.
- Polish report plots with a compound reflectivity-plus-RC overview, incident-angle
  labels, and least-squares parameter/correlation diagnostics.
- Keep Markdown reports; no GUI, Excel, JSON input, HTML, Auger, XES, XMCD,
  single-crystal diffraction, BO fallback, or automatic no-code JAX residuals.

## Validation Targets

```bash
python -m pytest tests/test_project_workflow.py -q
python -m pytest -q
```

Smoke checks:

```bash
swanx init runs/projectspec_v12_smoke
python runs/projectspec_v12_smoke/run_project.py
swanx inspect runs/projectspec_v12_smoke/project.yaml
swanx validate runs/projectspec_v12_smoke/project.yaml
```
