# ProjectSpec JAX Least-Squares Example

This folder is the runnable ProjectSpec fitting counterpart to the default
`swanx init` tutorial. It uses the shared C/LaNiO3/SrTiO3 synthetic data, loads
reflectivity plus La 4d, O 1s, Ti 2p, and C 1s rocking curves, and fits the
same fixed-shape JAX least-squares model through the internal
`run.optimizer.residual: "auto_fixed_grid"` path. Its ProjectSpec uses the
maintained edge-polynomial rocking-curve normalization default: first and last
10 percent, polynomial order 2. It enables identifiability analysis and
`run.outputs.next_project`, so each fitting run can write
`next_project/project_best_start.yaml`, `next_project/project_reduced.yaml`, and
`next_project/reduction_notes.md` as reviewed starting points for follow-up
fits.

```powershell
swanx validate examples/04_fitting/projectspec_jax_least_squares/project.yaml
python examples/04_fitting/projectspec_jax_least_squares/run_project.py
```

Outputs are written under `runs/` in this folder.
