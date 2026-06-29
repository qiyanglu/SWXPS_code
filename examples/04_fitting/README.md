# Fitting Examples

Start with `projectspec_jax_least_squares/` when you want the same scope as the
default `swanx init` tutorial: ProjectSpec input, reflectivity plus four
rocking-curve datasets, off-peak normalization, fixed-grid slicing, an explicit
JAX least-squares residual factory, and the normal report folder output.

The standalone scripts demonstrate fitting APIs directly from Python on the
same C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 geometry used by the examples and
synthetic benchmark. They keep the residual lightweight by fitting shared LNO
and STO thicknesses against reflectivity. Use them when you need custom
residuals or a scripted fitting experiment.

```powershell
swanx validate examples/04_fitting/projectspec_jax_least_squares/project.yaml
python examples/04_fitting/projectspec_jax_least_squares/run_project.py
python examples/04_fitting/jax_gradient_reflectivity_fit.py
python examples/04_fitting/jax_least_squares_reflectivity_fit.py
```
