# Fitting Examples

These standalone scripts demonstrate fitting APIs directly from Python on the
same C/[LaNiO3/SrTiO3 (LNO/STO)]x20/SrTiO3 geometry used by the examples and
synthetic benchmark. They keep the residual lightweight by fitting shared LNO
and STO thicknesses against reflectivity. Use ProjectSpec first for ordinary
project setup; use these when you need custom residuals or a scripted fitting
experiment.

```powershell
python examples/04_fitting/jax_gradient_reflectivity_fit.py
python examples/04_fitting/jax_least_squares_reflectivity_fit.py
```
