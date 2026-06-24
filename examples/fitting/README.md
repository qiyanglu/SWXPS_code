# Fitting examples

These compact synthetic examples demonstrate the maintained optional JAX
fitting backends through the `swanx` namespace:

```powershell
python examples/fitting/jax_gradient_reflectivity_fit.py
python examples/fitting/jax_least_squares_reflectivity_fit.py
```

Both fit one film thickness against synthetic reflectivity generated at a true
thickness of 24 Angstrom. They require the corresponding optional dependencies
(`gradient` or `least-squares`).
