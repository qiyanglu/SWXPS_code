# AGENTS.md

This repository implements a transparent Python program for simulating x-ray reflectivity and, later, standing-wave XPS from multilayer thin films.

## Project goals

1. Start with multilayer x-ray reflectivity using the Parratt recursion.
2. Keep the code physically transparent and easy to verify.
3. Prefer simple, tested code over feature completeness.
4. Do not implement fitting, electric-field profiles, or XPS intensities until reflectivity is validated.

## Physics conventions

- Incidence angle is the grazing angle relative to the sample surface, in degrees.
- Photon energy is in eV.
- Wavelength is in Angstrom.
- Layer thickness and roughness are in Angstrom.
- Each layer has complex refractive index:

  n = 1 - delta + i beta

- The first layer is vacuum.
- The last layer is a semi-infinite substrate.
- The initial implementation should use s-polarization only.
- p-polarization can be added later.

## Coding rules

- Use Python.
- Use numpy for numerical calculations.
- Use scipy only when needed.
- Always use JAX, Flax.nnx, Optax for deep learning jobs.
- Always use tensorflow_probability.substrates.jax for probabilistic modeling jobs.
- Always use tensorflow.data for dataset management.
- Always use JAX for autodifferentiation.      
- Always use optuna for hyperparameter search.
- Keep the core Parratt calculation independent of xraydb or other optical-constant databases.
- Write small functions with clear names.
- Add tests before adding major new features.

## Testing rules

Every implementation of reflectivity must pass:

1. A two-layer stack vacuum/substrate reproduces the Fresnel reflectivity.
2. A stack with identical refractive indices gives near-zero reflectivity.
3. A periodic multilayer shows a Bragg peak near:

   m lambda = 2 d sin(theta)

4. Reflectivity should not become unphysically larger than 1 except for small numerical tolerance.

## Planning rule
Always create or update `PLAN.md` for every non-trivial task before making code changes.
Keep `PLAN.md` concise and current as work progresses.



