# AGENTS.md

This repository implements transparent Python tools for x-ray reflectivity and
standing-wave XPS from multilayer thin films.

## Project goals

1. Preserve the validated Parratt-reflectivity foundation and its physical transparency.
2. Support transfer-matrix fields, rough interfaces, and normalized SW-XPS rocking curves.
3. Support tested fitting backends while treating experimental fits as provisional until physically validated.
4. Prefer small, readable, tested changes over feature completeness.

## Physics conventions

- Incidence angle is the grazing angle relative to the sample surface, in degrees.
- Photon energy is in eV.
- Wavelength, layer thickness, and roughness are in Angstrom.
- Each layer has complex refractive index `n = 1 - delta + i beta`.
- The first layer is vacuum and the last layer is a semi-infinite substrate.
- The validated implementation uses s-polarization; p-polarization is future work.

## Coding rules

- Use Python and NumPy for core numerical calculations.
- Use SciPy only where optimization or numerical tooling requires it.
- Keep the Parratt core independent of optical-constant databases.
- Keep public functions small and clearly named.
- Do not change physical behavior as part of repository or documentation cleanup.
- Add tests before major numerical or physical features.

## Testing rules

Reflectivity implementations must retain tests showing that:

1. A vacuum/substrate stack reproduces Fresnel reflectivity.
2. Identical refractive indices give near-zero reflectivity.
3. A periodic multilayer has a Bragg peak near `m lambda = 2 d sin(theta)`.
4. Reflectivity does not exceed 1 beyond small numerical tolerance.

New field, XPS, preprocessing, fitting, or backend behavior requires focused
tests and must preserve the full regression suite.

## Repository organization

- `src/swxps`: maintained package code.
- `tests`: regression tests.
- `examples`: compact reproducible examples and maintained case-study runners.
- `artifacts/runs`: generated local outputs; ignored by Git.
- `artifacts/archive`: superseded local experiments; ignored by Git.
- `docs`: architecture, roadmap, plans, and historical development records.

## Planning rule

For any substantial change, first create or update an execution plan following
`PLANS.md`. Keep active plans concise; move completed long-form logs to
`docs/history`.
