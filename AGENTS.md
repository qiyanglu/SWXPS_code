# AGENTS.md

This repository implements `swanx` (standing-wave analysis for X-ray spectroscopy): transparent Python tools for X-ray reflectivity and
standing-wave XPS from multilayer thin films.

## Project goals

1. Preserve the validated Parratt-reflectivity foundation and its transparency.
2. Support tested transfer-matrix fields, rough interfaces, and SW-XPS curves.
3. Support fitting backends while treating experimental fits as provisional.
4. Prefer small, readable, tested changes over feature completeness.

## Physics conventions

- Incidence angle is the grazing angle relative to the sample surface, in degrees.
- Photon energy is in eV.
- Wavelength, thickness, and roughness are in Angstrom.
- Refractive index is `n = 1 - delta + i beta`.
- The first layer is vacuum; the last is a semi-infinite substrate.
- The validated implementation uses s-polarization.

## Coding rules

- Use Python and NumPy for core numerical calculations.
- Use SciPy only where optimization or numerical tooling requires it.
- Keep the Parratt core independent of optical-constant databases.
- Keep functions small and clearly named.
- Do not change physical behavior during repository or documentation cleanup.
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

- `src/swanx`: maintained package code and primary namespace.
- `src/swxps`: temporary compatibility aliases for old imports.
- `tests`: regression tests.
- `examples`: compact tutorial scripts only.
- `case_studies`: experimental inputs, maintained runners, and canonical results.
- `benchmarks`: synthetic fitting and performance benchmarks.
- `runs`: generated local outputs; ignored by Git except its README.
- `archive`: superseded local experiments; ignored by Git except its README.
- `docs`: architecture, roadmap, plans, handoff state, and historical records.

## Planning rule

For any substantial change, first create or update an execution plan following
`PLANS.md`. Keep active plans concise and move completed long-form logs to
`docs/history`.

## Git rule

Do not stage, commit, amend, or push changes unless the user explicitly asks for
that Git action in the current request. Updating handoff documentation does not
imply permission to commit it. Leave completed revisions in the working tree
for user review when no explicit Git instruction is given.

## Session continuity rule

At the end of any substantial coding session, update `docs/PROJECT_STATE.md`
and `docs/TODO.md` so the project can be continued from another machine without
access to the local Codex transcript.
