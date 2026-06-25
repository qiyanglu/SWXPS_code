# Polarization support

## Goal

Add s, p, and mixed s/p polarization support while preserving existing default
s-polarized behavior.

## Scope

- Add `polarization` to high-level reflectivity and rocking-curve requests.
- Use admittance `Y_s = kz` and `Y_p = kz / n^2` in interface matrices.
- Compute p-polarized field intensity from the specified parallel/perpendicular
  field components.
- Support mixed polarization as a raw weighted sum before normalization.
- Update NumPy, unified-grid, and JAX paths.
- Add focused tests for backward compatibility, p-vs-s differences, mixed
  linearity, and accepted request modes.

## Non-goals

- No change to default results for callers that omit `polarization`.
- No broad refactor of optics or fitting code.
