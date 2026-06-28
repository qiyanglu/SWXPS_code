# SWANX active milestone plan

> Current status (2026-06-27): `swanx` is the only supported namespace, unified
> slicing is the default high-level path, polarization support is implemented,
> fitting diagnostics are implemented, and the YAML ProjectSpec workflow is the
> primary human-editable workflow via `swanx init`, `swanx inspect`, `swanx validate`, and `swanx run`.

## Goal

Maintain the validated reflectivity, field, SW-XPS, fitting, and ProjectSpec
workflow platform while improving physical validation, runtime evidence, and
reproducibility of representative workflows.

## Physics background

Stable conventions are documented in `AGENTS.md` and `docs/architecture.md`.
The complete derivations and chronological record are preserved in
`docs/history/XR_REFLECTIVITY_DEVELOPMENT_LOG.md`.

## Current workflow surfaces

- Human-editable projects: `swanx init my_project`, edit
  `my_project/project.yaml`, then run `python my_project/run_project.py`;
  automation can use `swanx validate` and `swanx run`.
- Custom Python workflows: `swanx.io` builds explicit simulation/fitting objects
  consumed by `swanx` requests and `swanx.fitting`.
- Default ProjectSpec outputs belong under the project folder in
  `my_project/runs/`; other generated outputs belong in ignored `runs/`.
- Local/private experimental runners and inputs belong in ignored
  `case_studies/`.

## Implementation rules

Future milestones must name their scoped package, test, example, benchmark, or
case-study files. Generated fit output goes to `runs/`; superseded experiments
go to `archive/`. No core physics, optimizer, or backend behavior should change
without a focused validation plan.

## Near-term steps

1. Preserve NumPy/JAX numerical parity and reflectivity regression tests.
2. Keep ProjectSpec examples small, editable, and routed through existing IO,
   simulation, fitting, and report APIs.
3. Improve documentation for fixed-shape JAX least-squares and ProjectSpec
   optimizer callback factories.
4. Review experimental rocking-curve preprocessing and normalization on
   representative local inputs when available.
5. Validate fitted structures against independent physical expectations.
6. Profile representative workflows before restructuring performance-critical
   code.
7. Add cross sections, new optimizers, or new report frontends only through
   separate planned milestones.

## Tests

- Run the full test suite for every package behavior change.
- Add parity tests for new backend behavior.
- Retain the reflectivity validations from `AGENTS.md`.
- Preserve explicit `slicing=None` step-based behavior while testing unified
  slicing defaults.
- Add focused ProjectSpec tests for any new YAML schema behavior.

## Validation

Experimental results are not quantitative until bounds, weights, optical
constants, IMFPs, chemistry, and optimizer sensitivity have been reviewed. New
discretization is not accepted until thin-layer convergence, thick-layer cost,
fixed JAX shapes, and NumPy/JAX parity are demonstrated.

Latest full validation recorded in `docs/PROJECT_STATE.md`:

```bash
python -m pytest -q
# run before handoff; exact counts are intentionally not pinned here
```
