# Project state

Last updated: 2026-06-22

This is the machine-independent handoff for the SWXPS repository. Read this
file, `docs/TODO.md`, `AGENTS.md`, and the root `README.md` before continuing a
substantial coding session.

## Git state at handoff

- Branch: `main`.
- Latest implementation commit: `01f8d6f`
  (`Add workflow benchmark and cache scientific tables`).
- Documentation handoff commit: `97e27b1`.
- Slicing planning commit before the current refinement: `cdcf827`.
- Local `runs/` and `archive/` contents are ignored and do not travel through Git.

## Current capabilities

- Validated s-polarized Parratt reflectivity using grazing angles in degrees.
- Transfer-matrix reflectivity and depth-dependent electric fields.
- Rough-interface grading through effective sharp slices.
- Material and concentration profiles and normalized SW-XPS rocking curves.
- Experimental preprocessing and joint reflectivity/rocking-curve objectives.
- Bayesian optimization, JAX L-BFGS-B, and JAX/Jacobian TRF least squares.
- Declarative stack builders backed by cached local optical/IMFP tables.
- NumPy/JAX parity coverage for maintained backend behavior.

Experimental fits remain provisional until bounds, weights, normalization,
optical constants, IMFPs, chemistry, and optimizer sensitivity are reviewed.

## Planned unified slicing milestone

The current step-based roughness and field grids independently calculate their
lengths from trial thickness. This can undersample thin layers and change JAX
shapes during fitting.

The confirmed design is documented in
`docs/plans/adaptive_fixed_shape_slicing_2026-06-22.md`:

- `max_slice_thickness` is user configurable and defaults to 2 Angstrom;
- `min_slices` initially defaults to 10 per positive finite nominal layer;
- roughness, field, concentration/IMFP, attenuation, and RC integration use one
  shared cell-centered grid;
- JAX fitting counts are fixed once from a capacity stack built at upper bounds;
- trial thickness changes cell widths, not array shapes;
- the new path is opt-in and existing APIs remain intact.

No source code has been changed for this milestone yet.

## Recent completed work

The repository was reorganized into maintained package, tutorials, case
studies, benchmarks, local runs, archive, and documentation areas.
Optical-constant and IMFP parsers use bounded metadata-aware caches.

The C/[LNO/STO]x8/STO benchmark is
`benchmarks/performance/profile_forward_workflow.py`. On the original machine,
its 61-angle run measured an 8.28x cached table-load speedup and a
0.023589-second complete fitting objective. These are local baselines only.

## Verification status

Last full implementation verification:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Result: 91 passed, 46 existing `np.trapz` deprecation warnings. The 2026-06-22
slicing sessions have changed documentation only.

## Repository map

- `src/swxps/`: maintained implementation.
- `tests/`: regression and parity tests.
- `examples/`: compact tutorials.
- `case_studies/`: maintained experimental runners and canonical results.
- `benchmarks/`: synthetic and performance benchmarks.
- `OPC/`, `IMFP/`: local scientific tables.
- `runs/`, `archive/`: ignored local outputs and superseded experiments.
- `docs/architecture.md`: code/data flow and performance boundaries.
- `docs/plans/`: active and scoped plans.
- `docs/history/`: superseded handoffs and chronological records.

## Current direction

Implement only the pure policy, fixed-plan, and grid materialization layer
first. Review its tests before connecting it to roughness or fields. Preserve
the complete legacy path until unified-grid convergence and NumPy/JAX parity
are demonstrated.
