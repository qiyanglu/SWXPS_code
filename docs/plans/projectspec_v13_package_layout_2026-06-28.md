# ProjectSpec v1.3 Package Layout Cleanup

Goal: clean internal package layout without changing physics, optimizer behavior,
ProjectSpec behavior, or report output formats.

Planned/implemented scope:

- Move maintained fitting backend implementations under `swanx.fitting`.
- Keep root `swanx.bo`, `swanx.jax_gradient`, and `swanx.jax_least_squares` as compatibility shims.
- Split YAML project reporting into focused modules under `swanx.project.reporting`.
- Keep `swanx.project.reports` as the compatibility facade.
- Add import-stability tests for `swanx`, `swanx.project`, `swanx.fitting`, and root backend shims.
- Preserve BO as optional and do not add no-code JAX residual generation.

Validation target:

- Focused import/report tests.
- Existing ProjectSpec workflow tests.
- Full `python -m pytest` before committing.
