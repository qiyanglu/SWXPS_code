# Step 10 clean user entry workflow

## Goal

Make the beginner workflow read as:

```text
data files -> swanx.io -> simulation requests / fitting data -> simulation + fitting + diagnostics
```

without changing optics, XPS, reflectivity, fitting algorithms, or numerical behavior.

## Scope

- Move tutorial data from `examples/data/` to root-level `data/`.
- Update maintained examples, tests, README, and active docs to use `data/...`.
- Keep `swanx.io` as the narrow public file-input API.
- Keep rocking-curve normalization owned by `swanx.preprocessing`.
- Keep active docs concise and free of migration-log wording.
- Run the requested tests and examples.

## Non-goals

- No physics or optimizer changes.
- No historical doc rewrites.
- No Git staging, commit, or push unless explicitly requested.
