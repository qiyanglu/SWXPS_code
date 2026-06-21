# Local artifacts

This directory separates generated scientific artifacts from maintained examples.

- `runs/`: current and previous optimizer runs.
- `archive/`: superseded scripts and historical experiment bundles.

Both subdirectories are ignored by Git. Raw experimental inputs, maintained
runners, and canonical promoted results remain under `examples/`.

Do not treat files here as package dependencies. A maintained runner may use a
local run as an optional starting point, but canonical inputs and results should
remain documented in the case-study folder.
