# Fitting module cleanup

Status: Implemented in this cleanup pass.

## Goal

Remove obsolete local case-study-dependent tests and make the fitting source
layout less surprising without changing fitting behavior or physics kernels.

## Scope

- Remove active tests that depend on ignored local `case_studies/` files.
- Move optimizer-independent fitting implementation from `swanx._fitting` to
  `swanx.fitting.core`.
- Keep `swanx._fitting` as a thin compatibility shim for old local scripts.
- Update active repository docs so `case_studies/` is described as local/private
  and ignored by Git.

## Validation

Focused fitting/import tests should cover the compatibility shim, public
`swanx.fitting` exports, and existing optimization behavior.