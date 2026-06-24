"""Temporary broad compatibility facade for the former :mod:`swxps` API.

New code should use the frozen top-level :mod:`swanx` API. This namespace
continues to expose historical names so existing scripts remain functional.
"""

from importlib import import_module
import sys

from swanx import _legacy_api

for _name in _legacy_api.__all__:
    globals()[_name] = getattr(_legacy_api, _name)

_MODULE_ALIASES = {
    "fitting": "swanx._fitting",
    "xps": "swanx._xps",
    "diagnostics": "swanx.diagnostics",
}

_MODULES = (
    "bo", "constants", "diagnostics", "fields", "fit_diagnostics", "fitting",
    "imfp", "jax_gradient", "jax_least_squares", "layers",
    "optical_constants", "preprocessing", "profiles", "reflectivity",
    "result_exports", "simulation",
    "simulation_unified", "slicing", "stack_builders", "stack_visualization",
    "unified_grid", "xps",
)

for _module_name in _MODULES:
    _target = _MODULE_ALIASES.get(_module_name, f"swanx.{_module_name}")
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_module_name}"] = _module
    globals()[_module_name] = _module

__all__ = list(_legacy_api.__all__)
