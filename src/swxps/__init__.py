"""Compatibility alias for :mod:`swanx`.

New code should import :mod:`swanx`. This namespace remains temporarily so
existing scripts continue to run unchanged.
"""

from importlib import import_module
import sys

import swanx as _swanx

for _name in dir(_swanx):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_swanx, _name)

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

__all__ = list(_swanx.__all__)
