"""Compatibility alias for :mod:`swanx.reflectivity_jax`."""

from importlib import import_module
import sys

_module = import_module("swanx.reflectivity_jax")
sys.modules[__name__] = _module
