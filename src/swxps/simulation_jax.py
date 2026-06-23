"""Compatibility alias for :mod:`swanx.simulation_jax`."""

from importlib import import_module
import sys

_module = import_module("swanx.simulation_jax")
sys.modules[__name__] = _module
