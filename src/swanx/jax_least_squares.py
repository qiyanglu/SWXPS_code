"""Compatibility shim for :mod:`swanx.fitting.jax_least_squares`.

The maintained JAX least-squares optimizer lives in
``swanx.fitting.jax_least_squares``. This root module remains so older imports
such as ``import swanx.jax_least_squares`` keep working.
"""

from .fitting.jax_least_squares import *  # noqa: F401,F403