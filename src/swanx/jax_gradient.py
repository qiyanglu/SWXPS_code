"""Compatibility shim for :mod:`swanx.fitting.jax_gradient`.

The maintained JAX-gradient optimizer lives in ``swanx.fitting.jax_gradient``.
This root module remains so older imports such as ``import swanx.jax_gradient``
keep working.
"""

from .fitting.jax_gradient import *  # noqa: F401,F403