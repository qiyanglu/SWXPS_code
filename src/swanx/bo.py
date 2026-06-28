"""Compatibility shim for :mod:`swanx.fitting.bo`.

The maintained Bayesian-optimization backend lives in ``swanx.fitting.bo``.
This root module remains so older imports such as ``import swanx.bo`` keep
working.
"""

from .fitting.bo import *  # noqa: F401,F403