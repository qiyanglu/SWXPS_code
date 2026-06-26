"""Compatibility shim for optimizer-independent fitting helpers.

The maintained implementation lives in :mod:`swanx.fitting.core`. New internal
imports should use that module or the public :mod:`swanx.fitting` package.
"""

from .fitting.core import *  # noqa: F401,F403