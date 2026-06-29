"""Periodogram methods and the method registry.

Importing this package registers every built-in method (GLS, BLS, ...). Downstream code
should look methods up by name via :func:`get_method` rather than importing the classes
directly, so the orchestration stays method-agnostic.
"""

from __future__ import annotations

# Importing each method module registers it under its uppercase name.
from cuperiod.methods import bls as bls  # noqa: F401
from cuperiod.methods import conditional_entropy as conditional_entropy  # noqa: F401
from cuperiod.methods import gls as gls  # noqa: F401
from cuperiod.methods import pdm as pdm  # noqa: F401
from cuperiod.methods import string_length as string_length  # noqa: F401
from cuperiod.methods.base import (
    MethodInfo,
    PeriodogramMethod,
    get_method,
    list_methods,
    method_names,
    register,
    registered_methods,
)

__all__ = [
    "MethodInfo",
    "PeriodogramMethod",
    "get_method",
    "list_methods",
    "method_names",
    "register",
    "registered_methods",
]
