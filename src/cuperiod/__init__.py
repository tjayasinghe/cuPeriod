"""cuPeriod: optimized, GPU-accelerated periodograms for astronomy.

Quick start
-----------
>>> import cuperiod as cup                                    # doctest: +SKIP
>>> pg = cup.periodogram((time, mag, mag_err), "GLS")         # doctest: +SKIP
>>> pg.best_period()                                          # doctest: +SKIP
>>> for peak in pg.best_periods(10):                          # doctest: +SKIP
...     print(peak.period, peak.power)

The recommended import alias is ``cup``. The two entry points are
:func:`periodogram` (a single light curve) and :func:`batch_periodograms` (many).
"""

from __future__ import annotations

# Importing the methods package registers the built-in methods (GLS, BLS, ...).
from cuperiod import methods as methods
from cuperiod.api import best_periods, periodogram, to_input
from cuperiod.batch import BatchSummary, batch_periodograms
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import (
    BatchSettings,
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    StringLengthSettings,
    TLSSettings,
)
from cuperiod.core.device import GpuInfo, free_gpu_memory, gpu_info, suggest_gpu_workers
from cuperiod.core.errors import (
    BackendUnavailableError,
    ColumnResolutionError,
    CuPeriodError,
    InsufficientDataError,
    UnknownMethodError,
)
from cuperiod.core.grid import GridSpec, log_period_grid, uniform_frequency_grid
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import MultiResult, Peak, Periodogram
from cuperiod.methods.base import MethodInfo, get_method, list_methods, method_names

__version__ = "0.1.0"

__all__ = [
    "BLSSettings",
    "BackendUnavailableError",
    "BatchSettings",
    "BatchSummary",
    "CESettings",
    "ColumnMap",
    "ColumnResolutionError",
    "CuPeriodError",
    "Domain",
    "GLSSettings",
    "GpuInfo",
    "GridSpec",
    "InsufficientDataError",
    "LightCurve",
    "MHAOVSettings",
    "MethodInfo",
    "MultiBandLightCurve",
    "MultiResult",
    "PDMSettings",
    "Peak",
    "Periodogram",
    "StringLengthSettings",
    "TLSSettings",
    "UnknownMethodError",
    "__version__",
    "batch_periodograms",
    "best_periods",
    "free_gpu_memory",
    "get_method",
    "gpu_info",
    "list_methods",
    "log_period_grid",
    "method_names",
    "periodogram",
    "suggest_gpu_workers",
    "to_input",
    "uniform_frequency_grid",
]
