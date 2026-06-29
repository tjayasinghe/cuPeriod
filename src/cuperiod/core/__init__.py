"""Core, method-agnostic building blocks shared by every periodogram."""

from __future__ import annotations

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import (
    BatchSettings,
    BLSSettings,
    GLSSettings,
    PDMSettings,
)
from cuperiod.core.device import GpuInfo, free_gpu_memory, gpu_info, suggest_gpu_workers
from cuperiod.core.errors import (
    BackendUnavailableError,
    ColumnResolutionError,
    CuPeriodError,
    InsufficientDataError,
    UnknownMethodError,
)
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import MultiResult, Peak, Periodogram

__all__ = [
    "BLSSettings",
    "BackendUnavailableError",
    "BatchSettings",
    "ColumnMap",
    "ColumnResolutionError",
    "CuPeriodError",
    "Domain",
    "GLSSettings",
    "GpuInfo",
    "GridSpec",
    "InsufficientDataError",
    "LightCurve",
    "MultiBandLightCurve",
    "MultiResult",
    "PDMSettings",
    "Peak",
    "Periodogram",
    "UnknownMethodError",
    "free_gpu_memory",
    "gpu_info",
    "suggest_gpu_workers",
]
