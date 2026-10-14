"""Core, method-agnostic building blocks shared by every periodogram."""

from __future__ import annotations

from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.config import (
    BatchSettings,
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    PdotSettings,
    PreWhitenSettings,
    SessionSettings,
    SpacingSettings,
    StringLengthSettings,
    SuperSmootherSettings,
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
from cuperiod.core.grid import (
    GridSizeWarning,
    GridSpec,
    GridTooLargeError,
    check_grid_size,
    default_frequency_grid,
    log_period_grid,
    pseudo_nyquist_frequency,
    regular_cadence_nyquist,
    uniform_frequency_grid,
    uniform_grid_size,
)
from cuperiod.core.lightcurve import (
    LightCurve,
    MultiBandLightCurve,
    raise_if_unusable,
)
from cuperiod.core.result import MultiResult, Peak, Periodogram

__all__ = [
    "BLSSettings",
    "BackendUnavailableError",
    "BatchSettings",
    "CESettings",
    "ColumnMap",
    "ColumnResolutionError",
    "CuPeriodError",
    "Domain",
    "GLSSettings",
    "GpuInfo",
    "GridSizeWarning",
    "GridSpec",
    "GridTooLargeError",
    "InsufficientDataError",
    "LightCurve",
    "MHAOVSettings",
    "MultiBandLightCurve",
    "MultiResult",
    "PDMSettings",
    "PdotSettings",
    "Peak",
    "Periodogram",
    "PreWhitenSettings",
    "SessionSettings",
    "SpacingSettings",
    "StringLengthSettings",
    "SuperSmootherSettings",
    "TLSSettings",
    "UnknownMethodError",
    "check_grid_size",
    "coerce_domain",
    "default_frequency_grid",
    "free_gpu_memory",
    "gpu_info",
    "log_period_grid",
    "pseudo_nyquist_frequency",
    "raise_if_unusable",
    "regular_cadence_nyquist",
    "suggest_gpu_workers",
    "uniform_frequency_grid",
    "uniform_grid_size",
]
