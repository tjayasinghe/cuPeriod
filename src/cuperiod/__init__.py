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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

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
    PreWhitenSettings,
    SpacingSettings,
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
from cuperiod.diagnostics import (
    AliasCandidate,
    AliasReport,
    WindowPeak,
    alias_diagnostics,
)
from cuperiod.methods.base import MethodInfo, get_method, list_methods, method_names
from cuperiod.multiband.fap_mb import MultibandFAP, multiband_fap
from cuperiod.prewhiten import (
    AmplitudeSpectrum,
    Combination,
    MultiSineFit,
    PeriodSpacingSeries,
    PreWhitenResult,
    Sinusoid,
    SpacingSpectrum,
    SpectrumEngine,
    amplitude_spectrum,
    baluev_fap,
    batch_prewhiten,
    buoyancy_radius,
    echelle,
    find_period_spacing,
    fit_multisine,
    identify_combinations,
    prewhiten,
    spacing_spectrum,
    spectral_window,
)

try:
    __version__ = _version("cuperiod")
except PackageNotFoundError:  # pragma: no cover - source tree, no metadata
    __version__ = "1.2.0.dev0"

__all__ = [
    "AliasCandidate",
    "AliasReport",
    "AmplitudeSpectrum",
    "BLSSettings",
    "BackendUnavailableError",
    "BatchSettings",
    "BatchSummary",
    "CESettings",
    "ColumnMap",
    "ColumnResolutionError",
    "Combination",
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
    "MultiSineFit",
    "MultibandFAP",
    "PDMSettings",
    "Peak",
    "Periodogram",
    "PeriodSpacingSeries",
    "PreWhitenResult",
    "PreWhitenSettings",
    "Sinusoid",
    "SpacingSettings",
    "SpacingSpectrum",
    "SpectrumEngine",
    "StringLengthSettings",
    "TLSSettings",
    "UnknownMethodError",
    "WindowPeak",
    "__version__",
    "alias_diagnostics",
    "amplitude_spectrum",
    "baluev_fap",
    "batch_periodograms",
    "batch_prewhiten",
    "best_periods",
    "buoyancy_radius",
    "echelle",
    "find_period_spacing",
    "fit_multisine",
    "free_gpu_memory",
    "get_method",
    "gpu_info",
    "identify_combinations",
    "list_methods",
    "log_period_grid",
    "method_names",
    "multiband_fap",
    "periodogram",
    "prewhiten",
    "spacing_spectrum",
    "spectral_window",
    "suggest_gpu_workers",
    "to_input",
    "uniform_frequency_grid",
]
