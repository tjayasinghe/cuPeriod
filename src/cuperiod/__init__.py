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
    PdotSettings,
    PreWhitenSettings,
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
from cuperiod.pdot import (
    EddingtonPlakidisResult,
    EphemerisFit,
    LTTEFit,
    LTTESearchResult,
    PdotBest,
    PdotSearchResult,
    PdotSensitivity,
    PeriodLawFit,
    PeriodPoint,
    PeriodProfile,
    PhaseTemplate,
    PhaseTrackResult,
    SegmentShift,
    SlidingPeriodogram,
    WWZResult,
    available_laws,
    available_models,
    batch_pdot,
    compare_period_laws,
    convert_pdot,
    detectable_pdot,
    detrend,
    eddington_plakidis,
    fit_ephemeris,
    fit_period_law,
    ltte_search,
    mass_loss_rate,
    mass_transfer_rate,
    pdot_from_mass_loss,
    pdot_from_mass_transfer,
    pdot_search,
    period_profile,
    phase_track,
    running_median,
    sliding_periodogram,
    weighted_epoch,
    wwz,
)
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
    __version__ = "1.3.0"

__all__ = [
    "__version__",
    "AliasCandidate",
    "AliasReport",
    "AmplitudeSpectrum",
    "BackendUnavailableError",
    "BatchSettings",
    "BatchSummary",
    "BLSSettings",
    "CESettings",
    "ColumnMap",
    "ColumnResolutionError",
    "Combination",
    "CuPeriodError",
    "Domain",
    "EddingtonPlakidisResult",
    "EphemerisFit",
    "LTTEFit",
    "LTTESearchResult",
    "GLSSettings",
    "GpuInfo",
    "GridSpec",
    "InsufficientDataError",
    "LightCurve",
    "MethodInfo",
    "MHAOVSettings",
    "MultibandFAP",
    "MultiBandLightCurve",
    "MultiResult",
    "MultiSineFit",
    "PDMSettings",
    "PdotBest",
    "PdotSearchResult",
    "PdotSensitivity",
    "PdotSettings",
    "Peak",
    "PeriodLawFit",
    "Periodogram",
    "PeriodPoint",
    "PeriodProfile",
    "PeriodSpacingSeries",
    "PhaseTemplate",
    "PhaseTrackResult",
    "PreWhitenResult",
    "PreWhitenSettings",
    "SegmentShift",
    "Sinusoid",
    "SlidingPeriodogram",
    "SpacingSettings",
    "SpacingSpectrum",
    "SpectrumEngine",
    "StringLengthSettings",
    "SuperSmootherSettings",
    "TLSSettings",
    "UnknownMethodError",
    "WindowPeak",
    "WWZResult",
    "alias_diagnostics",
    "amplitude_spectrum",
    "available_laws",
    "available_models",
    "baluev_fap",
    "batch_pdot",
    "batch_periodograms",
    "batch_prewhiten",
    "best_periods",
    "buoyancy_radius",
    "compare_period_laws",
    "convert_pdot",
    "detectable_pdot",
    "detrend",
    "echelle",
    "eddington_plakidis",
    "find_period_spacing",
    "fit_ephemeris",
    "fit_multisine",
    "fit_period_law",
    "ltte_search",
    "mass_loss_rate",
    "mass_transfer_rate",
    "pdot_from_mass_loss",
    "pdot_from_mass_transfer",
    "free_gpu_memory",
    "get_method",
    "gpu_info",
    "identify_combinations",
    "list_methods",
    "log_period_grid",
    "method_names",
    "multiband_fap",
    "pdot_search",
    "period_profile",
    "periodogram",
    "phase_track",
    "prewhiten",
    "running_median",
    "sliding_periodogram",
    "spacing_spectrum",
    "spectral_window",
    "suggest_gpu_workers",
    "to_input",
    "uniform_frequency_grid",
    "weighted_epoch",
    "wwz",
]
