"""Period-change (dP/dt) tools: joint searches, phase tracking, ephemeris fits.

The v1.3 subsystem for measuring period *derivatives*:

- :func:`pdot_search` — joint (period, dP/dt) grid search by time-warping any
  registered periodogram method (detection / localization).
- :func:`phase_track` — segmented template-phase tracking at a fixed period:
  an automated O-C diagram with per-segment uncertainties (measurement).
- :func:`fit_ephemeris` — quadratic-ephemeris fits to O-C style timing data,
  including user-supplied times of minimum/maximum.
- :func:`sliding_periodogram` — the ordinary periodogram on a sliding window: a
  time-frequency map for period wander, glitches, and drift sanity checks.
- :func:`eddington_plakidis` — the classical test separating random
  cycle-to-cycle period fluctuation from a genuine secular change.
- :func:`wwz` — the Weighted Wavelet Z-transform: a time-frequency map whose
  window scales with the trial frequency, the standard tool for the period
  wander of Miras and semi-regulars.
"""

from cuperiod.pdot.result import (
    DAYS_PER_MYR,
    SECONDS_PER_YEAR,
    PdotBest,
    PdotSearchResult,
)
from cuperiod.pdot.scan import pdot_search, weighted_epoch
from cuperiod.pdot.sliding import SlidingPeriodogram, sliding_periodogram
from cuperiod.pdot.stability import EddingtonPlakidisResult, eddington_plakidis
from cuperiod.pdot.track import (
    EphemerisFit,
    PhaseTemplate,
    PhaseTrackResult,
    SegmentShift,
    fit_ephemeris,
    fit_template,
    phase_track,
)
from cuperiod.pdot.wwz import WWZResult, wwz

__all__ = [
    "DAYS_PER_MYR",
    "SECONDS_PER_YEAR",
    "EddingtonPlakidisResult",
    "EphemerisFit",
    "PdotBest",
    "PdotSearchResult",
    "PhaseTemplate",
    "PhaseTrackResult",
    "SegmentShift",
    "SlidingPeriodogram",
    "WWZResult",
    "eddington_plakidis",
    "fit_ephemeris",
    "fit_template",
    "pdot_search",
    "phase_track",
    "sliding_periodogram",
    "weighted_epoch",
    "wwz",
]
