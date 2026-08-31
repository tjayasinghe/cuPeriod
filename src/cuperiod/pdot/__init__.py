"""Period-change (dP/dt) tools: joint searches, phase tracking, ephemeris fits.

The v1.3 subsystem for measuring period *derivatives*:

- :func:`pdot_search` — joint (period, dP/dt) grid search by time-warping any
  registered periodogram method (detection / localization).
- :func:`phase_track` — segmented template-phase tracking at a fixed period:
  an automated O-C diagram with per-segment uncertainties (measurement).
- :func:`fit_ephemeris` — quadratic-ephemeris fits to O-C style timing data,
  including user-supplied times of minimum/maximum.
"""

from cuperiod.pdot.result import (
    DAYS_PER_MYR,
    SECONDS_PER_YEAR,
    PdotBest,
    PdotSearchResult,
)
from cuperiod.pdot.scan import pdot_search, weighted_epoch
from cuperiod.pdot.track import (
    EphemerisFit,
    PhaseTemplate,
    PhaseTrackResult,
    SegmentShift,
    fit_ephemeris,
    fit_template,
    phase_track,
)

__all__ = [
    "DAYS_PER_MYR",
    "SECONDS_PER_YEAR",
    "EphemerisFit",
    "PdotBest",
    "PdotSearchResult",
    "PhaseTemplate",
    "PhaseTrackResult",
    "SegmentShift",
    "fit_ephemeris",
    "fit_template",
    "pdot_search",
    "phase_track",
    "weighted_epoch",
]
