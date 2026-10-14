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
- :func:`period_profile` — the period measured independently in consecutive
  intervals: a model-free ``P(t)`` track for when a single rate will not do.
- :func:`fit_period_law` — parametric period-vs-time laws (including the
  finite-time-singularity form used for merging binaries) fitted to that track.
- :func:`batch_pdot` — the whole chain over many light curves, with per-star
  tables, an O-C catalogue, and optional diagnostic figures.
- :func:`detrend` — running-median removal of the slow brightness trend that
  otherwise aliases onto the search grid.
- :func:`ltte_search` — the cyclic O-C: a light-travel-time orbit (a third
  body) fitted jointly with the ephemeris polynomial, so a non-monotonic O-C
  is not mistaken for a secular rate.
- :func:`mass_loss_rate` / :func:`mass_transfer_rate` / :func:`detectable_pdot`
  — what a secular rate would mean physically, and how small a one the
  timing data could have measured.

:mod:`cuperiod.pdot.plots` draws the matching diagnostic figures; it needs
matplotlib (``pip install "cuperiod[plot]"``) and so is imported on demand
rather than re-exported here.
"""

from cuperiod.pdot.batch import batch_pdot, pdot_to_rows
from cuperiod.pdot.detrend import detrend, running_median
from cuperiod.pdot.laws import (
    AUTO_LAWS,
    PeriodLawFit,
    available_laws,
    compare_period_laws,
    fit_period_law,
)
from cuperiod.pdot.ltte import LTTEFit, LTTESearchResult, ltte_delay, ltte_search
from cuperiod.pdot.models import PeriodModel, available_models, get_model
from cuperiod.pdot.physics import (
    PdotSensitivity,
    convert_pdot,
    detectable_pdot,
    mass_loss_rate,
    mass_transfer_rate,
    pdot_from_mass_loss,
    pdot_from_mass_transfer,
    pdot_units,
)
from cuperiod.pdot.profile import PeriodPoint, PeriodProfile, period_profile
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
    "AUTO_LAWS",
    "DAYS_PER_MYR",
    "SECONDS_PER_YEAR",
    "EddingtonPlakidisResult",
    "EphemerisFit",
    "LTTEFit",
    "LTTESearchResult",
    "PdotBest",
    "PdotSearchResult",
    "PdotSensitivity",
    "PeriodLawFit",
    "PeriodModel",
    "PeriodPoint",
    "PeriodProfile",
    "PhaseTemplate",
    "PhaseTrackResult",
    "SegmentShift",
    "SlidingPeriodogram",
    "WWZResult",
    "available_laws",
    "available_models",
    "batch_pdot",
    "compare_period_laws",
    "convert_pdot",
    "detectable_pdot",
    "detrend",
    "eddington_plakidis",
    "fit_ephemeris",
    "fit_period_law",
    "fit_template",
    "get_model",
    "ltte_delay",
    "ltte_search",
    "mass_loss_rate",
    "mass_transfer_rate",
    "pdot_from_mass_loss",
    "pdot_from_mass_transfer",
    "pdot_search",
    "pdot_to_rows",
    "pdot_units",
    "period_profile",
    "phase_track",
    "running_median",
    "sliding_periodogram",
    "weighted_epoch",
    "wwz",
]
