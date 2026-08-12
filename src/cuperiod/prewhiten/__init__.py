"""Automated, uncertainty-aware pre-whitening for classical pulsators.

Frequency analysis of δ Scuti, γ Doradus and SPB stars is traditionally an interactive
Period04 session: find the tallest peak, fit it, subtract it, look again, and decide by
eye when to stop. That does not survive contact with a TESS full-frame-image catalogue,
let alone PLATO. This subpackage automates the loop end to end and makes every judgement
call an explicit, recorded setting.

.. code-block:: python

    import cuperiod as cup

    solution = cup.prewhiten((time, mag, mag_err))
    print(solution.summary())

    series = cup.find_period_spacing(solution.period, solution.amplitude)
    print(series.summary())

What it provides
----------------
* A batch-capable GPU/NUFFT **amplitude spectrum** (:mod:`~cuperiod.prewhiten.spectrum`)
  whose sampling-only terms are cached, so each pre-whitening iteration costs a single
  transform.
* **Iterative extraction** (:mod:`~cuperiod.prewhiten.engine`) with a simultaneous
  non-linear re-fit of every component after each step.
* **Principled stopping criteria** — Breger signal-to-noise, false-alarm probability,
  ΔBIC, an amplitude floor — combined and always reported.
* **Error propagation** (:mod:`~cuperiod.prewhiten.uncertainty`): least-squares
  covariance including the correlations between close frequencies, the classical
  Montgomery & O'Donoghue formulae, or a residual bootstrap; optionally inflated by the
  Schwarzenberg-Czerny correlation factor.
* **Combination-frequency identification**
  (:mod:`~cuperiod.prewhiten.combinations`) with uncertainty-aware tolerances and
  chance-coincidence rates.
* **g-mode period-spacing tools** (:mod:`~cuperiod.prewhiten.spacing`): comb search,
  tilted-pattern extraction, échelle coordinates, buoyancy radius.
* **Batch pre-whitening** (:mod:`~cuperiod.prewhiten.batch`) over CPU or GPU worker
  pools, written to Parquet/CSV.
"""

from __future__ import annotations

from cuperiod.prewhiten.batch import batch_prewhiten
from cuperiod.prewhiten.combinations import Combination, identify_combinations
from cuperiod.prewhiten.engine import default_prewhiten_grid, prewhiten
from cuperiod.prewhiten.fap import baluev_fap
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid
from cuperiod.prewhiten.spacing import (
    PeriodSpacingSeries,
    SpacingSpectrum,
    buoyancy_radius,
    echelle,
    find_period_spacing,
    spacing_spectrum,
)
from cuperiod.prewhiten.spectrum import (
    AmplitudeSpectrum,
    SpectrumEngine,
    amplitude_spectrum,
    noise_level,
    spectral_window,
)
from cuperiod.prewhiten.uncertainty import (
    Uncertainties,
    analytic_uncertainties,
    bootstrap_uncertainties,
    component_uncertainties,
    correlation_factor,
)

__all__ = [
    "AmplitudeSpectrum",
    "Combination",
    "MultiSineFit",
    "PeriodSpacingSeries",
    "PreWhitenResult",
    "Sinusoid",
    "SpacingSpectrum",
    "SpectrumEngine",
    "Uncertainties",
    "amplitude_spectrum",
    "analytic_uncertainties",
    "baluev_fap",
    "batch_prewhiten",
    "bootstrap_uncertainties",
    "buoyancy_radius",
    "component_uncertainties",
    "correlation_factor",
    "default_prewhiten_grid",
    "echelle",
    "find_period_spacing",
    "fit_multisine",
    "identify_combinations",
    "noise_level",
    "prewhiten",
    "spacing_spectrum",
    "spectral_window",
]
