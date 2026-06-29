"""Multi-band generalized Lomb-Scargle (VanderPlas & Ivezić 2015).

Several filters of the same star are fit jointly with a shared sinusoid on a common
phase plus per-band offset/amplitude terms — the model astropy implements as
``LombScargleMultiband``. This recovers a period even when no single band is sampled
well enough on its own, which is the common situation for multi-survey photometry.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import GLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram


def gls_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: GLSSettings,
) -> Periodogram:
    """Compute the multi-band GLS power on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Frequency grid (built from the stacked baseline).
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : GLSSettings
        GLS settings (``fit_mean`` selects the floating-mean base model).

    Returns
    -------
    Periodogram
        The joint power spectrum (backend ``"astropy"``).

    Raises
    ------
    InsufficientDataError
        If too few finite points remain across all bands.
    """
    from astropy.timeseries import LombScargleMultiband

    finite = mblc.finite()
    time, value, error, band = finite.stacked()
    n = int(time.size)
    if n < settings.min_detections:
        raise InsufficientDataError(
            f"GLS multiband: {n} finite points < min_detections "
            f"{settings.min_detections}"
        )
    baseline = float(time.max() - time.min()) if n else 0.0
    if baseline <= 0.0:
        raise InsufficientDataError("GLS multiband: no usable time baseline")

    ls = LombScargleMultiband(
        time,
        value,
        band,
        error,
        nterms_base=1,
        nterms_band=1,
    )
    frequency = grid.frequency
    power = np.nan_to_num(
        np.asarray(ls.power(frequency), dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return Periodogram.from_spectrum(
        method="GLS",
        backend="astropy",
        frequency=frequency,
        power=power,
        objective_sense="max",
        n_samples=n,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["gls_multiband_power"]
