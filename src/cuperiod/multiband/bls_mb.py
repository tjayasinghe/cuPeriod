"""Multi-band Box Least Squares.

Bands of the same star share one ``(period, duration, mid-transit time)`` ephemeris but
keep independent depths, since a real eclipse/transit appears at the same phase in every
filter (with a filter-dependent depth). Each band is searched on a common period/phase
grid; the per-period statistic is the quadrature sum of the bands' depth-SNRs (a
likelihood-ratio stack), and the reported box geometry at each period comes from the
band detecting it most strongly.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import BLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod

#: A band needs at least this many finite points to contribute a box search.
_MIN_BAND_POINTS = 3


def bls_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: BLSSettings,
    backend: str,
    method: PeriodogramMethod,
) -> Periodogram:
    """Compute the multi-band BLS statistic on a shared ephemeris grid.

    Parameters
    ----------
    grid : GridSpec
        Unused for segmentation (segments are rebuilt from the stacked baseline); kept
        for interface symmetry.
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : BLSSettings
        BLS settings.
    backend : str
        Concrete backend (``"numpy"``/``"astropy"``/``"cupy"``).
    method : PeriodogramMethod
        The owning :class:`~cuperiod.methods.bls.BLSMethod` (unused; reserved).

    Returns
    -------
    Periodogram
        Combined statistic with per-period box geometry from the strongest band.
    """
    from cuperiod.methods.bls import _SEGMENT_FIELDS, _segment_grids, _segment_power

    finite = mblc.finite()
    bands = [
        lc.in_domain(Domain.FLUX)
        for lc in finite.bands.values()
        if lc.n >= _MIN_BAND_POINTS
    ]
    if not bands:
        raise InsufficientDataError("BLS multiband: no band has enough finite points")
    n_total = sum(lc.n for lc in bands)

    stacked_time = np.concatenate([lc.time for lc in bands])
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("BLS multiband: no usable time baseline")
    segments = _segment_grids(baseline, settings)
    if not segments:
        raise InsufficientDataError(
            "BLS multiband: baseline too short for the configured period range"
        )

    period_parts: list[FloatArray] = []
    combined_parts: list[FloatArray] = []
    depth_parts: list[FloatArray] = []
    dsnr_parts: list[FloatArray] = []
    dur_parts: list[FloatArray] = []
    t0_parts: list[FloatArray] = []
    band_caches: list[dict[str, object]] = [{} for _ in bands]
    for periods, durations in segments:
        per_band = []
        for lc, device_cache in zip(bands, band_caches, strict=True):
            err = lc.error if lc.error is not None else np.ones_like(lc.value)
            seg = _segment_power(
                backend,  # type: ignore[arg-type]
                lc.time,
                lc.value,
                err,
                periods,
                durations,
                settings,
                device_cache,  # type: ignore[arg-type]
            )
            per_band.append(seg)
        dsnr = np.vstack([np.clip(s["depth_snr"], 0.0, None) for s in per_band])
        depth = np.vstack([s["depth"] for s in per_band])
        dur = np.vstack([s["duration"] for s in per_band])
        t0 = np.vstack([s["transit_time"] for s in per_band])
        combined = np.sum(dsnr**2, axis=0)
        best = np.argmax(dsnr, axis=0)
        cols = np.arange(dsnr.shape[1])
        period_parts.append(periods)
        combined_parts.append(combined)
        dsnr_parts.append(np.sqrt(combined))
        depth_parts.append(depth[best, cols])
        dur_parts.append(dur[best, cols])
        t0_parts.append(t0[best, cols])

    _ = _SEGMENT_FIELDS  # documents the shared per-band field set
    period = np.concatenate(period_parts)
    power = np.nan_to_num(np.concatenate(combined_parts), nan=0.0, posinf=0.0)
    mean, std = float(power.mean()), float(power.std())
    sde = (power - mean) / std if std > 0.0 else np.zeros_like(power)
    extras = {
        "depth": np.concatenate(depth_parts),
        "depth_snr": np.concatenate(dsnr_parts),
        "duration": np.concatenate(dur_parts),
        "t0": np.concatenate(t0_parts),
        "sde": sde,
    }
    return Periodogram.from_spectrum(
        method="BLS",
        backend=backend,
        frequency=1.0 / period,
        power=power,
        objective_sense="max",
        n_samples=n_total,
        baseline=baseline,
        extras=extras,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["bls_multiband_power"]
