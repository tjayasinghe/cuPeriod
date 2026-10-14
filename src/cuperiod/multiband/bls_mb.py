"""Multi-band Box Least Squares.

Bands of the same star share one trial *period*: each band is box-searched on its own
over the common period grid (its own best duration and mid-transit time at every
period), and the per-period statistic is the quadrature sum of the bands' best
depth-SNRs, ``sum_k max(SNR_k, 0)^2`` — a likelihood-ratio stack of independent
per-band searches. Only the period is shared: a band's box is not forced onto another
band's duration or phase, so the stack also rewards noise maxima at unrelated phases,
and in sparse data it is weaker than a joint search with one shared box would be. The
reported box geometry (depth, duration, mid-transit time) at each period comes from
the band detecting it most strongly; ``extras["band_index"]`` names that band (an
index into ``meta["bands"]``), and its depth is in that band's own flux units.

The stack is only an SNR stack when every band's depth-SNR is dimensionless, so a
band's errors are used only when *every* contributing band has them (the same rule
as the multi-band GLS). Otherwise every band is searched with a constant error equal
to its own robust scatter (1.4826 × the median absolute deviation of its flux), which
keeps the bands commensurate whatever their flux scale or zero point.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import BLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod

#: A band needs at least this many finite points to contribute a box search.
_MIN_BAND_POINTS = 3

#: Scale factor turning a median absolute deviation into a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826


def _robust_scatter(flux: FloatArray) -> float:
    """A band's noise scale for when it has no errors: ``1.4826 * MAD`` of its flux.

    The median absolute deviation ignores the few in-eclipse points, so the scale
    tracks the noise rather than the signal. A band whose flux is mostly one repeated
    value (MAD 0) falls back to the standard deviation, and a constant band to 1 — it
    has no box to detect, so its scale is immaterial.
    """
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(flux - np.median(flux))))
    if not (np.isfinite(sigma) and sigma > 0.0):
        sigma = float(np.std(flux))
    if not (np.isfinite(sigma) and sigma > 0.0):
        sigma = 1.0
    return sigma


def _band_errors(bands: list[LightCurve]) -> list[FloatArray]:
    """Per-band flux errors that make every band's depth-SNR dimensionless.

    The quadrature stack weights a band by ``SNR^2``, so a band searched with unit
    errors would count by its raw flux scale squared (``10**(0.8 * dmag)``) and a
    bright, noisy band would outvote a faint, precise one. Real errors are used when
    every band has them; otherwise all bands get their own constant robust scatter —
    mixing real errors with a scatter estimate would weigh the two kinds of band
    against each other on different footings (the multi-band GLS applies the same
    all-or-nothing rule).
    """
    if all(lc.error is not None for lc in bands):
        return [np.asarray(lc.error, dtype=np.float64) for lc in bands]
    return [np.full(lc.n, _robust_scatter(lc.value)) for lc in bands]


def bls_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: BLSSettings,
    backend: str,
    method: PeriodogramMethod,
) -> Periodogram:
    """Compute the multi-band BLS statistic on a shared period grid.

    Parameters
    ----------
    grid : GridSpec
        Trial grid, searched exactly as given and segmented as single-band BLS
        segments it (:func:`~cuperiod.methods.bls._grid_segments`): the default
        grid at its recorded log segments, any other grid into log segments whose
        box durations track the period.
    mblc : MultiBandLightCurve
        Two or more bands of one star; a band contributes when it has at least 3
        finite points.
    settings : BLSSettings
        BLS settings.
    backend : str
        Concrete backend (``"numba"``, ``"numpy"``, ``"astropy"``, ``"cupy"`` or
        ``"torch:<device>"``).
    method : PeriodogramMethod
        The owning :class:`~cuperiod.methods.bls.BLSMethod` (unused; reserved).

    Returns
    -------
    Periodogram
        The per-period sum of the bands' squared depth-SNRs, with the box geometry
        of the strongest band (``extras["band_index"]`` indexes ``meta["bands"]``,
        the contributing bands).

    Raises
    ------
    InsufficientDataError
        If no band has 3 finite points, the contributing bands hold fewer than
        ``min_detections`` points, the grid is empty, or the baseline is empty.
    """
    from cuperiod.methods.bls import _SEGMENT_FIELDS, _grid_segments, _segment_power

    finite = mblc.finite()
    used = [
        (name, lc.in_domain(Domain.FLUX))
        for name, lc in finite.bands.items()
        if lc.n >= _MIN_BAND_POINTS
    ]
    if not used:
        raise InsufficientDataError("BLS multiband: no band has enough finite points")
    bands = [lc for _, lc in used]
    n_total = sum(lc.n for lc in bands)
    if n_total < settings.min_detections:
        raise InsufficientDataError(
            f"BLS multiband: {n_total} finite points across usable bands "
            f"< min_detections {settings.min_detections}"
        )
    stacked_time = np.concatenate([lc.time for lc in bands])
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("BLS multiband: no usable time baseline")
    segments = _grid_segments(grid, settings)
    if not segments:
        raise InsufficientDataError("BLS multiband: empty trial grid")
    errors = _band_errors(bands)

    period_parts: list[FloatArray] = []
    combined_parts: list[FloatArray] = []
    depth_parts: list[FloatArray] = []
    dsnr_parts: list[FloatArray] = []
    dur_parts: list[FloatArray] = []
    t0_parts: list[FloatArray] = []
    band_parts: list[FloatArray] = []
    band_caches: list[dict[str, object]] = [{} for _ in bands]
    for periods, durations in segments:
        per_band = []
        for lc, err, device_cache in zip(bands, errors, band_caches, strict=True):
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
        band_parts.append(best.astype(np.float64))

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
        "band_index": np.concatenate(band_parts),
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
        meta={**dict(finite.meta), "bands": tuple(name for name, _ in used)},
    )


__all__ = ["bls_multiband_power"]
