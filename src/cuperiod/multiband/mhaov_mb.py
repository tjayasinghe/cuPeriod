"""Multi-band multiharmonic Analysis of Variance.

Each band of the same star is fit with its own trigonometric polynomial (independent
harmonic amplitudes and phases) at a *shared* trial frequency, and the per-band
regression and residual sums of squares are pooled into a single F-statistic. This lets
a periodic signal be detected jointly across filters even when no single band constrains
it well, while still allowing the (filter-dependent) harmonic amplitudes to differ.

The sums are pooled in each band's own variance units: band ``k`` is centered and
divided by its sample standard deviation ``s_k`` before the fit, so its regression and
residual sums enter as ``SS_k / s_k^2`` (its total sum of squares becomes ``n_k - 1``).
Raw sums of squares are in the band's magnitude or flux units squared, which let the
noisiest or largest-scale band decide the period on its own; the other pooled methods
standardize per band in the same spirit (see :mod:`cuperiod.multiband.pdm_mb`). The F
ratio of a single band is shift- and scale-free, so one band still collapses to the
single-band statistic, and an exactly constant band — no variance to standardize by —
is left out of the pooled sums.
"""

from __future__ import annotations

import math

from cuperiod.core._typing import FloatArray
from cuperiod.core.config import MHAOVSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.mhaov import aov_multiband_power


def _standardized(value: FloatArray) -> FloatArray | None:
    """``(value - mean) / s``, ``s`` the sample std; ``None`` for a constant band.

    The variance is computed exactly as the AOV kernel computes its total sum of
    squares (about the plain mean), so the standardized band's total is ``n - 1`` up
    to round-off. The F ratio is invariant to both the shift and the scale (the fit
    has a constant term), but centering also spares the kernel the cancellation of
    a large mean against a small scatter, so the pooled statistic does not depend on
    the bands' mean levels even at round-off.
    """
    if float(value.max()) == float(value.min()):
        return None
    centered: FloatArray = value - float(value.mean())
    total_ss = float((centered * centered).sum())
    return centered / math.sqrt(total_ss / (value.size - 1))


def mhaov_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: MHAOVSettings,
    backend: str,
) -> Periodogram:
    """Compute the pooled multi-band MHAOV power on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared frequency grid (built from the stacked baseline).
    mblc : MultiBandLightCurve
        Two or more bands of one star; a band contributes when it has at least
        ``2 * n_harmonics + 2`` finite points.
    settings : MHAOVSettings
        MHAOV settings (harmonic order, backend, ...).
    backend : str
        Concrete backend (``"numpy"``, ``"numba"``, ``"cupy"`` or
        ``"torch:<device>"``).

    Returns
    -------
    Periodogram
        Pooled AOV F-statistic of the variance-standardized bands.
        ``meta["bands"]`` lists the bands that passed the point-count floor.

    Raises
    ------
    InsufficientDataError
        If no band has enough finite points for the harmonic order, the usable
        bands hold fewer than ``min_detections`` points, or the baseline is empty.
    """
    finite = mblc.finite()
    min_points = 2 * settings.n_harmonics + 2
    used = [(name, lc) for name, lc in finite.bands.items() if lc.n >= min_points]
    if not used:
        raise InsufficientDataError(
            "MHAOV multiband: no band has enough finite points for "
            f"{settings.n_harmonics} harmonics"
        )
    n_total = sum(lc.n for _, lc in used)
    if n_total < settings.min_detections:
        raise InsufficientDataError(
            f"MHAOV multiband: {n_total} finite points across usable bands "
            f"< min_detections {settings.min_detections}"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("MHAOV multiband: no usable time baseline")

    bands: list[tuple[FloatArray, FloatArray]] = []
    for _, lc in used:
        scaled = _standardized(lc.value)
        if scaled is not None:
            bands.append((lc.time, scaled))

    frequency = grid.frequency
    power = aov_multiband_power(
        frequency,
        bands,
        n_harmonics=settings.n_harmonics,
        backend=backend,  # type: ignore[arg-type]
        batch=settings.batch_periods,
        precision=settings.precision,
    )
    return Periodogram.from_spectrum(
        method="MHAOV",
        backend=backend,
        frequency=frequency,
        power=power,
        objective_sense="max",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": tuple(name for name, _ in used)},
    )


__all__ = ["mhaov_multiband_power"]
