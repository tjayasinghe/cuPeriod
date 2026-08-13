"""Multi-band Phase Dispersion Minimization.

Every filter of the same star shares one period but keeps its *own* mean light curve:
the mean magnitude, the amplitude and even the folded shape are filter-dependent, so
nothing is gained by forcing the bands onto a common curve. Each band is therefore
phase-folded and binned on its own (Stellingwerf 1978) and only the resulting
*dispersions* are pooled,

    Theta_mb(f) = sum_k w_k Theta_k(f) / sum_k w_k,    w_k = max(n_k - n_bins, 1),

with ``Theta_k`` the single-band Stellingwerf ratio of band ``k`` and ``n_k`` its finite
point count. A band too sparse or too noisy to pin the period down alone still pulls its
weight, and — as in the single-band case — the statistic collapses at the true period,
so this is a *minimization* method.

The weights are the bands' within-bin degrees of freedom under Stellingwerf's
fixed-dof convention (``n_k`` points spread over ``n_bins`` bins leave
``n_k - n_bins`` of them, floored at one for a band with barely more points than
bins; the realized denominator differs only where a fold leaves bins empty). With
those weights the pooled statistic is *exactly* the pooled within-bin sum of squares
over the pooled degrees of freedom of the per-band **standardized** data:
``Theta_k = s_k^2 / sigma_k^2`` already divides band ``k``'s scatter by that band's
own variance, so standardizing each band before pooling is implicit and needs no
separate step. That matters because filters differ in amplitude and photometric
precision; pooling raw sums of squares would let the noisiest band decide the period
on its own.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import PDMSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.pdm import pdm_theta


def pdm_multiband_theta(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: PDMSettings,
    backend: str,
) -> Periodogram:
    """Compute the pooled multi-band PDM statistic on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared trial grid (built from the stacked baseline); used as periods.
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : PDMSettings
        PDM settings (bin count, covers, batching, ...); a band participates when it
        has at least ``n_bins + 2`` finite points.
    backend : str
        Concrete backend (``"numpy"``, ``"numba"``, ``"cupy"``, ``"torch:<device>"``).

    Returns
    -------
    Periodogram
        Degrees-of-freedom-weighted mean of the per-band ``Theta`` (minimized at the
        true period).

    Raises
    ------
    InsufficientDataError
        If no band has ``n_bins + 2`` finite points, or the stacked baseline is empty.
    """
    finite = mblc.finite()
    min_points = settings.n_bins + 2
    bands = [lc for lc in finite.bands.values() if lc.n >= min_points]
    if not bands:
        raise InsufficientDataError(
            f"PDM multiband: no band has {min_points} finite points "
            f"(n_bins {settings.n_bins} + 2)"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("PDM multiband: no usable time baseline")

    periods = grid.period
    pooled = np.zeros(periods.size, dtype=np.float64)
    weight_total = 0.0
    n_total = 0
    for lc in bands:
        theta = pdm_theta(
            lc.time,
            lc.value,
            periods,
            n_bins=settings.n_bins,
            n_covers=settings.n_covers,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        weight = float(max(lc.n - settings.n_bins, 1))
        pooled += weight * theta
        weight_total += weight
        n_total += lc.n
    power = pooled / weight_total

    return Periodogram.from_spectrum(
        method="PDM",
        backend=backend,
        frequency=1.0 / periods,
        power=power,
        objective_sense="min",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["pdm_multiband_theta"]
