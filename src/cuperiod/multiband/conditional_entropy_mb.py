"""Multi-band conditional-entropy period search.

Conditional entropy asks how well phase predicts brightness (Graham et al. 2013). That
question is filter-local: the g-band and r-band amplitudes, means and colours differ, so
a single joint phase-magnitude histogram would be smeared by the band offsets alone and
would look disordered at *every* trial period. Each band therefore gets its own
``(phase, magnitude)`` histogram — the single-band kernel rescales magnitudes to the
band's own range internally, which is exactly the per-band standardization wanted here —
and only the resulting entropies are pooled,

    H_mb(f) = sum_k n_k H_k(f) / sum_k n_k,

with ``H_k`` the conditional entropy of band ``k`` and ``n_k`` its finite point count.
Since ``H_k`` is itself an average over that band's points (its Shannon sum divided by
``n_k``), weighting by ``n_k`` makes ``H_mb`` the conditional entropy per observation of
the whole data set: a well-sampled band counts for as much as its observations are
worth, and a handful of points in a third filter cannot outvote it. Ordered folds give
low entropy in every band at once, so this is a *minimization* method.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import CESettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.conditional_entropy import conditional_entropy

#: A band needs at least this many finite points to fill a usable 2-D histogram.
_MIN_BAND_POINTS = 8


def ce_multiband_entropy(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: CESettings,
    backend: str,
) -> Periodogram:
    """Compute the pooled multi-band conditional entropy on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared trial grid (built from the stacked baseline); used as periods.
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : CESettings
        CE settings (histogram resolution, batching, ...); a band participates when it
        has at least ``max(n_phase_bins, 8)`` finite points.
    backend : str
        Concrete backend (``"numpy"``, ``"numba"``, ``"cupy"``, ``"torch:<device>"``).

    Returns
    -------
    Periodogram
        Point-count-weighted mean of the per-band entropies (minimized at the true
        period).

    Raises
    ------
    InsufficientDataError
        If no band has enough finite points, or the stacked baseline is empty.
    """
    finite = mblc.finite()
    min_points = max(settings.n_phase_bins, _MIN_BAND_POINTS)
    bands = [lc for lc in finite.bands.values() if lc.n >= min_points]
    if not bands:
        raise InsufficientDataError(
            f"CE multiband: no band has {min_points} finite points"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("CE multiband: no usable time baseline")

    periods = grid.period
    pooled = np.zeros(periods.size, dtype=np.float64)
    n_total = 0
    for lc in bands:
        entropy = conditional_entropy(
            lc.time,
            lc.value,
            periods,
            n_phase_bins=settings.n_phase_bins,
            n_mag_bins=settings.n_mag_bins,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        pooled += float(lc.n) * entropy
        n_total += lc.n
    power = pooled / float(n_total)

    return Periodogram.from_spectrum(
        method="CE",
        backend=backend,
        frequency=1.0 / periods,
        power=power,
        objective_sense="min",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["ce_multiband_entropy"]
