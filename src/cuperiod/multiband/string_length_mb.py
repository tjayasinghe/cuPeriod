"""Multi-band string-length period search.

The string length measures how tightly a *folded* light curve joins up (Lafler & Kinman
1965; Dworetsky 1983), and a fold is only smooth within one filter: bands differ in mean
magnitude and amplitude, so stringing all bands together would spend most of the string
hopping between filters rather than tracing the variability. Each band is therefore
folded and strung on its own — the single-band kernel rescales each band's magnitudes to
Dworetsky's span of 0.5, so the phase and magnitude axes contribute comparably in every
filter regardless of its amplitude — and the lengths are pooled by point count,

    L_mb(f) = sum_k n_k L_k(f) / sum_k n_k,

with ``L_k`` the string length of band ``k`` and ``n_k`` its finite point count. The
weighting reflects that a string over ``n_k`` points is a sum of ``n_k`` steps: pooling
by ``n_k`` compares bands on a per-step footing, so a densely sampled band is not
diluted by a sparse one. A shared period shortens every band's string at once, so this
is a *minimization* method.

Like the single-band statistic, the pooled length also dips at integer multiples of the
true period (overlaid copies pack the fold tightly), so bounding the search below the
first subharmonic keeps the fundamental the global minimum.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import StringLengthSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.string_length import string_length

#: A band needs at least this many finite points to make a meaningful string.
_MIN_BAND_POINTS = 8


def string_length_multiband(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: StringLengthSettings,
    backend: str,
) -> Periodogram:
    """Compute the pooled multi-band string length on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared trial grid (built from the stacked baseline); used as periods.
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : StringLengthSettings
        String-length settings (batching, precision, ...); a band participates when it
        has at least 8 finite points.
    backend : str
        Concrete backend (``"numpy"``, ``"numba"``, ``"cupy"``, ``"torch:<device>"``).

    Returns
    -------
    Periodogram
        Point-count-weighted mean of the per-band string lengths (minimized at the true
        period).

    Raises
    ------
    InsufficientDataError
        If no band has enough finite points, or the stacked baseline is empty.
    """
    finite = mblc.finite()
    bands = [lc for lc in finite.bands.values() if lc.n >= _MIN_BAND_POINTS]
    if not bands:
        raise InsufficientDataError(
            f"string-length multiband: no band has {_MIN_BAND_POINTS} finite points"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("string-length multiband: no usable time baseline")

    periods = grid.period
    pooled = np.zeros(periods.size, dtype=np.float64)
    n_total = 0
    for lc in bands:
        length = string_length(
            lc.time,
            lc.value,
            periods,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        pooled += float(lc.n) * length
        n_total += lc.n
    power = pooled / float(n_total)

    return Periodogram.from_spectrum(
        method="STRINGLENGTH",
        backend=backend,
        frequency=1.0 / periods,
        power=power,
        objective_sense="min",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["string_length_multiband"]
