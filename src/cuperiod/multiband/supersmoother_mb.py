"""Multi-band SuperSmoother.

Every filter of the same star shares one trial period but keeps its own folded
shape — SuperSmoother is fully non-parametric, so there is no shared-phase model to
pool into. Following gatspy's ``SuperSmootherMultiband``, each band is smoothed
independently on the shared grid and the per-band scores are combined with
baseline-error weights,

    score_mb(f) = sum_k B_k score_k(f) / sum_k B_k,
    B_k = mean_i |y_i - mu_k| / dy_i  over band k,

with ``mu_k`` band ``k``'s inverse-variance weighted mean. ``B_k`` is band ``k``'s
mean absolute standardized deviation about its own mean — the denominator of that
band's score — so the combined statistic is exactly the *total* fractional reduction
in mean absolute deviation across all bands: a noisy or flat band contributes little
weight, and with one band the combination collapses to the single-band score.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import SuperSmootherSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.supersmoother import supersmoother_score

#: A band needs this many finite points for its smallest (3-point) span windows.
_MIN_BAND_POINTS = 3


def supersmoother_multiband(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: SuperSmootherSettings,
    backend: str,
) -> Periodogram:
    """Compute the baseline-weighted multi-band SuperSmoother score on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared trial grid (built from the stacked baseline); used as periods.
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : SuperSmootherSettings
        SuperSmoother settings (spans, bass enhancement, batching, ...); a band
        participates when it has at least 3 finite points.
    backend : str
        Concrete backend (``"numpy"``, ``"numba"``, ``"cupy"``, ``"torch:<device>"``).

    Returns
    -------
    Periodogram
        Baseline-error-weighted mean of the per-band scores (maximized at the
        true period).

    Raises
    ------
    InsufficientDataError
        If too few finite points remain across usable bands, or the stacked
        baseline is empty.
    """
    finite = mblc.finite()
    bands = [lc for lc in finite.bands.values() if lc.n >= _MIN_BAND_POINTS]
    n_total = sum(lc.n for lc in bands)
    if not bands or n_total < settings.min_detections:
        raise InsufficientDataError(
            f"SUPERSMOOTHER multiband: {n_total} finite points across usable "
            f"bands < min_detections {settings.min_detections}"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("SUPERSMOOTHER multiband: no usable time baseline")

    periods = grid.period
    combined = np.zeros(periods.size, dtype=np.float64)
    weight_total = 0.0
    for lc in bands:
        if lc.error is None:
            inv_dy = np.ones(lc.n, dtype=np.float64)
        else:
            inv_dy = 1.0 / np.asarray(lc.error, dtype=np.float64)
        w = inv_dy * inv_dy
        mu = float(np.dot(w, lc.value) / w.sum())
        b_k = float(np.mean(np.abs(lc.value - mu) * inv_dy))
        if not np.isfinite(b_k) or b_k <= 0.0:
            continue  # a constant band carries no deviation to reduce
        score = supersmoother_score(
            lc.time,
            lc.value,
            lc.error,
            periods,
            primary_spans=settings.primary_spans,
            middle_span=settings.middle_span,
            final_span=settings.final_span,
            bass_enhancement=settings.bass_enhancement,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        combined += b_k * score
        weight_total += b_k
    power = combined / weight_total if weight_total > 0.0 else combined

    return Periodogram.from_spectrum(
        method="SUPERSMOOTHER",
        backend=backend,
        frequency=1.0 / periods,
        power=power,
        objective_sense="max",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["supersmoother_multiband"]
