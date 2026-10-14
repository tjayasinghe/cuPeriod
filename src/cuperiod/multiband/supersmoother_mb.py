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
in mean absolute deviation across all bands: a band that varies little (relative to
its errors, when it has them) carries proportionally little weight, and with one band
the combination collapses to the single-band score.

Errors are used only when *every* contributing band has them (the rule of the
multi-band GLS); otherwise every band runs unweighted (``dy = 1``). With a mix, an
error-less band's ``B_k`` would be in magnitudes and the others' in sigmas, which
effectively drops the error-less band. Unweighted, ``B_k`` is the band's mean
absolute deviation in its own units, as in gatspy with ``dy = 1``. An exactly
constant band has nothing to reduce and is left out of the combination.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import SuperSmootherSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.supersmoother import _is_constant, supersmoother_score

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
        true period). ``meta["bands"]`` lists the participating bands.

    Raises
    ------
    InsufficientDataError
        If too few finite points remain across usable bands, or the stacked
        baseline is empty.
    """
    finite = mblc.finite()
    used = [
        (name, lc) for name, lc in finite.bands.items() if lc.n >= _MIN_BAND_POINTS
    ]
    n_total = sum(lc.n for _, lc in used)
    if not used or n_total < settings.min_detections:
        raise InsufficientDataError(
            f"SUPERSMOOTHER multiband: {n_total} finite points across usable "
            f"bands < min_detections {settings.min_detections}"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("SUPERSMOOTHER multiband: no usable time baseline")
    use_errors = all(lc.error is not None for _, lc in used)

    periods = grid.period
    combined = np.zeros(periods.size, dtype=np.float64)
    weight_total = 0.0
    for _, lc in used:
        if _is_constant(lc.value):
            # A constant band carries no deviation to reduce. Test the data, not
            # B_k: the weighted mean of a constant is rarely bit-equal to it, which
            # leaves a rounding-level B_k > 0 on a meaningless score. The test is
            # the score's own (relative, a few ulps), so a band constant only to
            # rounding is left out of the weight sum too, not just scored zero.
            continue
        dy = np.asarray(lc.error, dtype=np.float64) if use_errors else None
        inv_dy = np.ones(lc.n, dtype=np.float64) if dy is None else 1.0 / dy
        w = inv_dy * inv_dy
        mu = float(np.dot(w, lc.value) / w.sum())
        b_k = float(np.mean(np.abs(lc.value - mu) * inv_dy))
        if not np.isfinite(b_k) or b_k <= 0.0:
            continue
        score = supersmoother_score(
            lc.time,
            lc.value,
            dy,
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
        meta={**dict(finite.meta), "bands": tuple(name for name, _ in used)},
    )


__all__ = ["supersmoother_multiband"]
