"""The Eddington-Plakidis test: random period wander vs. secular change.

A smooth-looking O-C curve is not evidence of a smooth period change: a star
whose period fluctuates *randomly* from cycle to cycle accumulates a random
walk in the times of maximum, and a parabola will happily fit the result
(Sterken 2005 demonstrates this with literal dice rolls). Eddington & Plakidis
(1929, MNRAS 90, 65) built the classical discriminator. If ``a(E)`` is the O-C
delay at cycle ``E`` and ``u(x) = |a(E+x) - a(E)|`` the accumulated delay
between two maxima ``x`` cycles apart, then under random per-cycle fluctuation

    <u(x)>**2 = 2*alpha**2 + x*epsilon**2,

where ``alpha`` is the mean (absolute) measurement error of one timing and
``epsilon`` the mean (absolute) random fluctuation of the period per cycle: a
line in ``x`` whose slope measures the wander and whose intercept the timing
noise. A significant slope says the O-C is (at least partly) a random walk and
a parabolic dP/dt fit alone should not be trusted.

Two refinements over the classical recipe, both load-bearing:

- A genuine secular dP/dt also grows ``u(x)``, so the quadratic trend is
  removed first (``detrend=2``; Csornyei et al. 2025 make the need explicit).
- Polynomial detrending *absorbs part of the walk itself* — a random walk
  correlates strongly with smooth polynomials, which is Sterken's warning in
  reverse — biasing the naive slope low by tens of percent. This estimator
  therefore fits the *exact* expectation: with ``A = I - H`` the detrending
  projector, ``E[u(i,j)**2]`` is a known linear combination of the per-cycle
  walk variance and the timing variance (propagated through ``A`` and the
  walk covariance ``min(E_i, E_j)``), and those two variances are solved by
  weighted least squares. It reduces to the classical line when detrending
  distorts nothing, and stays unbiased when it does.

``alpha``/``epsilon`` are reported in the 1929 mean-absolute convention
(Gaussian sigma times ``sqrt(2/pi)``); ``alpha_sigma``/``epsilon_sigma`` give
the Gaussian sigmas.

The uncertainties are those of the estimator itself. The lag-bin means are not
independent measurements -- every one is a quadratic form in the *same*
detrended residuals, and under a walk they rise and fall together -- so the
scatter of the bins about the fitted line says little about how far the fitted
variances can be off. The final weighted fit is a fixed linear map from the
bin means to the two variances, so each variance is ``r^T Q_k r`` for the
detrended residuals ``r`` and a known banded matrix ``Q_k``, and for Gaussian
residuals of covariance ``C`` its variance is the exact ``2 tr(Q_k C Q_k C)``.
The errors use ``C`` at the fitted variances; ``slope_significance`` uses ``C``
of pure timing noise at the observed residual scatter, which makes it a
z-score under the no-wander null.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.errors import InsufficientDataError

#: Fewest pairs a lag bin needs before its mean enters the fit.
_MIN_PAIRS: int = 5

#: Largest event count accepted. No n x n matrix is ever formed: the estimate
#: costs O(n * max_lag) and its error bars O(n^2 * (max_lag + 256)) time in
#: O(n * (max_lag + 256)) memory -- about 130 MB and 2 s at this cap with the
#: default lag range -- but a timing list this long wants thinning anyway.
_MAX_EVENTS: int = 5000

_MEAN_ABS: float = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True, eq=False)
class EddingtonPlakidisResult:
    """Result of the Eddington-Plakidis fluctuation test.

    Attributes
    ----------
    alpha : float
        Mean absolute timing error of a single time of maximum (days). 0.0
        if the fitted timing variance came out negative. Weakly constrained
        when the wander dominates (its information lives only at zero lag) —
        ``epsilon`` is the primary output of this test.
    epsilon : float
        Mean absolute random period fluctuation per cycle (days). 0.0 if the
        fitted walk variance came out negative.
    alpha_err, epsilon_err : float
        1-sigma uncertainties of the estimator, from the exact covariance of
        the lag-bin means at the fitted variances (NaN where the parameter was
        clipped to 0). They scale with the estimate, so they are best read in
        log space: a walk measured at ``epsilon +- epsilon_err`` is uncertain
        by a factor, not by an additive amount.
    slope_significance : float
        The fitted walk variance in units of its standard deviation under the
        null of pure timing noise (at the observed residual scatter): a z-score,
        about N(0, 1) when there is no wander. Above ~3, random cycle-to-cycle
        fluctuation is present. 0.0 for an exactly periodic series (every O-C
        exactly zero), which has nothing to test.
    lags : numpy.ndarray
        Cycle separations with at least the minimum pair count.
    mean_u : numpy.ndarray
        Root-mean-square accumulated delay per lag (days), after detrending.
    n_pairs : numpy.ndarray
        Number of event pairs behind each lag bin.
    detrend : int
        Polynomial degree removed from the O-C before differencing.
    """

    alpha: float
    epsilon: float
    alpha_err: float
    epsilon_err: float
    slope_significance: float
    lags: FloatArray
    mean_u: FloatArray
    n_pairs: FloatArray
    detrend: int

    def __eq__(self, other: object) -> bool:
        """Value equality, arrays included (NaN equal to NaN).

        The generated ``__eq__`` compares the ndarray fields with ``==`` and
        raises "truth value of an array is ambiguous" on any two results.
        """
        if not isinstance(other, EddingtonPlakidisResult):
            return NotImplemented
        return all(
            np.array_equal(
                np.asarray(getattr(self, f.name)),
                np.asarray(getattr(other, f.name)),
                equal_nan=True,
            )
            for f in fields(self)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def alpha_sigma(self) -> float:
        """Gaussian-sigma equivalent of ``alpha`` (days)."""
        return self.alpha / _MEAN_ABS

    @property
    def epsilon_sigma(self) -> float:
        """Gaussian-sigma equivalent of ``epsilon`` (days)."""
        return self.epsilon / _MEAN_ABS

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalars plus the lag profile."""
        return {
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "alpha_err": self.alpha_err,
            "epsilon_err": self.epsilon_err,
            "alpha_sigma": self.alpha_sigma,
            "epsilon_sigma": self.epsilon_sigma,
            "slope_significance": self.slope_significance,
            "detrend": self.detrend,
            "lags": self.lags.tolist(),
            "mean_u": self.mean_u.tolist(),
            "n_pairs": self.n_pairs.tolist(),
        }


#: Rows per block of the trace products in :func:`_trace_products`.
_BLOCK: int = 256


def _walk_times(e_rel: FloatArray, x: FloatArray) -> FloatArray:
    """``M @ x`` for the walk covariance ``M_ij = min(E_i, E_j)``, ``E`` ascending.

    ``(M x)_i = sum_{j <= i} E_j x_j + E_i sum_{j > i} x_j``: two cumulative
    sums, O(n) per column instead of a dense n x n product.
    """
    e = e_rel.reshape((-1,) + (1,) * (x.ndim - 1))
    below = np.cumsum(e * x, axis=0)
    csum = np.cumsum(x, axis=0)
    return np.asarray(below + e * (csum[-1] - csum), dtype=np.float64)


def _pairs(cycles: FloatArray, top: int) -> tuple[IntArray, IntArray]:
    """Index pairs ``i < j`` (row-major) at most ``top`` cycles apart.

    Enumerated row by row from the sorted cycles, so only the pairs the lag
    profile uses are ever built -- not all ``n(n-1)/2`` of them.
    """
    n = cycles.size
    stop = np.searchsorted(cycles, cycles + top, side="right")
    count = stop - np.arange(n) - 1
    ii = np.repeat(np.arange(n), count)
    first = np.cumsum(count) - count
    jj = ii + 1 + (np.arange(ii.size) - np.repeat(first, count))
    return ii, jj


def _pair_expectations(
    u_mat: FloatArray, e_rel: FloatArray, ii: IntArray, jj: IntArray, lag: IntArray
) -> tuple[FloatArray, FloatArray]:
    """``E[u^2]`` of every pair per unit walk and per unit timing variance.

    The detrended covariance is ``s_eps^2 A M A + s_alpha^2 A`` with
    ``A = I - U U^T``. For the difference ``d = e_j - e_i`` and ``c = U_j - U_i``
    (rows of ``U``): ``d^T A d = 2 - |c|^2`` and
    ``d^T A M A d = lag - 2 c.(R_j - R_i) + c^T G c``, with ``R = M U`` and
    ``G = U^T M U`` -- the dense ``A M A`` (n x n, O(n^3) to form) reduced to
    n x (detrend + 1).
    """
    r_mat = _walk_times(e_rel, u_mat)
    g_mat = u_mat.T @ r_mat
    c = u_mat[jj] - u_mat[ii]
    g_noise = 2.0 - np.einsum("pk,pk->p", c, c)
    cross = np.einsum("pk,pk->p", c, r_mat[jj] - r_mat[ii])
    g_walk = lag - 2.0 * cross + np.einsum("pk,pk->p", c @ g_mat, c)
    return g_walk, g_noise


def _band(ii: IntArray, jj: IntArray, w: FloatArray, n: int, h: int) -> FloatArray:
    """Band storage ``band[i, h + j - i] = Q[i, j]`` of the pair Laplacian.

    ``Q = sum_p w_p (e_j - e_i)(e_j - e_i)^T``; every pair is at most ``h``
    indices apart, so ``Q`` is banded.
    """
    band = np.zeros((n, 2 * h + 1), dtype=np.float64)
    band[:, h] = np.bincount(ii, weights=w, minlength=n) + np.bincount(
        jj, weights=w, minlength=n
    )
    off = jj - ii
    band[ii, h + off] = -w
    band[jj, h - off] = -w
    return band


def _band_block(
    band: FloatArray, h: int, a: int, b: int, lo: int, hi: int
) -> FloatArray:
    """Rows ``a:b``, columns ``lo:hi`` of the banded matrix, dense."""
    out = np.zeros((b - a, hi - lo), dtype=np.float64)
    rows = np.arange(a, b)[:, None]
    cols = rows - h + np.arange(2 * h + 1)[None, :]
    ok = (cols >= lo) & (cols < hi)
    out[np.broadcast_to(rows - a, cols.shape)[ok], (cols - lo)[ok]] = band[a:b][ok]
    return out


def _trace_products(
    bands: list[FloatArray], h: int, apply_cov: Callable[[FloatArray], FloatArray]
) -> FloatArray:
    """``tr(Q_k C Q_k C)`` for each banded ``Q_k``; ``C`` symmetric, as an operator.

    Column block by column block: from the one block ``C[:, J']`` (``J``
    widened by the band) come ``(Q C)[J, :] = Q[J, J'] C[J', :]`` and
    ``(Q C)[:, J] = Q C[:, J]``, and the trace collects
    ``sum_{i, j in J} (Q C)[i, j] (Q C)[j, i]``. Memory is O(n (block + h)),
    the work O(n^2 (block + h)) in matrix products.
    """
    n = bands[0].shape[0]
    spans = [
        (a, min(n, a + _BLOCK), max(0, a - h), min(n, a + _BLOCK + h))
        for a in range(0, n, _BLOCK)
    ]
    blocks = [[_band_block(band, h, *span) for span in spans] for band in bands]
    out = np.zeros(len(bands), dtype=np.float64)
    for idx, (a, b, lo, hi) in enumerate(spans):
        c_wide = np.empty((n, hi - lo), dtype=np.float64)  # C[:, lo:hi]
        # Applied to a few unit columns at a time: the operator's temporaries
        # are each as large as its input.
        for s in range(lo, hi, _BLOCK // 4):
            e = min(hi, s + _BLOCK // 4)
            unit = np.zeros((n, e - s), dtype=np.float64)
            unit[np.arange(s, e), np.arange(e - s)] = 1.0
            c_wide[:, s - lo : e - lo] = apply_cov(unit)
        c_j = c_wide[:, a - lo : b - lo]  # C[:, a:b]
        for k, qb in enumerate(blocks):
            row_part = qb[idx] @ c_wide.T  # (Q C)[a:b, :]
            col_part = np.empty((n, b - a), dtype=np.float64)
            for idx2, (a2, b2, lo2, hi2) in enumerate(spans):
                col_part[a2:b2] = qb[idx2] @ c_j[lo2:hi2]  # (Q C)[:, a:b]
            out[k] += float(np.sum(col_part * row_part.T))
    return out


def _null_trace(band: FloatArray, h: int, u_mat: FloatArray) -> float:
    """``tr(Q A Q A)`` for ``A = I - U U^T``, from the band alone.

    ``tr(QQ) - 2 |QU|_F^2 + |U^T Q U|_F^2`` (``Q`` symmetric): O(n h) work, so
    the null spread costs nothing next to the estimate itself.
    """
    n = band.shape[0]
    pad = np.zeros((n + 2 * h, u_mat.shape[1]), dtype=np.float64)
    pad[h : h + n] = u_mat
    qu = np.zeros_like(u_mat)
    for t in range(2 * h + 1):
        qu += band[:, t, None] * pad[t : t + n]
    g = u_mat.T @ qu
    return float(np.sum(band * band) - 2.0 * np.sum(qu * qu) + np.sum(g * g))


def eddington_plakidis(
    times: Any,
    *,
    period: float,
    t0: float | None = None,
    max_lag: int | None = None,
    detrend: int = 2,
) -> EddingtonPlakidisResult:
    """Test a timing series for random cycle-to-cycle period fluctuations.

    Assigns cycle numbers against the trial ephemeris (like
    :func:`cuperiod.fit_ephemeris`), removes a polynomial trend of degree
    ``detrend``, bins the squared accumulated delays by cycle separation, and
    solves the timing variance and per-cycle walk variance from their exact
    detrending-corrected expectations (see the module docstring).

    Parameters
    ----------
    times : array-like
        Times of maximum/minimum light (days).
    period : float
        Trial period (days).
    t0 : float, optional
        Trial epoch; defaults to the first event.
    max_lag : int, optional
        Largest cycle separation used; default is a third of the span in
        cycles, capped at 100.
    detrend : int, default 2
        Polynomial degree removed from the O-C first. 2 removes a secular
        quadratic (a real dP/dt) so it cannot masquerade as wander; use 1 to
        test against a linear ephemeris only.

    Returns
    -------
    EddingtonPlakidisResult

    Examples
    --------
    >>> ep = eddington_plakidis(maxima, period=331.0)      # doctest: +SKIP
    >>> ep.epsilon, ep.slope_significance                  # doctest: +SKIP

    Notes
    -----
    The classical estimator with an exact-expectation correction; for
    likelihood-based model selection between jitter and period-change models
    see Koen (2006, MNRAS 365, 489). Cycle numbering shares
    :func:`cuperiod.fit_ephemeris`'s nearest-integer caveat.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    t = np.sort(t[np.isfinite(t)])
    # `not (finite and > 0)`: NaN passes a plain `<= 0` test and only failed
    # much later, converting a NaN cycle count to an integer.
    if not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"period must be positive and finite, got {period}")
    if t0 is not None and not math.isfinite(t0):
        raise ValueError(f"t0 must be finite, got {t0}")
    if not 0 <= detrend <= 4:
        raise ValueError(f"detrend must be between 0 and 4, got {detrend}")
    if t.size < max(8, detrend + 4):
        raise InsufficientDataError(
            f"eddington_plakidis: need at least {max(8, detrend + 4)} event "
            f"times, got {t.size}"
        )
    epoch = float(t[0]) if t0 is None else float(t0)
    cycles = np.round((t - epoch) / period)
    oc = t - epoch - period * cycles
    # Duplicate cycle assignments (two timings of the same event) are averaged.
    uniq, inverse = np.unique(cycles, return_inverse=True)
    if uniq.size < t.size:
        dup_sums = np.zeros(uniq.size)
        dup_counts = np.zeros(uniq.size)
        np.add.at(dup_sums, inverse, oc)
        np.add.at(dup_counts, inverse, 1.0)
        oc = dup_sums / dup_counts
        cycles = uniq
    n = cycles.size
    if n > _MAX_EVENTS:
        raise InsufficientDataError(
            f"eddington_plakidis: {n} distinct cycles exceeds the exact-"
            f"estimator cap of {_MAX_EVENTS}; thin the timing list"
        )

    span = int(cycles[-1] - cycles[0])
    top = max(3, min(span // 3, 100))
    if max_lag is not None:
        top = min(int(max_lag), span)
    if top < 3:
        raise InsufficientDataError(
            "eddington_plakidis: too few distinct cycles for a lag profile"
        )

    # Detrending: an orthonormal basis U of the polynomial, A = I - U U^T,
    # and the detrended delays. Nothing below forms an n x n matrix.
    basis = np.vander(cycles - cycles.mean(), detrend + 1, increasing=True)
    u_mat = np.asarray(np.linalg.qr(basis)[0], dtype=np.float64)
    resid = oc - u_mat @ (u_mat.T @ oc)
    e_rel = cycles - cycles[0]

    ii, jj = _pairs(cycles, top)
    if ii.size == 0:
        raise InsufficientDataError("eddington_plakidis: no usable pairs")
    lag_flat = (cycles[jj] - cycles[ii]).astype(np.int64)
    d = resid[jj] - resid[ii]
    g_walk, g_noise = _pair_expectations(u_mat, e_rel, ii, jj, lag_flat)

    sums_u2 = np.bincount(lag_flat, weights=d * d, minlength=top + 1)
    sums_gw = np.bincount(lag_flat, weights=g_walk, minlength=top + 1)
    sums_gn = np.bincount(lag_flat, weights=g_noise, minlength=top + 1)
    counts = np.bincount(lag_flat, minlength=top + 1)
    del d, g_walk, g_noise
    good = np.flatnonzero(counts >= _MIN_PAIRS)
    good = good[good >= 1]
    if good.size < 3:
        raise InsufficientDataError(
            "eddington_plakidis: fewer than 3 lag bins with "
            f"{_MIN_PAIRS}+ pairs; supply more event times or raise max_lag"
        )
    m = counts[good].astype(np.float64)
    y = sums_u2[good] / m
    x_mat = np.stack([sums_gn[good] / m, sums_gw[good] / m], axis=1)

    def result(**values: float) -> EddingtonPlakidisResult:
        return EddingtonPlakidisResult(
            **values,
            lags=good.astype(np.float64),
            mean_u=np.sqrt(y),
            n_pairs=m,
            detrend=detrend,
        )

    if not np.any(y > 0.0):
        # Every detrended delay is exactly zero (an exact ephemeris, or timings
        # rounded far coarser than their scatter): no wander and no noise to
        # measure, and the reweighting below would divide by zero.
        return result(
            alpha=0.0,
            epsilon=0.0,
            alpha_err=float("nan"),
            epsilon_err=float("nan"),
            slope_significance=0.0,
        )

    # Weighted LSQ for (sigma_alpha^2, sigma_eps^2). A bin mean of squared
    # Gaussian differences has variance ~ 2*E[u^2]^2/m, so after a count-
    # weighted first pass the fit is reweighted once by the model-predicted
    # variance — without this, the large-lag bins (huge u^2, huge scatter)
    # drown the timing-noise term. ``lin`` is the linear map the final pass
    # applies to the bin means: coef = lin^T y.
    weights = m.copy()
    coef = np.zeros(2)
    lin = np.zeros((good.size, 2))
    for _ in range(2):
        xtw = x_mat.T * weights[None, :]
        lin = np.asarray(np.linalg.solve(xtw @ x_mat, xtw), dtype=np.float64).T
        coef = lin.T @ y
        floor = 0.05 * float(np.median(y))
        predicted = np.maximum(x_mat @ coef, max(floor, np.finfo(float).tiny))
        weights = m / (2.0 * predicted * predicted)
    var_alpha, var_eps = float(coef[0]), float(coef[1])

    # The two variances as quadratic forms r^T Q_k r in the residuals: Q_k is
    # the pair Laplacian with weight lin[x, k] / m_x on every pair of lag x.
    weight_of_lag = np.zeros((2, top + 1))
    weight_of_lag[:, good] = (lin / m[:, None]).T
    pair_w = weight_of_lag[:, lag_flat]
    used = np.flatnonzero(counts[lag_flat] >= _MIN_PAIRS)
    ii, jj, pair_w = ii[used], jj[used], pair_w[:, used]
    half_band = int(np.max(jj - ii))
    bands = [_band(ii, jj, pair_w[k], n, half_band) for k in range(2)]
    del ii, jj, pair_w, lag_flat

    # Errors: the covariance at the fitted variances.
    if var_eps > 0.0 or var_alpha > 0.0:
        s_eps, s_alpha = max(var_eps, 0.0), max(var_alpha, 0.0)

        def apply_cov(x: FloatArray) -> FloatArray:
            ax = x - u_mat @ (u_mat.T @ x)
            sx = s_eps * _walk_times(e_rel, ax) + s_alpha * ax
            return np.asarray(sx - u_mat @ (u_mat.T @ sx), dtype=np.float64)

        var_cov = 2.0 * _trace_products(bands, half_band, apply_cov)
        var_alpha_err = float(np.sqrt(max(var_cov[0], 0.0)))
        var_eps_err = float(np.sqrt(max(var_cov[1], 0.0)))
    else:
        var_alpha_err = var_eps_err = float("nan")

    # Significance: var_eps against its spread under pure timing noise at the
    # observed residual scatter (C = s^2 A), which has a closed form in Q_k.
    s2 = float(np.dot(resid, resid)) / max(1, n - (detrend + 1))
    null_sd = s2 * math.sqrt(max(2.0 * _null_trace(bands[1], half_band, u_mat), 0.0))
    significance = var_eps / null_sd if null_sd > 0.0 else float("nan")

    sigma_alpha = float(np.sqrt(var_alpha)) if var_alpha > 0.0 else 0.0
    sigma_eps = float(np.sqrt(var_eps)) if var_eps > 0.0 else 0.0
    # d(sigma)/d(var) = 1/(2*sigma); the mean-abs factor carries through.
    alpha_err = (
        _MEAN_ABS * var_alpha_err / (2.0 * sigma_alpha)
        if sigma_alpha > 0.0
        else float("nan")
    )
    epsilon_err = (
        _MEAN_ABS * var_eps_err / (2.0 * sigma_eps)
        if sigma_eps > 0.0
        else float("nan")
    )
    return result(
        alpha=sigma_alpha * _MEAN_ABS,
        epsilon=sigma_eps * _MEAN_ABS,
        alpha_err=alpha_err,
        epsilon_err=epsilon_err,
        slope_significance=float(significance),
    )


__all__ = ["EddingtonPlakidisResult", "eddington_plakidis"]
