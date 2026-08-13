"""SuperSmoother periodogram (Friedman 1984; Reimann 1994).

For each trial period the data are phase-folded and fit with Friedman's variable-span
smoother: three primary local-linear smooths (span fractions 0.05 / 0.2 / 0.5 of the
points), leave-one-out cross-validation to pick the best span at every phase point,
and a final smoothing pass over the blended curve. The periodogram statistic follows
gatspy's ``SuperSmoother``::

    score = 1 - mean_i |y_i - model_i| / dy_i  /  mean_i |y_i - mu| / dy_i

with ``mu`` the inverse-variance weighted mean — the fractional reduction in mean
absolute (error-standardized) deviation, *maximized* at the true period. Being fully
non-parametric, it fits any repeating shape without a harmonic budget; the price is
cost (a sort plus several smooths per trial period) and a soft spectrum. Values can
dip slightly below 0 when a fold fits worse than the constant model. Note that a
fold at an integer *multiple* of the true period is still a coherent repeating
curve, so ``2P``, ``3P``, ... score nearly as high as ``P`` itself: read the
shortest period of a high-score family as the candidate (or bound the search from
above), and let :func:`cuperiod.alias_diagnostics` arbitrate the family.

Semantics follow ``supersmoother`` (VanderPlas), the reference implementation used by
gatspy, with three deliberate choices:

* span windows are forced to odd point counts (``max(3, int(span*N))``, +1 if even),
  matching the upstream fix for Friedman's odd-span assumption;
* folding is always periodic — windows wrap around phase 0/1 via period-shifted
  padding, so the edge-window pathologies of the plain smoother cannot occur;
* degenerate windows (duplicate phases) fall back to the weighted mean instead of
  raising, and the bass-enhancement factor is clamped to ``[0, 1]`` (the reference
  can emit NaN for ``alpha`` between 9 and 10).

All backends share one vectorized array-API kernel — ``numpy`` on the CPU, ``cupy``
on NVIDIA, ``torch`` on any torch device — plus a numba-parallel CPU tier; every
window sum comes from prefix sums, so the cost per trial period is ``O(N log N)``
for the sort plus ``O(N)`` per span.

References: Friedman 1984, "A Variable Span Smoother" (LCS Tech. Rep. 5 /
SLAC-PUB-3477); Reimann 1994 (PhD thesis, UC Berkeley); VanderPlas & Ivezić 2015,
ApJ 812, 18 (gatspy).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import SuperSmootherSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

#: Trial periods per vectorized batch on host backends when ``batch`` is auto (0).
DEFAULT_BATCH: Final = 1024

#: Transient byte budgets per trial-period chunk; the chunk adapts to the
#: light-curve length, so long curves cannot blow memory. Device backends get a
#: far larger budget — their single-shot latency is dominated by per-chunk
#: dispatch overhead, so fewer, larger chunks are strictly faster at identical
#: results.
_CHUNK_BYTES: Final = 1 << 27
_DEVICE_CHUNK_BYTES: Final = 1 << 29

#: Rough number of (chunk, N)-sized float64 workspaces alive at once.
_WORKSPACES: Final = 28


def _resolve_chunk(batch: int, n: int, on_device: bool) -> int:
    """Effective trial-period chunk length: explicit ``batch`` (byte-capped), or auto.

    An explicit ``batch > 0`` is honored up to the byte budget; ``batch <= 0``
    fills the device budget outright, or keeps the classic ``DEFAULT_BATCH`` on
    host backends. The chunk length does not affect the result.
    """
    per_row = max(1, n * 8 * _WORKSPACES)
    cap = max(1, (_DEVICE_CHUNK_BYTES if on_device else _CHUNK_BYTES) // per_row)
    if batch > 0:
        return min(batch, cap)
    return cap if on_device else min(DEFAULT_BATCH, cap)


def span_windows(spans: tuple[float, ...], n: int) -> tuple[int, ...]:
    """Span fractions to odd window point counts: ``max(3, int(s*n))``, +1 if even."""
    out = []
    for s in spans:
        j = max(3, int(s * n))
        out.append(j if j % 2 else j + 1)
    return tuple(out)


def _baseline_error(y: FloatArray, w: FloatArray, inv_dy: FloatArray) -> float:
    """Mean absolute standardized deviation about the weighted mean (gatspy)."""
    mu = float(np.dot(w, y))
    return float(np.mean(np.abs(y - mu) * inv_dy))


# --- vectorized array-API kernel ---------------------------------------------


def _cumsum0(xp: ModuleType, a: Any) -> Any:
    """Prefix sums along the last axis with a leading zero column."""
    c = xp.cumulative_sum(a, axis=-1)
    return xp.concat([xp.zeros_like(c[..., :1]), c], axis=-1)


def _take_rows(xp: ModuleType, a: Any, idx: Any) -> Any:
    """Per-row gather of ``a`` (same shape as ``idx``) along the last axis."""
    if hasattr(xp, "take_along_axis"):
        return xp.take_along_axis(a, idx, axis=-1)
    import torch  # pragma: no cover - older array-api-compat without the wrapper

    return torch.take_along_dim(a, idx, dim=-1)  # pragma: no cover


def _pad_phase(xp: ModuleType, x: Any, h: int) -> Any:
    """Circular phase padding: one wrapped copy each side, shifted by ∓1."""
    return xp.concat([x[:, -h:] - 1.0, x, x[:, :h] + 1.0], axis=-1)


def _pad_vals(xp: ModuleType, a: Any, h: int) -> Any:
    """Circular value padding matching :func:`_pad_phase` (no shift)."""
    return xp.concat([a[:, -h:], a, a[:, :h]], axis=-1)


def _window_sums(c: Any, m: int, h: int, n: int) -> Any:
    """Window sums (halfwidth ``m``) at every center from a padded prefix array."""
    return c[:, h + m + 1 : h + m + 1 + n] - c[:, h - m : h - m + n]


def _line_values(
    xp: ModuleType,
    shared: tuple[Any, Any, Any],
    c_wa: Any,
    c_wxa: Any,
    a: Any,
    xs: Any,
    ws: Any,
    m: int,
    h: int,
    n: int,
    cv: bool,
) -> Any:
    """Windowed weighted linear fit evaluated at each center point.

    ``shared`` holds the prefix sums of ``w``, ``w*x``, ``w*x^2`` over the padded
    row (span-independent); ``c_wa``/``c_wxa`` those of ``w*a`` and ``w*x*a``. With
    ``cv=True`` each point's own contribution is subtracted from the five sums
    before the fit — the exact leave-one-out value of the reference. Windows whose
    abscissa spread cancels (duplicate phases) fall back to the window's weighted
    mean instead of raising like the reference does.
    """
    c_w, c_wx, c_wxx = shared
    w_sum = _window_sums(c_w, m, h, n)
    tw = _window_sums(c_wx, m, h, n)
    ttw = _window_sums(c_wxx, m, h, n)
    yw = _window_sums(c_wa, m, h, n)
    tyw = _window_sums(c_wxa, m, h, n)
    if cv:
        w_sum = w_sum - ws
        tw = tw - ws * xs
        ttw = ttw - ws * xs * xs
        yw = yw - ws * a
        tyw = tyw - ws * xs * a
    denom = w_sum * ttw - tw * tw
    slope = tyw * w_sum - tw * yw
    icept = ttw * yw - tyw * tw
    eps = float(xp.finfo(xs.dtype).eps)
    good = denom > (32.0 * eps) * w_sum * ttw
    denom_safe = xp.where(good, denom, xp.ones_like(denom))
    w_safe = xp.where(w_sum > 0.0, w_sum, xp.ones_like(w_sum))
    return xp.where(good, (slope * xs + icept) / denom_safe, yw / w_safe)


def _chunk_scores(
    xp: ModuleType,
    xs: Any,
    ys: Any,
    ws: Any,
    ids: Any,
    span_fracs: tuple[float, ...],
    windows: tuple[int, ...],
    j_mid: int,
    j_fin: int,
    alpha: float | None,
    baseline: float,
) -> Any:
    """SuperSmoother scores for one chunk of already-folded, sorted rows.

    ``xs``/``ys``/``ws``/``ids`` are ``(chunk, N)``: normalized phase (sorted
    ascending per row), band-mean-centered values, normalized inverse-variance
    weights, and ``1/dy``, all gathered to the sorted order.
    """
    n = int(xs.shape[-1])
    h = max(*windows, j_mid, j_fin) // 2
    x_p = _pad_phase(xp, xs, h)
    w_p = _pad_vals(xp, ws, h)
    shared = (
        _cumsum0(xp, w_p),
        _cumsum0(xp, w_p * x_p),
        _cumsum0(xp, w_p * x_p * x_p),
    )

    def smooth(a: Any, c_wa: Any, c_wxa: Any, j: int, cv: bool) -> Any:
        return _line_values(
            xp, shared, c_wa, c_wxa, a, xs, ws, j // 2, h, n, cv
        )

    def prefixes(a: Any) -> tuple[Any, Any]:
        wa = w_p * _pad_vals(xp, a, h)
        return _cumsum0(xp, wa), _cumsum0(xp, wa * x_p)

    c_wy, c_wxy = prefixes(ys)
    curves = [smooth(ys, c_wy, c_wxy, j, cv=True) for j in windows]

    if len(curves) == 1:
        raw = curves[0]
    else:
        sm_res = []
        for curve in curves:
            resid = xp.abs(curve - ys) * ids
            c_wr, c_wxr = prefixes(resid)
            sm_res.append(smooth(resid, c_wr, c_wxr, j_mid, cv=False))
        stack = xp.stack(sm_res)
        kmin = xp.argmin(stack, axis=0)
        best = xp.zeros_like(ys) + span_fracs[0]
        for k in range(1, len(span_fracs)):
            best = xp.where(kmin == k, span_fracs[k], best)
        if alpha is not None:
            # Friedman's bass enhancement, with the reference's NaN hazard
            # closed: the smoothed residuals can undershoot 0, so the ratio is
            # clamped before the fractional power and the factor to [0, 1].
            eps = float(xp.finfo(xs.dtype).eps)
            minres = xp.min(stack, axis=0)
            last = sm_res[-1]
            base = xp.clip(minres, 0.0, None) / xp.where(
                last > eps, last, xp.full_like(last, eps)
            )
            factor = xp.clip(base ** (10.0 - alpha), 0.0, 1.0)
            best = best + factor * (span_fracs[-1] - best)
        c_wb, c_wxb = prefixes(best)
        sm_spans = smooth(best, c_wb, c_wxb, j_mid, cv=False)
        q = xp.clip(sm_spans, span_fracs[0], span_fracs[-1])
        # Piecewise-linear blend of the primary curves at the smoothed span
        # (the reference's ``multinterp``): every interval's line is anchored
        # at its lower node, and the where-chain assigns q == node upward.
        raw = curves[-2] + (q - span_fracs[-2]) * (
            curves[-1] - curves[-2]
        ) / (span_fracs[-1] - span_fracs[-2])
        for k in range(len(span_fracs) - 2, 0, -1):
            lin = curves[k - 1] + (q - span_fracs[k - 1]) * (
                curves[k] - curves[k - 1]
            ) / (span_fracs[k] - span_fracs[k - 1])
            raw = xp.where(q < span_fracs[k], lin, raw)

    c_wm, c_wxm = prefixes(raw)
    model = smooth(raw, c_wm, c_wxm, j_fin, cv=False)
    err = xp.sum(xp.abs(ys - model) * ids, axis=-1) / float(n)
    return 1.0 - err / baseline


def _score_batches(
    xp: ModuleType,
    tau: Any,
    y: Any,
    w: Any,
    inv_dy: Any,
    freqs: FloatArray,
    *,
    span_fracs: tuple[float, ...],
    windows: tuple[int, ...],
    j_mid: int,
    j_fin: int,
    alpha: float | None,
    baseline: float,
    chunk: int,
) -> FloatArray:
    """Fold, sort, and score every trial frequency in ``chunk``-sized pieces.

    The scores accumulate on the compute device and cross to the host **once**
    at the end — a per-chunk transfer would synchronize the stream every
    iteration, which is what used to dominate single-shot GPU latency.
    """
    nf = int(freqs.shape[0])
    fdtype = tau.dtype
    dev = device_ref(tau)
    freqs_dev = xp.asarray(freqs, dtype=fdtype, device=dev)
    out = xp.empty(nf, dtype=fdtype, device=dev)
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        fc = freqs_dev[start:stop]
        x = xp.remainder(tau[None, :] * fc[:, None], 1.0)
        order = xp.argsort(x, axis=-1)
        xs = _take_rows(xp, x, order)
        out[start:stop] = _chunk_scores(
            xp, xs, y[order], w[order], inv_dy[order],
            span_fracs, windows, j_mid, j_fin, alpha, baseline,
        )
    return to_host(out)


# --- numba fast CPU tier ------------------------------------------------------

_NUMBA_SS_KERNEL: Any = None


def _numba_ss_kernel() -> Any:
    """Lazily compile (once) and cache the numba SuperSmoother kernel.

    The same chain as the vectorized path — sort, circular pad, prefix sums,
    constant-span linear fits with exact leave-one-out, span selection,
    interpolation, final smooth — one trial frequency per ``prange`` iteration.
    """
    global _NUMBA_SS_KERNEL
    if _NUMBA_SS_KERNEL is not None:
        return _NUMBA_SS_KERNEL
    from numba import njit, prange

    @njit(cache=True)  # pragma: no cover - njit
    def _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, a, m, h, n, cv, out):  # type: ignore[no-untyped-def]
        eps = 2.220446049250313e-16
        for i in range(n):
            lo = h + i - m
            hi = h + i + m + 1
            w_sum = cw[hi] - cw[lo]
            tw = cwx[hi] - cwx[lo]
            ttw = cwxx[hi] - cwxx[lo]
            yw = cwa[hi] - cwa[lo]
            tyw = cwxa[hi] - cwxa[lo]
            if cv:
                w_sum -= ws[i]
                tw -= ws[i] * xs[i]
                ttw -= ws[i] * xs[i] * xs[i]
                yw -= ws[i] * a[i]
                tyw -= ws[i] * xs[i] * a[i]
            denom = w_sum * ttw - tw * tw
            if denom > 32.0 * eps * w_sum * ttw:
                slope = tyw * w_sum - tw * yw
                icept = ttw * yw - tyw * tw
                out[i] = (slope * xs[i] + icept) / denom
            elif w_sum > 0.0:
                out[i] = yw / w_sum
            else:
                out[i] = 0.0

    @njit(cache=True)  # pragma: no cover - njit
    def _prefix(vals, xpad, wpad, m_pad, cwa, cwxa):  # type: ignore[no-untyped-def]
        acc_a = 0.0
        acc_xa = 0.0
        cwa[0] = 0.0
        cwxa[0] = 0.0
        for j in range(m_pad):
            wa = wpad[j] * vals[j]
            acc_a += wa
            acc_xa += wa * xpad[j]
            cwa[j + 1] = acc_a
            cwxa[j + 1] = acc_xa

    @njit(cache=True)  # pragma: no cover - njit
    def _pad_circular(a, h, n, out):  # type: ignore[no-untyped-def]
        for j in range(h):
            out[j] = a[n - h + j]
        for j in range(n):
            out[h + j] = a[j]
        for j in range(h):
            out[h + n + j] = a[j]

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(  # type: ignore[no-untyped-def]
        tau, y, w, inv_dy, freqs, windows, span_fracs, j_mid, j_fin,
        alpha, use_alpha, baseline,
    ):
        nf = freqs.shape[0]
        n = tau.shape[0]
        n_spans = windows.shape[0]
        h = j_mid // 2
        if j_fin // 2 > h:
            h = j_fin // 2
        for k in range(n_spans):
            if windows[k] // 2 > h:
                h = windows[k] // 2
        m_pad = n + 2 * h
        eps = 2.220446049250313e-16
        out = np.empty(nf)
        for p in prange(nf):
            f = freqs[p]
            x = np.empty(n)
            for j in range(n):
                q = tau[j] * f
                x[j] = q - np.floor(q)
            order = np.argsort(x)
            xs = np.empty(n)
            ys = np.empty(n)
            ws = np.empty(n)
            ids = np.empty(n)
            for j in range(n):
                o = order[j]
                xs[j] = x[o]
                ys[j] = y[o]
                ws[j] = w[o]
                ids[j] = inv_dy[o]
            xpad = np.empty(m_pad)
            for j in range(h):
                xpad[j] = xs[n - h + j] - 1.0
                xpad[h + n + j] = xs[j] + 1.0
            for j in range(n):
                xpad[h + j] = xs[j]
            wpad = np.empty(m_pad)
            _pad_circular(ws, h, n, wpad)

            cw = np.empty(m_pad + 1)
            cwx = np.empty(m_pad + 1)
            cwxx = np.empty(m_pad + 1)
            acc_w = 0.0
            acc_x = 0.0
            acc_xx = 0.0
            cw[0] = 0.0
            cwx[0] = 0.0
            cwxx[0] = 0.0
            for j in range(m_pad):
                wj = wpad[j]
                xj = xpad[j]
                acc_w += wj
                acc_x += wj * xj
                acc_xx += wj * xj * xj
                cw[j + 1] = acc_w
                cwx[j + 1] = acc_x
                cwxx[j + 1] = acc_xx

            apad = np.empty(m_pad)
            cwa = np.empty(m_pad + 1)
            cwxa = np.empty(m_pad + 1)
            _pad_circular(ys, h, n, apad)
            _prefix(apad, xpad, wpad, m_pad, cwa, cwxa)
            curves = np.empty((n_spans, n))
            for k in range(n_spans):
                _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, ys,
                      windows[k] // 2, h, n, True, curves[k])

            if n_spans == 1:
                raw = curves[0].copy()
            else:
                sm_res = np.empty((n_spans, n))
                resid = np.empty(n)
                for k in range(n_spans):
                    for j in range(n):
                        d = curves[k, j] - ys[j]
                        resid[j] = (d if d >= 0.0 else -d) * ids[j]
                    _pad_circular(resid, h, n, apad)
                    _prefix(apad, xpad, wpad, m_pad, cwa, cwxa)
                    _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, resid,
                          j_mid // 2, h, n, False, sm_res[k])
                best = np.empty(n)
                for j in range(n):
                    kbest = 0
                    vbest = sm_res[0, j]
                    for k in range(1, n_spans):
                        if sm_res[k, j] < vbest:
                            vbest = sm_res[k, j]
                            kbest = k
                    best[j] = span_fracs[kbest]
                if use_alpha:
                    for j in range(n):
                        minres = sm_res[0, j]
                        for k in range(1, n_spans):
                            if sm_res[k, j] < minres:
                                minres = sm_res[k, j]
                        if minres < 0.0:
                            minres = 0.0
                        last = sm_res[n_spans - 1, j]
                        if last < eps:
                            last = eps
                        factor = (minres / last) ** (10.0 - alpha)
                        if factor > 1.0:
                            factor = 1.0
                        best[j] = best[j] + factor * (
                            span_fracs[n_spans - 1] - best[j]
                        )
                _pad_circular(best, h, n, apad)
                _prefix(apad, xpad, wpad, m_pad, cwa, cwxa)
                sm_spans = np.empty(n)
                _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, best,
                      j_mid // 2, h, n, False, sm_spans)
                raw = np.empty(n)
                for j in range(n):
                    q = sm_spans[j]
                    if q < span_fracs[0]:
                        q = span_fracs[0]
                    elif q > span_fracs[n_spans - 1]:
                        q = span_fracs[n_spans - 1]
                    k = n_spans - 1
                    for kk in range(1, n_spans):
                        if q < span_fracs[kk]:
                            k = kk
                            break
                    raw[j] = curves[k - 1, j] + (q - span_fracs[k - 1]) * (
                        curves[k, j] - curves[k - 1, j]
                    ) / (span_fracs[k] - span_fracs[k - 1])

            _pad_circular(raw, h, n, apad)
            _prefix(apad, xpad, wpad, m_pad, cwa, cwxa)
            model = np.empty(n)
            _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, raw,
                  j_fin // 2, h, n, False, model)
            acc = 0.0
            for j in range(n):
                d = ys[j] - model[j]
                acc += (d if d >= 0.0 else -d) * ids[j]
            out[p] = 1.0 - (acc / n) / baseline
        return out

    _NUMBA_SS_KERNEL = _kernel
    return _kernel


# --- public dispatch ----------------------------------------------------------


def supersmoother_score(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray | None,
    periods: FloatArray,
    *,
    primary_spans: tuple[float, ...] = (0.05, 0.2, 0.5),
    middle_span: float = 0.2,
    final_span: float = 0.05,
    bass_enhancement: float | None = None,
    backend: str = "numpy",
    batch: int = 0,
    precision: str = "auto",
) -> FloatArray:
    """SuperSmoother periodogram score for each trial period.

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    dy : numpy.ndarray or None
        1-sigma errors, or ``None`` for uniform weights.
    periods : numpy.ndarray
        Trial periods (days).
    primary_spans : tuple of float, default (0.05, 0.2, 0.5)
        Candidate span fractions, strictly increasing, each in (0, 1].
    middle_span, final_span : float
        Spans of the residual/span smoothing and of the final pass.
    bass_enhancement : float, optional
        Friedman's ``alpha`` in [0, 10]; ``None`` disables it.
    backend : str, default "numpy"
        ``"numpy"``, ``"numba"``, ``"cupy"``, or ``"torch"`` / ``"torch:<device>"``.
    batch : int, default 0
        Trial periods per vectorized chunk (capped by a byte budget); 0
        auto-sizes the chunk — much larger on device backends, whose
        single-shot latency is per-chunk dispatch overhead, not arithmetic.
        The chunk does not affect the result.
    precision : str, default "auto"
        Compute dtype for the cupy/torch paths (numpy/numba always run float64).

    Returns
    -------
    numpy.ndarray
        ``score`` per period, maximized at the true period; 1 is a perfect fit
        and 0 means no improvement over a constant. All zeros for a constant
        signal.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = t.size
    if dy is None:
        inv_dy = np.ones(n, dtype=np.float64)
    else:
        inv_dy = 1.0 / np.ascontiguousarray(dy, dtype=np.float64)
    w = inv_dy * inv_dy
    w = w / w.sum()
    mu = float(np.dot(w, y))
    y0 = y - mu
    baseline = float(np.mean(np.abs(y0) * inv_dy))
    if not np.isfinite(baseline) or baseline <= 0.0:
        return np.zeros(periods_host.size, dtype=np.float64)
    tau = t - t.min()
    freqs = 1.0 / periods_host

    spans = tuple(float(s) for s in primary_spans)
    windows = span_windows(spans, n)
    j_mid = span_windows((middle_span,), n)[0]
    j_fin = span_windows((final_span,), n)[0]
    alpha = None if bass_enhancement is None else float(bass_enhancement)

    if backend == "numba":
        kernel = _numba_ss_kernel()
        return np.asarray(
            kernel(
                tau, y0, w, inv_dy, freqs,
                np.asarray(windows, dtype=np.int64),
                np.asarray(spans, dtype=np.float64),
                j_mid, j_fin,
                0.0 if alpha is None else alpha, alpha is not None, baseline,
            ),
            dtype=np.float64,
        )
    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        rdtype = (
            np.float32
            if resolve_precision(precision, "cuda") == "float32"
            else np.float64
        )
        xp = array_namespace(cp.asarray(tau))
        return _score_batches(
            xp,
            cp.asarray(tau.astype(rdtype)), cp.asarray(y0.astype(rdtype)),
            cp.asarray(w.astype(rdtype)), cp.asarray(inv_dy.astype(rdtype)),
            freqs,
            span_fracs=spans, windows=windows, j_mid=j_mid, j_fin=j_fin,
            alpha=alpha, baseline=baseline,
            chunk=_resolve_chunk(batch, n, on_device=True),
        )
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        tdtype = (
            torch.float32
            if resolve_precision(precision, device) == "float32"
            else torch.float64
        )
        tau_d = to_device_array(tau, device=device, dtype=tdtype)
        return _score_batches(
            array_namespace(tau_d),
            tau_d,
            to_device_array(y0, device=device, dtype=tdtype),
            to_device_array(w, device=device, dtype=tdtype),
            to_device_array(inv_dy, device=device, dtype=tdtype),
            freqs,
            span_fracs=spans, windows=windows, j_mid=j_mid, j_fin=j_fin,
            alpha=alpha, baseline=baseline,
            chunk=_resolve_chunk(batch, n, on_device=device != "cpu"),
        )
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    return _score_batches(
        array_namespace(periods_host),
        tau, y0, w, inv_dy, freqs,
        span_fracs=spans, windows=windows, j_mid=j_mid, j_fin=j_fin,
        alpha=alpha, baseline=baseline,
        chunk=_resolve_chunk(batch, n, on_device=False),
    )


# --- method wrapper -----------------------------------------------------------


class SuperSmootherMethod(PeriodogramMethod):
    """SuperSmoother method (numba/numpy CPU, cupy GPU, torch portable)."""

    name: ClassVar[str] = "SUPERSMOOTHER"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = True
    settings_cls: ClassVar[type] = SuperSmootherSettings
    cpu_backend: ClassVar[str] = "numpy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numba", "numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: SuperSmootherSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("SUPERSMOOTHER: no usable time baseline")
        minimum = settings.minimum_frequency or 1.0 / finite.baseline
        maximum = settings.maximum_frequency or pseudo_nyquist_frequency(
            finite.time, settings.nyquist_factor
        )
        return uniform_frequency_grid(
            finite.baseline,
            maximum_frequency=maximum,
            minimum_frequency=minimum,
            samples_per_peak=settings.samples_per_peak,
        )

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: SuperSmootherSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"SUPERSMOOTHER: {n} finite points < min_detections "
                f"{settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("SUPERSMOOTHER: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        score = supersmoother_score(
            finite.time,
            finite.value,
            finite.error,
            periods,
            primary_spans=settings.primary_spans,
            middle_span=settings.middle_span,
            final_span=settings.final_span,
            bass_enhancement=settings.bass_enhancement,
            backend=backend,
            batch=settings.batch_periods,
            precision=settings.precision,
        )
        return Periodogram.from_spectrum(
            method="SUPERSMOOTHER",
            backend=backend,
            frequency=1.0 / periods,
            power=score,
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: SuperSmootherSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        from cuperiod.multiband.supersmoother_mb import supersmoother_multiband

        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        return supersmoother_multiband(grid, mblc, settings, backend)

    def estimate_device_bytes(self, n_points: int) -> int:
        return _DEVICE_CHUNK_BYTES + 64 * 1024**2 + n_points * 8 * 8


register(SuperSmootherMethod())

__all__ = [
    "DEFAULT_BATCH",
    "SuperSmootherMethod",
    "span_windows",
    "supersmoother_score",
]
