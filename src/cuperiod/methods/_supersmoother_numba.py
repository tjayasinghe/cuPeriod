"""numba kernels of the SuperSmoother periodogram (the fast CPU tier).

Imported lazily by :func:`cuperiod.methods.supersmoother._numba_ss_kernel` only, so
numba stays an optional dependency: never import this module eagerly.

The kernels live at module level on purpose. numba's on-disk cache keys a function on
its pickled closure, and a closure over other jitted dispatchers pickles a per-process
id, so the same kernel nested in a factory (as it used to be) missed the cache in every
new process -- 5-9 s of JIT per CLI run, GUI session and batch worker, plus one more
cache file each time. Module-level helpers are resolved as globals instead, which keeps
the cache key stable. Any helper added here must stay module-level too.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

#: A phase bucket holding more points than this sends :func:`_stable_phase_order` to
#: a merge sort: the insertion sort inside a bucket is quadratic in its occupancy, and
#: crowded buckets mean clustered (or tied) phases, where the bucket sort gains nothing.
_MAX_BUCKET = 16


@njit(cache=True)  # pragma: no cover - njit
def _stable_phase_order(x):  # type: ignore[no-untyped-def]
    """Stable argsort of fold phases ``x`` (in ``[0, 1]``) in expected ``O(N)``.

    A counting pass scatters the indices into ``N`` equal-width phase buckets in input
    order, and an insertion sort with a strict comparison orders each bucket, so equal
    phases keep their input order: the result is the unique stable order, identical to
    the array-API backends' stable ``argsort``. A bucket's index ``int(x * N)`` is
    monotonic in ``x``, so the insertion sort never crosses a bucket boundary. Falls
    back to a merge sort (also stable) when a phase lies outside ``[0, 1]`` -- a zero
    or NaN period folds to NaN, which must not become a bucket index -- or when a
    bucket is crowded.
    """
    n = x.shape[0]
    start = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        ph = x[j]
        if not (ph >= 0.0 and ph <= 1.0):
            return np.argsort(x, kind="mergesort")
        b = int(ph * n)
        if b > n - 1:
            b = n - 1
        start[b + 1] += 1
    fullest = 0
    for b in range(n):
        if start[b + 1] > fullest:
            fullest = start[b + 1]
        start[b + 1] += start[b]
    if fullest > _MAX_BUCKET:
        return np.argsort(x, kind="mergesort")
    order = np.empty(n, dtype=np.int64)
    for j in range(n):
        b = int(x[j] * n)
        if b > n - 1:
            b = n - 1
        order[start[b]] = j
        start[b] += 1
    for i in range(1, n):
        key = order[i]
        ph = x[key]
        k = i - 1
        while k >= 0 and x[order[k]] > ph:
            order[k + 1] = order[k]
            k -= 1
        order[k + 1] = key
    return order


@njit(cache=True)  # pragma: no cover - njit
def _line(cw, cwx, cwxx, cwa, cwxa, xs, ws, a, m, h, n, cv, out):  # type: ignore[no-untyped-def]
    """Windowed weighted linear fit at every center (see ``_line_values``)."""
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
    """Prefix sums of ``w*a`` and ``w*x*a`` over the padded row."""
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
    """Circular value padding: ``h`` wrapped samples on each side."""
    for j in range(h):
        out[j] = a[n - h + j]
    for j in range(n):
        out[h + j] = a[j]
    for j in range(h):
        out[h + n + j] = a[j]


@njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
def ss_kernel(  # type: ignore[no-untyped-def]
    tau, y, w, inv_dy, freqs, windows, span_fracs, j_mid, j_fin,
    alpha, use_alpha, baseline,
):
    """SuperSmoother score per trial frequency, one frequency per ``prange`` step.

    The same chain as the vectorized path -- stable sort, circular pad, prefix sums,
    constant-span linear fits with exact leave-one-out, span selection,
    interpolation, final smooth.
    """
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
        # Stable, like the array-API sort: the windows are position-based, so tied
        # phases must be ordered the same way on every backend.
        order = _stable_phase_order(x)
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


__all__ = ["ss_kernel"]
