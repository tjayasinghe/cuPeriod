"""Independent reference periodograms for 1-1 validation.

Each function takes a light curve and an explicit frequency grid and returns the
statistic on that grid, so cuPeriod and the reference are compared on identical
grids. Sources:

  GLS          astropy.timeseries.LombScargle  (third-party, gold standard)
  PDM          PyAstronomy.pyTiming.pyPDM       (third-party, Stellingwerf)
  CE           Graham et al. 2013               (independent textbook impl)
  StringLength Dworetsky 1983                   (independent textbook impl)
  MHAOV        Schwarzenberg-Czerny 1996        (independent textbook impl)

BLS is validated against astropy.timeseries.BoxLeastSquares through cuPeriod's
own ``backend="astropy"`` path (guarantees an identical period/duration grid),
so it is handled in the harness, not here. TLS is validated separately against
the ``transitleastsquares`` package on the Mendeley transit dataset.
"""

from __future__ import annotations

import numpy as np


def ref_gls(t, y, dy, freqs):
    """astropy generalized Lomb-Scargle, ``normalization='standard'``."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy, fit_mean=True, center_data=True)
    return np.asarray(ls.power(freqs, normalization="standard"), dtype=np.float64)


def ref_pdm(t, y, freqs, *, n_bins=10, n_covers=3):
    """PyAstronomy Stellingwerf PDM theta on the given frequency grid."""
    from PyAstronomy.pyTiming import pyPDM
    # Scanner over an explicit frequency list (use our exact grid).
    f = np.ascontiguousarray(freqs, dtype=np.float64)
    df = float(np.median(np.diff(f)))
    scanner = pyPDM.Scanner(minVal=float(f[0]), maxVal=float(f[-1]) + 0.5 * df,
                            dVal=df, mode="frequency")
    pdm = pyPDM.PyPDM(np.asarray(t, float), np.asarray(y, float))
    rf, theta = pdm.pdmEquiBinCover(n_bins, n_covers, scanner)
    rf = np.asarray(rf); theta = np.asarray(theta)
    # align onto our grid (nearest) in case Scanner rounds the count
    if rf.shape == f.shape and np.allclose(rf, f, rtol=0, atol=0.51 * df):
        return theta.astype(np.float64)
    idx = np.clip(np.searchsorted(rf, f), 0, rf.size - 1)
    return theta[idx].astype(np.float64)


def ref_ce(t, y, freqs, *, n_phase=10, n_mag=10, batch=2048):
    """Conditional entropy H(m|phase), Graham et al. 2013 (natural log)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    tau = t - t.min()
    span = float(y.max() - y.min())
    n = t.size
    if span <= 0:
        return np.zeros(freqs.size)
    mag_bin = np.clip(((y - y.min()) / span * n_mag).astype(np.int64), 0, n_mag - 1)
    freqs = np.asarray(freqs, float)
    out = np.empty(freqs.size)
    ncell = n_phase * n_mag
    for s in range(0, freqs.size, batch):
        fb = freqs[s:s + batch]
        nf = fb.size
        phase = np.mod(tau[None, :] * fb[:, None], 1.0)
        pbin = np.clip((phase * n_phase).astype(np.int64), 0, n_phase - 1)
        cell = pbin * n_mag + mag_bin[None, :]
        flat = (np.arange(nf)[:, None] * ncell + cell).ravel()
        c = np.zeros(nf * ncell)
        np.add.at(c, flat, 1.0)
        c = c.reshape(nf, n_phase, n_mag)
        col = c.sum(axis=2, keepdims=True)
        mask = c > 0
        term = np.where(mask, c * (np.log(np.where(col > 0, col, 1.0)) -
                                   np.log(np.where(mask, c, 1.0))), 0.0)
        out[s:s + batch] = term.sum(axis=(1, 2)) / n
    return out


def ref_stringlength(t, y, freqs, *, batch=2048):
    """Dworetsky 1983 string length; magnitudes scaled to a span of 0.5."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    tau = t - t.min()
    span = float(y.max() - y.min())
    n = t.size
    if span <= 0:
        return np.zeros(freqs.size)
    m = (y - y.min()) / span * 0.5
    freqs = np.asarray(freqs, float)
    out = np.empty(freqs.size)
    for s in range(0, freqs.size, batch):
        fb = freqs[s:s + batch]
        nf = fb.size
        phase = np.mod(tau[None, :] * fb[:, None], 1.0)
        order = np.argsort(phase, axis=1)
        ph = np.take_along_axis(phase, order, axis=1)
        mm = np.take_along_axis(np.broadcast_to(m, (nf, n)), order, axis=1)
        dphi = np.diff(ph, axis=1); dmag = np.diff(mm, axis=1)
        total = np.sqrt(dphi * dphi + dmag * dmag).sum(axis=1)
        wphi = (ph[:, 0] + 1.0) - ph[:, -1]
        wmag = mm[:, 0] - mm[:, -1]
        out[s:s + batch] = total + np.sqrt(wphi * wphi + wmag * wmag)
    return out


def ref_mhaov(t, y, freqs, *, n_harmonics=3, batch=512):
    """Schwarzenberg-Czerny 1996 multiharmonic AOV F-statistic (least squares)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    tau = t - t.min()
    n = t.size
    ym = float(y.mean())
    total_ss = float(((y - ym) ** 2).sum())
    H = n_harmonics
    p = 2 * H + 1
    out = np.zeros(freqs.size)
    if total_ss <= 0 or n <= p:
        return out
    yc = y - ym
    freqs = np.asarray(freqs, float)
    for s in range(0, freqs.size, batch):
        fb = freqs[s:s + batch]
        for j, f in enumerate(fb):
            ang = 2.0 * np.pi * f * tau
            cols = [np.ones(n)]
            for k in range(1, H + 1):
                cols.append(np.cos(k * ang)); cols.append(np.sin(k * ang))
            A = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            model = A @ coef
            model_ss = float(((model - ym) ** 2).sum())
            resid_ss = total_ss - model_ss
            if resid_ss <= 0:
                out[s + j] = np.inf
            else:
                out[s + j] = ((n - p) / (2 * H)) * model_ss / resid_ss
    return out
