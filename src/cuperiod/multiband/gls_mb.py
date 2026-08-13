"""Multi-band generalized Lomb-Scargle (VanderPlas & Ivezić 2015).

Several filters of the same star are fit jointly at each trial frequency, which
recovers a period even when no single band is sampled well enough on its own — the
common situation for sparse multi-survey photometry and the Rubin/LSST cadence.

Three models are offered (``GLSSettings.mb_model``), all named by how the bands
share the signal:

``"offsets"`` (default)
    One sinusoid on a shared phase plus an independent constant offset per band —
    the shared-phase ``(N_base, N_band) = (1, 0)`` model of VanderPlas & Ivezić
    (2015), their recommended search model for sparse multi-band data: all phase
    information pools into two signal parameters while each band spends only one
    nuisance parameter. cuPeriod evaluates it in closed form from band-projected
    trigonometric sums — the per-band offsets are profiled out analytically,
    leaving the Zechmeister-Kürster assembly with every sum replaced by its
    band-centered counterpart — using ``K + 2`` NUFFTs, so the cost is close to a
    single single-band GLS over the stacked points.

``"perband"``
    The multi-phase ``(0, 1)`` model: each band is fit with its own floating-mean
    sinusoid (independent amplitude *and phase*) and the per-band standard powers
    are combined with the reference-chi-squared weights of VanderPlas & Ivezić
    (2015, eq. 23): ``P = sum_k chi2_0k P_k / sum_k chi2_0k``. Note astropy's
    ``method="fast"`` intends this combination but weighs bands by the summed
    squared periodogram instead of ``chi2_0k`` (making its result depend on the
    frequency grid); cuPeriod implements the published weighting.

``"flex"``
    The flexible regularized model as implemented by astropy's
    ``LombScargleMultiband`` flexible method: ``mb_nterms_base`` shared harmonics
    plus ``mb_nterms_band`` harmonics-with-offset per band, ridge-regularized to
    lift the base/band degeneracy. cuPeriod builds the per-frequency normal
    equations from per-band harmonic trig sums (NUFFT or direct) and solves them
    in batch, reproducing astropy's power to float64 accuracy — including the
    trace-scaled ridge — on CPU, CUDA, or any torch device.

The ``"astropy"`` backend delegates to ``astropy.timeseries.LombScargleMultiband``
(``offsets`` maps to the flexible solver with ``(1, 0)`` terms, ``perband`` to
``(0, 1)``) and is the reference the native paths are tested against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
from cuperiod.core.config import GLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.gls import (
    CufinufftGLS,
    _gls_sums_direct,
    _trig_sums,
    lombscargle_power,
    lombscargle_power_torch,
)

#: A band needs this many finite points to get its own sinusoid (perband model).
_MIN_PERBAND_POINTS = 3

#: Byte budget for one ``(chunk, p, p)`` normal-equation stack; the frequency
#: chunk adapts to the parameter count so large-``nterms``/many-band models
#: cannot blow device memory.
_FLEX_CHUNK_BYTES = 1 << 27


@dataclass(frozen=True)
class _MBPrep:
    """Stacked, band-contiguous arrays shared by the native multi-band paths.

    ``tau`` is referenced to the global earliest time; ``w`` sums to 1 across all
    bands (inverse-variance, or uniform when any band lacks errors — mixing
    weighted and unweighted bands would make the joint fit ill-defined) and
    ``w_scale`` restores the raw ``sum(1/dy^2)`` scale where an absolute
    normal-matrix scale matters (the flex model's un-traced ridge); ``y`` has each
    band's weighted mean removed, so the per-band data sums need no explicit
    offset terms and ``chi2_ref`` — the weighted variance about the per-band
    means — is the reference chi-squared of the offsets-only null model.
    """

    tau: FloatArray
    w: FloatArray
    y: FloatArray
    value: FloatArray
    slices: tuple[slice, ...]
    band_weight: tuple[float, ...]
    w_scale: float
    chi2_ref: float
    n: int
    baseline: float
    band_names: tuple[str, ...]


def _prep_multiband(mblc: MultiBandLightCurve, settings: GLSSettings) -> _MBPrep:
    """Validate, stack, weight, and band-center a multi-band light curve."""
    finite = mblc.finite()
    bands = [(name, lc) for name, lc in finite.bands.items() if lc.n > 0]
    n = sum(lc.n for _, lc in bands)
    if n < settings.min_detections:
        raise InsufficientDataError(
            f"GLS multiband: {n} finite points < min_detections "
            f"{settings.min_detections}"
        )
    time = np.concatenate([lc.time for _, lc in bands])
    baseline = float(time.max() - time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("GLS multiband: no usable time baseline")

    value = np.concatenate([lc.value for _, lc in bands])
    use_errors = all(lc.error is not None for _, lc in bands)
    if use_errors:
        error = np.concatenate([lc.error for _, lc in bands])  # type: ignore[misc]
        w = 1.0 / (error * error)
    else:
        w = np.ones_like(time)
    w_scale = float(w.sum())
    w /= w_scale

    tau = time - time.min()
    slices: list[slice] = []
    band_weight: list[float] = []
    y = value.copy()
    start = 0
    for _, lc in bands:
        stop = start + lc.n
        sl = slice(start, stop)
        wk = float(w[sl].sum())
        y[sl] -= float(np.dot(w[sl], value[sl])) / wk
        slices.append(sl)
        band_weight.append(wk)
        start = stop
    chi2_ref = float(np.dot(w, y * y))
    return _MBPrep(
        tau=tau,
        w=w,
        y=y,
        value=value,
        slices=tuple(slices),
        band_weight=tuple(band_weight),
        w_scale=w_scale,
        chi2_ref=chi2_ref,
        n=n,
        baseline=baseline,
        band_names=tuple(name for name, _ in bands),
    )


def _nufft_sums(
    tau: Any,
    strengths: Any,
    f0: float,
    df: float,
    nf: int,
    backend: str,
    eps: float,
    engine: object | None,
) -> Any:
    """One type-1 trig-sum transform, through the batch engine when one is supplied.

    The batch runner (and the interop partition kernel) hand the single-band
    :class:`~cuperiod.methods.gls.CufinufftGLS` engine to the multi-band path too;
    its bucketed plan cache serves the ``K + 2`` offsets transforms and the flex
    harmonic sums equally well, since a plan is fixed by mode count and ``n_trans``
    alone. Anything other than a cufinufft engine on the cufinufft backend falls
    through to the plain per-call transform.
    """
    if backend == "cufinufft" and isinstance(engine, CufinufftGLS):
        return engine.trig_sums(tau, strengths, f0, df, nf)
    return _trig_sums(tau, strengths, f0, df, nf, backend, eps)  # type: ignore[arg-type]


# --- offsets model -----------------------------------------------------------


def _offsets_assemble(
    xp: Any,
    c2: Any,
    s2: Any,
    yc: Any,
    ys: Any,
    band_cs: list[tuple[Any, Any]],
    band_weight: Sequence[Any],
    chi2_ref: Any,
) -> Any:
    """Offsets-model power from global and per-band trig sums.

    The per-band constant offsets are profiled out of the least squares, which
    turns the usual floating-mean corrections ``cc - c*c`` (etc.) into their
    band-projected counterparts ``CC - sum_k C_k^2 / W_k``: the same
    Zechmeister-Kürster quadratic form, evaluated in the subspace orthogonal to
    every band's constant. ``yc``/``ys`` are already band-centered because the
    data strengths were. Degenerate frequencies map to 0. Everything is
    elementwise, so the inputs may carry a leading batch axis (the bootstrap
    path stacks resamples, with per-batch ``band_weight``/``chi2_ref``
    columns).
    """
    cc = 0.5 * (1.0 + c2)
    ss = 0.5 * (1.0 - c2)
    cs = 0.5 * s2
    for (c_k, s_k), wk in zip(band_cs, band_weight, strict=True):
        cc = cc - c_k * c_k / wk
        ss = ss - s_k * s_k / wk
        cs = cs - c_k * s_k / wk
    denom = chi2_ref * (cc * ss - cs * cs)
    power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / denom
    return xp.where(xp.isfinite(power), power, xp.zeros_like(power))


def _offsets_power_nufft(
    prep: _MBPrep,
    f0: float,
    df: float,
    nf: int,
    backend: str,
    eps: float,
    engine: object | None = None,
) -> FloatArray:
    """Offsets-model power via NUFFT band sums (finufft on CPU, cufinufft on GPU).

    ``K + 2`` type-1 transforms: per band the weight sums ``(C_k, S_k)``, one
    global transform of the band-centered data strengths ``w*y``, and one global
    doubled-grid transform of the weights. On the cufinufft path all inputs move
    to the device once and the assembly stays there; only the final power crosses
    back to the host.
    """
    tau: Any = prep.tau
    w: Any = prep.w
    wy: Any = prep.w * prep.y
    if backend == "cufinufft":
        ensure_cuda_dll_path()
        import cupy as cp

        xp: Any = cp
        tau, w, wy = cp.asarray(tau), cp.asarray(w), cp.asarray(wy)
    else:
        xp = np

    band_cs: list[tuple[Any, Any]] = []
    for sl in prep.slices:
        sw_k = _nufft_sums(tau[sl], w[sl][None, :], f0, df, nf, backend, eps, engine)[0]
        band_cs.append((sw_k.real, sw_k.imag))
    swy = _nufft_sums(tau, wy[None, :], f0, df, nf, backend, eps, engine)[0]
    sw2 = _nufft_sums(tau, w[None, :], 2.0 * f0, 2.0 * df, nf, backend, eps, engine)[0]

    if xp is np:
        with np.errstate(divide="ignore", invalid="ignore"):
            power = _offsets_assemble(
                xp, sw2.real, sw2.imag, swy.real, swy.imag,
                band_cs, prep.band_weight, prep.chi2_ref,
            )
    else:
        power = _offsets_assemble(
            xp, sw2.real, sw2.imag, swy.real, swy.imag,
            band_cs, prep.band_weight, prep.chi2_ref,
        )
    return to_host(power)


def _offsets_power_torch(
    prep: _MBPrep,
    f0: float,
    df: float,
    nf: int,
    *,
    device: str,
    precision: str,
    freq_batch: int,
) -> FloatArray:
    """Offsets-model power via the portable per-band direct trig sums.

    Runs :func:`~cuperiod.methods.gls._gls_sums_direct` once per band (the global
    sums are the accumulated band sums) and feeds the band-projected assembly, so
    it works on any torch device — including float32-only Apple MPS.
    """
    import torch

    prec = resolve_precision(precision, device)
    tdtype = torch.float32 if prec == "float32" else torch.float64
    tau_d = to_device_array(prep.tau, device=device, dtype=tdtype)
    w_d = to_device_array(prep.w, device=device, dtype=tdtype)
    wy_d = to_device_array(prep.w * prep.y, device=device, dtype=tdtype)
    xp = array_namespace(tau_d)

    yc = ys = c2 = s2 = None
    band_cs: list[tuple[Any, Any]] = []
    for sl in prep.slices:
        c_k, s_k, yc_k, ys_k, c2_k, s2_k = _gls_sums_direct(
            xp, tau_d[sl], w_d[sl], wy_d[sl], f0, df, nf, freq_batch=freq_batch
        )
        band_cs.append((c_k, s_k))
        yc = yc_k if yc is None else yc + yc_k
        ys = ys_k if ys is None else ys + ys_k
        c2 = c2_k if c2 is None else c2 + c2_k
        s2 = s2_k if s2 is None else s2 + s2_k
    power = _offsets_assemble(
        xp, c2, s2, yc, ys, band_cs, prep.band_weight, prep.chi2_ref
    )
    return to_host(power)


# --- perband model -----------------------------------------------------------


def _perband_power(
    mblc: MultiBandLightCurve,
    f0: float,
    df: float,
    nf: int,
    backend: str,
    settings: GLSSettings,
    engine: object | None = None,
) -> tuple[FloatArray, int, float, tuple[str, ...]]:
    """Multi-phase ``(0, 1)`` model: chi2_0-weighted per-band floating-mean GLS.

    Each band with at least :data:`_MIN_PERBAND_POINTS` finite points is fit
    independently (its own amplitude and phase) with the native single-band GLS on
    the shared grid, and the standard-normalized powers are combined with the
    per-band reference chi-squared weights of VanderPlas & Ivezić (2015, eq. 23).
    """
    finite = mblc.finite()
    bands = [
        (name, lc) for name, lc in finite.bands.items()
        if lc.n >= _MIN_PERBAND_POINTS
    ]
    n = sum(lc.n for _, lc in bands)
    if not bands or n < settings.min_detections:
        raise InsufficientDataError(
            f"GLS multiband: {n} finite points across usable bands "
            f"< min_detections {settings.min_detections}"
        )
    time = np.concatenate([lc.time for _, lc in bands])
    baseline = float(time.max() - time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("GLS multiband: no usable time baseline")

    combined: FloatArray | None = None
    weight_sum = 0.0
    for _, lc in bands:
        w_k = np.ones(lc.n) if lc.error is None else 1.0 / (lc.error * lc.error)
        mean_k = float(np.dot(w_k, lc.value) / w_k.sum())
        chi2_0k = float(np.dot(w_k, (lc.value - mean_k) ** 2))
        if backend == "torch" or backend.startswith("torch:"):
            device = backend.split(":", 1)[1] if ":" in backend else "cpu"
            p_k = lombscargle_power_torch(
                lc.time, lc.value, lc.error, f0, df, nf,
                fit_mean=True,
                device=device,
                precision=settings.precision,
                freq_batch=settings.direct_freq_batch,
            )
        elif backend == "cufinufft" and isinstance(engine, CufinufftGLS):
            p_k = engine.power(
                lc.time, lc.value, lc.error, f0, df, nf, fit_mean=True
            )
        else:
            p_k = lombscargle_power(
                lc.time, lc.value, lc.error, f0, df, nf,
                fit_mean=True,
                backend=backend,  # type: ignore[arg-type]
                eps=settings.nufft_eps,
            )
        contribution = chi2_0k * p_k
        combined = contribution if combined is None else combined + contribution
        weight_sum += chi2_0k
    assert combined is not None
    power = combined / weight_sum if weight_sum > 0.0 else np.zeros_like(combined)
    return power, n, baseline, tuple(name for name, _ in bands)


# --- flexible model ----------------------------------------------------------


@dataclass(frozen=True)
class _BandSums:
    """Per-band harmonic trig sums feeding the flex normal equations.

    ``cw[m]``/``sw[m]`` hold ``sum_j w_j cos/sin(2 pi m f tau_j)`` over this
    band's points for ``m = 1..H`` (frequency arrays); ``cy[n]``/``sy[n]`` the
    same with strengths ``w*y``. ``w0``/``y0`` are the ``m = 0`` scalars (the
    band's total weight and weighted data sum).
    """

    cw: list[Any]
    sw: list[Any]
    cy: list[Any]
    sy: list[Any]
    w0: float
    y0: float


def _flex_orders(nterms_base: int, nterms_band: int) -> tuple[int, int]:
    """(weight-sum harmonic order, data-sum harmonic order) for the flex model.

    Normal-matrix entries are products of two harmonics of order at most
    ``max(nb, nk)``, so the weight sums are needed to twice that; the right-hand
    side pairs one harmonic with the data.
    """
    m = max(nterms_base, nterms_band)
    return 2 * m, m


def _flex_sums_nufft(
    prep: _MBPrep, f0: float, df: float, nf: int, backend: str, eps: float,
    h_w: int, h_wy: int, engine: object | None = None,
) -> list[_BandSums]:
    """Per-band harmonic sums via NUFFT (harmonic ``m`` runs on the ``m``-scaled
    grid; the ``w``/``w*y`` pair shares one batched transform where both are
    needed)."""
    tau: Any = prep.tau
    w: Any = prep.w
    wy: Any = prep.w * prep.y
    if backend == "cufinufft":
        ensure_cuda_dll_path()
        import cupy as cp

        tau, w, wy = cp.asarray(tau), cp.asarray(w), cp.asarray(wy)

    xp: Any = np
    if backend == "cufinufft":
        import cupy as cp

        xp = cp
    out: list[_BandSums] = []
    for sl, wk in zip(prep.slices, prep.band_weight, strict=True):
        cw: list[Any] = [None]
        sw: list[Any] = [None]
        cy: list[Any] = [None]
        sy: list[Any] = [None]
        pair = xp.stack([w[sl], wy[sl]])
        for m in range(1, h_w + 1):
            strengths = pair if m <= h_wy else pair[:1]
            sums = _nufft_sums(
                tau[sl], strengths, m * f0, m * df, nf, backend, eps, engine
            )
            cw.append(sums[0].real)
            sw.append(sums[0].imag)
            if m <= h_wy:
                cy.append(sums[1].real)
                sy.append(sums[1].imag)
        y0 = float(np.dot(prep.w[sl], prep.y[sl]))
        out.append(_BandSums(cw=cw, sw=sw, cy=cy, sy=sy, w0=wk, y0=y0))
    return out


def _flex_sums_direct(
    xp: Any, tau: Any, w: Any, wy: Any, f0: float, df: float, nf: int,
    h_w: int, h_wy: int, freq_batch: int,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """One band's harmonic sums by direct trig evaluation (the portable path).

    Harmonics are chained with the angle-addition recurrence, so ``cos``/``sin``
    are evaluated once per frequency regardless of the harmonic order. Chunked
    over frequency like the single-band direct path.
    """
    from cuperiod.methods.gls import _DIRECT_CHUNK_ELEMS

    fdtype = tau.dtype
    dev = device_ref(tau)
    n_points = int(tau.shape[0])
    chunk = max(1, min(freq_batch, _DIRECT_CHUNK_ELEMS // max(1, n_points)))
    freqs = f0 + df * xp.arange(nf, dtype=fdtype, device=dev)
    cw = [None] + [xp.empty(nf, dtype=fdtype, device=dev) for _ in range(h_w)]
    sw = [None] + [xp.empty(nf, dtype=fdtype, device=dev) for _ in range(h_w)]
    cy = [None] + [xp.empty(nf, dtype=fdtype, device=dev) for _ in range(h_wy)]
    sy = [None] + [xp.empty(nf, dtype=fdtype, device=dev) for _ in range(h_wy)]
    two_pi = 2.0 * float(np.pi)
    w_row = w[None, :]
    wy_row = wy[None, :]
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        ang = (two_pi * freqs[start:stop])[:, None] * tau[None, :]
        cos1 = xp.cos(ang)
        sin1 = xp.sin(ang)
        cos_m, sin_m = cos1, sin1
        for m in range(1, h_w + 1):
            cw[m][start:stop] = xp.sum(w_row * cos_m, axis=1)
            sw[m][start:stop] = xp.sum(w_row * sin_m, axis=1)
            if m <= h_wy:
                cy[m][start:stop] = xp.sum(wy_row * cos_m, axis=1)
                sy[m][start:stop] = xp.sum(wy_row * sin_m, axis=1)
            if m < h_w:
                cos_next = cos_m * cos1 - sin_m * sin1
                sin_m = sin_m * cos1 + cos_m * sin1
                cos_m = cos_next
    return cw, sw, cy, sy


#: Column descriptor: (band index or -1 for base, kind, harmonic order).
_FlexCol = tuple[int, str, int]


def _flex_columns(n_bands: int, nb: int, nk: int) -> list[_FlexCol]:
    """The flex design columns in astropy's order (base block, then per band)."""
    cols: list[_FlexCol] = [(-1, "const", 0)]
    for n in range(1, nb + 1):
        cols.append((-1, "sin", n))
        cols.append((-1, "cos", n))
    for b in range(n_bands):
        cols.append((b, "const", 0))
        for n in range(1, nk + 1):
            cols.append((b, "sin", n))
            cols.append((b, "cos", n))
    return cols


def _pair_entry(
    c_of: Any, s_of: Any, ki: str, ni: int, kj: str, nj: int
) -> Any:
    """``<col_i, col_j>_w`` from the restriction's harmonic sums.

    ``c_of(m)``/``s_of(m)`` return the ``cos``/``sin`` weight sums of the
    relevant restriction (one band, or the whole set) at harmonic ``m``; the
    product-to-sum identities reduce every entry to at most two of them.
    """
    if ki == "const" and kj == "const":
        return c_of(0)
    if ki == "const":
        return s_of(nj) if kj == "sin" else c_of(nj)
    if kj == "const":
        return s_of(ni) if ki == "sin" else c_of(ni)
    if ki == "sin" and kj == "sin":
        return 0.5 * (c_of(abs(ni - nj)) - c_of(ni + nj))
    if ki == "cos" and kj == "cos":
        return 0.5 * (c_of(abs(ni - nj)) + c_of(ni + nj))
    n_s, n_c = (ni, nj) if ki == "sin" else (nj, ni)
    if n_s == n_c:
        return 0.5 * s_of(n_s + n_c)
    sign = 1.0 if n_s > n_c else -1.0
    return 0.5 * (s_of(n_s + n_c) + sign * s_of(abs(n_s - n_c)))


def _flex_power_from_sums(
    xp: Any,
    band_sums: list[_BandSums],
    chi2_ref: float,
    w_scale: float,
    nf: int,
    settings: GLSSettings,
    fdtype: Any,
    dev: Any,
) -> Any:
    """Assemble and solve the regularized normal equations in frequency batches.

    Reproduces astropy's ``lombscargle_mbflex``: normal matrix ``G`` and
    right-hand side ``b`` built from the (band-centered, weighted) harmonic sums,
    ridge ``diag(G) += trace(G) * reg`` (or ``+= reg`` when
    ``mb_regularize_by_trace=False`` — the sums are restored to astropy's raw
    ``1/dy^2`` scale via ``w_scale`` so the absolute ridge means the same thing),
    then ``P = b . solve(G, b) / chi2_ref``. Singular chunks fall back to a
    per-frequency least-squares solve on the host, mirroring astropy's
    ``lstsq`` fallback.
    """
    nb, nk = settings.mb_nterms_base, settings.mb_nterms_band
    n_bands = len(band_sums)
    cols = _flex_columns(n_bands, nb, nk)
    p = len(cols)
    n_base = 1 + 2 * nb

    reg = np.zeros(p, dtype=np.float64)
    if settings.mb_reg_base is not None:
        reg[:n_base] = settings.mb_reg_base
    if settings.mb_reg_band is not None:
        reg[n_base:] = settings.mb_reg_band
    reg_d = (
        xp.asarray(reg, dtype=fdtype)
        if dev is None
        else xp.asarray(reg, dtype=fdtype, device=dev)
    )

    scale = float(w_scale)
    chi2_0 = chi2_ref * scale

    def restriction(b: int, sl: slice) -> tuple[Any, Any, Any, Any]:
        """(c_of, s_of, cy_of, sy_of) accessors for band ``b`` (or all, b=-1)."""
        if b >= 0:
            bs = band_sums[b]

            def c_of(m: int) -> Any:
                return bs.w0 * scale if m == 0 else scale * bs.cw[m][sl]

            def s_of(m: int) -> Any:
                return 0.0 if m == 0 else scale * bs.sw[m][sl]

            def cy_of(m: int) -> Any:
                return bs.y0 * scale if m == 0 else scale * bs.cy[m][sl]

            def sy_of(m: int) -> Any:
                return 0.0 if m == 0 else scale * bs.sy[m][sl]

        else:

            def c_of(m: int) -> Any:
                if m == 0:
                    return sum(bs.w0 for bs in band_sums) * scale
                total = band_sums[0].cw[m][sl]
                for bs in band_sums[1:]:
                    total = total + bs.cw[m][sl]
                return scale * total

            def s_of(m: int) -> Any:
                if m == 0:
                    return 0.0
                total = band_sums[0].sw[m][sl]
                for bs in band_sums[1:]:
                    total = total + bs.sw[m][sl]
                return scale * total

            def cy_of(m: int) -> Any:
                if m == 0:
                    return sum(bs.y0 for bs in band_sums) * scale
                total = band_sums[0].cy[m][sl]
                for bs in band_sums[1:]:
                    total = total + bs.cy[m][sl]
                return scale * total

            def sy_of(m: int) -> Any:
                if m == 0:
                    return 0.0
                total = band_sums[0].sy[m][sl]
                for bs in band_sums[1:]:
                    total = total + bs.sy[m][sl]
                return scale * total

        return c_of, s_of, cy_of, sy_of

    diag_idx = xp.arange(p) if dev is None else xp.arange(p, device=dev)
    power = (
        xp.empty(nf, dtype=fdtype)
        if dev is None
        else xp.empty(nf, dtype=fdtype, device=dev)
    )
    chunk = max(256, _FLEX_CHUNK_BYTES // (p * p * 8))
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        sl = slice(start, stop)
        n_chunk = stop - start
        if dev is None:
            g = xp.zeros((n_chunk, p, p), dtype=fdtype)
            b_vec = xp.zeros((n_chunk, p), dtype=fdtype)
        else:
            g = xp.zeros((n_chunk, p, p), dtype=fdtype, device=dev)
            b_vec = xp.zeros((n_chunk, p), dtype=fdtype, device=dev)
        accessors = {b: restriction(b, sl) for b in range(-1, n_bands)}
        for i, (bi, ki, ni) in enumerate(cols):
            c_of, s_of, cy_of, sy_of = accessors[bi]
            b_vec[:, i] = sy_of(ni) if ki == "sin" else cy_of(ni)
            for j in range(i, p):
                bj, kj, nj = cols[j]
                if bi >= 0 and bj >= 0 and bi != bj:
                    continue  # disjoint band indicators: exactly zero
                # The tighter restriction wins: base x band-k pairs live on
                # band k's points.
                c_r, s_r, _, _ = accessors[bj if bj >= 0 else bi]
                entry = _pair_entry(c_r, s_r, ki, ni, kj, nj)
                g[:, i, j] = entry
                if j != i:
                    g[:, j, i] = entry
        trace = xp.sum(g[:, diag_idx, diag_idx], axis=-1)
        if settings.mb_regularize_by_trace:
            g[:, diag_idx, diag_idx] += trace[:, None] * reg_d
        else:
            g[:, diag_idx, diag_idx] += reg_d
        try:
            theta = xp.linalg.solve(g, b_vec[..., None])[..., 0]
            dchi2 = xp.sum(b_vec * theta, axis=-1)
        except Exception:
            # Singular normal matrix somewhere in the chunk (e.g. ridge disabled):
            # astropy falls back to lstsq per frequency; do the same on the host.
            g_h, b_h = to_host(g), to_host(b_vec)
            dchi2_h = np.empty(n_chunk, dtype=np.float64)
            for k in range(n_chunk):
                try:
                    theta_k = np.linalg.solve(g_h[k], b_h[k])
                except np.linalg.LinAlgError:
                    theta_k = np.linalg.lstsq(g_h[k], b_h[k], rcond=None)[0]
                dchi2_h[k] = float(b_h[k] @ theta_k)
            dchi2 = (
                xp.asarray(dchi2_h, dtype=fdtype)
                if dev is None
                else xp.asarray(dchi2_h, dtype=fdtype, device=dev)
            )
        power[sl] = dchi2 / chi2_0
    return xp.where(xp.isfinite(power), power, xp.zeros_like(power))


def _flex_power_nufft(
    prep: _MBPrep, f0: float, df: float, nf: int, backend: str,
    settings: GLSSettings, engine: object | None = None,
) -> FloatArray:
    """Flex-model power via NUFFT harmonic sums + batched normal-equation solve."""
    h_w, h_wy = _flex_orders(settings.mb_nterms_base, settings.mb_nterms_band)
    band_sums = _flex_sums_nufft(
        prep, f0, df, nf, backend, settings.nufft_eps, h_w, h_wy, engine
    )
    if backend == "cufinufft":
        import cupy as cp

        xp: Any = cp
        fdtype: Any = cp.float64
        dev = None
        power = _flex_power_from_sums(
            xp, band_sums, prep.chi2_ref, prep.w_scale, nf, settings, fdtype, dev
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            power = _flex_power_from_sums(
                np, band_sums, prep.chi2_ref, prep.w_scale, nf, settings,
                np.float64, None,
            )
    return to_host(power)


def _flex_power_torch(
    prep: _MBPrep, f0: float, df: float, nf: int, *,
    device: str, settings: GLSSettings,
) -> FloatArray:
    """Flex-model power via direct per-band harmonic sums on any torch device."""
    import torch

    prec = resolve_precision(settings.precision, device)
    tdtype = torch.float32 if prec == "float32" else torch.float64
    h_w, h_wy = _flex_orders(settings.mb_nterms_base, settings.mb_nterms_band)
    tau_d = to_device_array(prep.tau, device=device, dtype=tdtype)
    w_d = to_device_array(prep.w, device=device, dtype=tdtype)
    wy_d = to_device_array(prep.w * prep.y, device=device, dtype=tdtype)
    xp = array_namespace(tau_d)
    band_sums: list[_BandSums] = []
    for sl, wk in zip(prep.slices, prep.band_weight, strict=True):
        cw, sw, cy, sy = _flex_sums_direct(
            xp, tau_d[sl], w_d[sl], wy_d[sl], f0, df, nf, h_w, h_wy,
            settings.direct_freq_batch,
        )
        y0 = float(np.dot(prep.w[sl], prep.y[sl]))
        band_sums.append(_BandSums(cw=cw, sw=sw, cy=cy, sy=sy, w0=wk, y0=y0))
    power = _flex_power_from_sums(
        xp, band_sums, prep.chi2_ref, prep.w_scale, nf, settings,
        tdtype, device_ref(tau_d),
    )
    return to_host(power)


# --- astropy reference path --------------------------------------------------


def _astropy_power(
    grid: GridSpec, mblc: MultiBandLightCurve, settings: GLSSettings
) -> tuple[FloatArray, int, float, tuple[str, ...]]:
    """Reference multi-band power from ``astropy.timeseries.LombScargleMultiband``.

    The ``"offsets"`` model maps to the flexible solver with
    ``(nterms_base, nterms_band) = (1, 0)`` and ``"perband"`` to ``(0, 1)`` —
    the same model spaces, lifted from exact degeneracy by astropy's default
    ridge — while ``"flex"`` passes the configured term counts and
    regularization through.
    """
    from astropy.timeseries import LombScargleMultiband

    finite = mblc.finite()
    time, value, error, band = finite.stacked()
    n = int(time.size)
    if n < settings.min_detections:
        raise InsufficientDataError(
            f"GLS multiband: {n} finite points < min_detections "
            f"{settings.min_detections}"
        )
    baseline = float(time.max() - time.min()) if n else 0.0
    if baseline <= 0.0:
        raise InsufficientDataError("GLS multiband: no usable time baseline")
    if settings.mb_model == "offsets":
        nterms_base, nterms_band = 1, 0
    elif settings.mb_model == "perband":
        nterms_base, nterms_band = 0, 1
    else:
        nterms_base, nterms_band = settings.mb_nterms_base, settings.mb_nterms_band
    ls = LombScargleMultiband(
        time,
        value,
        band,
        error,
        nterms_base=nterms_base,
        nterms_band=nterms_band,
        reg_base=settings.mb_reg_base,
        reg_band=settings.mb_reg_band,
        regularize_by_trace=settings.mb_regularize_by_trace,
    )
    power = np.nan_to_num(
        np.asarray(ls.power(grid.frequency, method="flexible"), dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return power, n, baseline, finite.band_names


# --- entry point -------------------------------------------------------------


def gls_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: GLSSettings,
    backend: str,
    engine: object | None = None,
) -> Periodogram:
    """Compute the multi-band GLS power on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Frequency grid (built from the stacked baseline).
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : GLSSettings
        GLS settings; ``mb_model`` selects the multi-band model and the
        ``mb_*`` fields configure the flexible variant.
    backend : str
        Resolved backend: ``"finufft"``, ``"cufinufft"``, ``"torch"`` /
        ``"torch:<device>"``, or ``"astropy"``.
    engine : object, optional
        A :class:`~cuperiod.methods.gls.CufinufftGLS` batch engine; on the
        cufinufft backend its plan cache is reused across bands, harmonics, and
        light curves. Ignored on every other backend. The bootstrap-FAP pass
        deliberately does not use it: the resample chunks batch through
        transforms whose ``n_trans`` varies with the grid size, and caching a
        plan per ``n_trans`` value would balloon device memory in a batch run.

    Returns
    -------
    Periodogram
        The joint power spectrum. ``meta["bands"]`` lists the contributing bands
        and ``meta["mb_model"]`` records the model.

    Raises
    ------
    InsufficientDataError
        If too few finite points remain across all bands, or the grid is empty.
    """
    if grid.size == 0:
        raise InsufficientDataError("GLS multiband: empty trial grid")

    finite_meta = dict(mblc.finite().meta)
    if backend == "astropy" or not grid.uniform:
        power, n, baseline, band_names = _astropy_power(grid, mblc, settings)
        actual_backend = "astropy"
        frequency = grid.frequency
    else:
        f0, df, nf = grid.uniform_frequency_params()
        frequency = f0 + df * np.arange(nf, dtype=np.float64)
        is_torch = backend == "torch" or backend.startswith("torch:")
        if settings.mb_model == "perband":
            if is_torch:
                device = resolve_torch_device(backend, settings.device)
                actual_backend = f"torch:{device}"
                backend = actual_backend
            else:
                actual_backend = backend
            power, n, baseline, band_names = _perband_power(
                mblc, f0, df, nf, backend, settings, engine
            )
        else:
            prep = _prep_multiband(mblc, settings)
            n, baseline, band_names = prep.n, prep.baseline, prep.band_names
            if settings.mb_model == "offsets":
                if is_torch:
                    device = resolve_torch_device(backend, settings.device)
                    actual_backend = f"torch:{device}"
                    power = _offsets_power_torch(
                        prep, f0, df, nf,
                        device=device,
                        precision=settings.precision,
                        freq_batch=settings.direct_freq_batch,
                    )
                else:
                    actual_backend = backend
                    power = _offsets_power_nufft(
                        prep, f0, df, nf, backend, settings.nufft_eps, engine
                    )
            else:  # flex
                if is_torch:
                    device = resolve_torch_device(backend, settings.device)
                    actual_backend = f"torch:{device}"
                    power = _flex_power_torch(
                        prep, f0, df, nf, device=device, settings=settings
                    )
                else:
                    actual_backend = backend
                    power = _flex_power_nufft(
                        prep, f0, df, nf, backend, settings, engine
                    )
    power = np.nan_to_num(
        np.asarray(power, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    extras: dict[str, FloatArray] = {}
    meta: dict[str, Any] = {
        **finite_meta, "bands": band_names, "mb_model": settings.mb_model,
    }
    if (
        settings.mb_fap_bootstrap > 0
        and actual_backend != "astropy"
        and grid.uniform
    ):
        from cuperiod.core.peaks import local_maxima
        from cuperiod.multiband.fap_mb import MultibandFAP, bootstrap_null_max

        null_max = bootstrap_null_max(
            mblc, grid, settings, backend,
            settings.mb_fap_bootstrap, settings.mb_fap_seed,
        )
        calibration = MultibandFAP(
            null_max=null_max,
            n_bootstrap=settings.mb_fap_bootstrap,
            seed=settings.mb_fap_seed,
            model=settings.mb_model,
            backend=actual_backend,
        )
        fap = np.full(power.shape, np.nan, dtype=np.float64)
        candidates = local_maxima(power)
        if candidates.size:
            fap[candidates] = calibration.fap(power[candidates])
        extras["fap"] = fap
        meta["fap_method"] = "bootstrap"
        meta["fap_n_bootstrap"] = settings.mb_fap_bootstrap
        meta["fap_seed"] = settings.mb_fap_seed
        meta["fap_level_10pct"] = float(np.quantile(null_max, 0.90))
        if settings.mb_fap_bootstrap >= 99:
            meta["fap_level_1pct"] = float(np.quantile(null_max, 0.99))
    return Periodogram.from_spectrum(
        method="GLS",
        backend=actual_backend,
        frequency=frequency,
        power=power,
        objective_sense="max",
        n_samples=n,
        baseline=baseline,
        extras=extras,
        meta=meta,
    )


__all__ = ["gls_multiband_power"]
