"""Bootstrap false-alarm statistics for the multi-band GLS.

astropy's ``LombScargleMultiband`` ships no false-alarm probabilities at all —
the single-band analytic formulas assume one sinusoid fit to one band. cuPeriod
calibrates the multi-band periodogram honestly instead, with a within-band
bootstrap: under the null hypothesis of no coherent signal, each band's
``(value, error)`` pairs are exchangeable across that band's epochs, so they are
resampled with replacement while every observation time stays fixed. That
preserves the window function, the per-band sample sizes, and the
heteroskedastic error distribution while destroying phase coherence. The maximum
power over the searched grid is recorded for each resample, and the false-alarm
probability of an observed peak is its rank in that null sample:
``FAP(z) = (1 + #{max_r >= z}) / (R + 1)``.

For the default ``offsets`` model on a NUFFT backend the bootstrap is nearly
free: the times never change, so every resample reuses the same nonuniform
points and the whole batch runs as multi-transform NUFFTs — the same ``K + 2``
transforms as one power evaluation, each carrying a stack of bootstrap
strengths. The ``perband``/``flex`` models and the torch backend fall back to an
explicit loop over resamples (still native-speed per iteration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import GLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.methods.gls import _trig_sums
from cuperiod.multiband.gls_mb import (
    _MBPrep,
    _offsets_assemble,
    _prep_multiband,
    gls_multiband_power,
)


@dataclass(frozen=True)
class MultibandFAP:
    """The null distribution of the multi-band peak power, from a bootstrap.

    Parameters
    ----------
    null_max : numpy.ndarray
        Maximum power over the searched grid for each of the ``n_bootstrap``
        within-band resamples.
    n_bootstrap : int
        Number of resamples.
    seed : int
        Seed the resampling used.
    model : str
        The multi-band model that was calibrated (``mb_model``).
    backend : str
        Backend that computed the null powers.
    """

    null_max: FloatArray
    n_bootstrap: int
    seed: int
    model: str
    backend: str

    def fap(self, power: Any) -> Any:
        """False-alarm probability of ``power`` against the bootstrap null.

        ``FAP(z) = (1 + #{null_max >= z}) / (n_bootstrap + 1)`` — the add-one
        rank estimate, whose smallest resolvable value is
        ``1 / (n_bootstrap + 1)``. Accepts a scalar or an array.
        """
        z = np.asarray(power, dtype=np.float64)
        sorted_null = np.sort(self.null_max)
        below = np.searchsorted(sorted_null, z, side="left")
        out = (1.0 + (sorted_null.size - below)) / (self.n_bootstrap + 1.0)
        return float(out) if np.ndim(power) == 0 else out

    def level(self, fap: float) -> float:
        """Power threshold whose false-alarm probability is ``fap``.

        The ``1 - fap`` quantile of the null maxima. Requires
        ``fap >= 1 / (n_bootstrap + 1)`` — below that the bootstrap cannot
        resolve the tail and more resamples are needed.
        """
        if not 0.0 < fap <= 1.0:
            raise ValueError(f"fap must be in (0, 1], got {fap}")
        if fap * (self.n_bootstrap + 1) < 1.0:
            raise ValueError(
                f"fap={fap} is below the bootstrap resolution "
                f"1/(n_bootstrap+1) = {1.0 / (self.n_bootstrap + 1):.2e}; "
                "increase n_bootstrap"
            )
        return float(np.quantile(self.null_max, 1.0 - fap))


#: Device-memory budget for one bootstrap batch of trig-sum outputs (bytes).
_FAP_CHUNK_BYTES = 1 << 28


def _resample_arrays(
    prep: _MBPrep, n_bootstrap: int, seed: int
) -> tuple[FloatArray, FloatArray, list[FloatArray], FloatArray]:
    """Within-band pair resampling, vectorized over all resamples.

    Returns the globally re-normalized weights ``(R, N)``, the band-centered
    resampled values' strengths ``w * y`` ``(R, N)``, the per-band total weights
    (each ``(R, 1)``), and the per-resample reference chi-squared ``(R, 1)``.
    """
    rng = np.random.default_rng(seed)
    n = prep.tau.size
    u_raw = prep.w * prep.w_scale
    u = np.empty((n_bootstrap, n), dtype=np.float64)
    y = np.empty((n_bootstrap, n), dtype=np.float64)
    for sl in prep.slices:
        n_k = sl.stop - sl.start
        idx = rng.integers(0, n_k, size=(n_bootstrap, n_k))
        u[:, sl] = u_raw[sl][idx]
        y[:, sl] = prep.value[sl][idx]
    w = u / u.sum(axis=1, keepdims=True)
    band_weight: list[FloatArray] = []
    for sl in prep.slices:
        w_k = w[:, sl].sum(axis=1, keepdims=True)
        mean_k = (w[:, sl] * y[:, sl]).sum(axis=1, keepdims=True) / w_k
        y[:, sl] -= mean_k
        band_weight.append(w_k)
    chi2_ref = (w * y * y).sum(axis=1, keepdims=True)
    return w, w * y, band_weight, chi2_ref


def _null_max_offsets_nufft(
    prep: _MBPrep,
    f0: float,
    df: float,
    nf: int,
    backend: str,
    eps: float,
    n_bootstrap: int,
    seed: int,
) -> FloatArray:
    """Null peak powers for the offsets model via multi-transform NUFFTs.

    All resamples share the observation times, so each batch runs the same
    ``K + 2`` transforms as a single power evaluation with the bootstrap
    strengths stacked along ``n_trans``; only the per-batch maxima return to
    the host.
    """
    w, wy, band_weight, chi2_ref = _resample_arrays(prep, n_bootstrap, seed)

    tau: Any = prep.tau
    if backend == "cufinufft":
        ensure_cuda_dll_path()
        import cupy as cp

        xp: Any = cp
        tau = cp.asarray(tau)
    else:
        xp = np

    n_bands = len(prep.slices)
    denom = 16 * max(nf, 1) * (n_bands + 6)
    r_chunk = max(1, _FAP_CHUNK_BYTES // denom)
    null_max = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, r_chunk):
        stop = min(start + r_chunk, n_bootstrap)
        rows = slice(start, stop)
        w_c: Any = xp.asarray(w[rows])
        wy_c: Any = xp.asarray(wy[rows])
        band_cs = []
        weights = []
        for sl, w_k in zip(prep.slices, band_weight, strict=True):
            sw_k = _trig_sums(tau[sl], w_c[:, sl], f0, df, nf, backend, eps)  # type: ignore[arg-type]
            band_cs.append((sw_k.real, sw_k.imag))
            weights.append(xp.asarray(w_k[rows]))
        swy = _trig_sums(tau, wy_c, f0, df, nf, backend, eps)  # type: ignore[arg-type]
        sw2 = _trig_sums(tau, w_c, 2.0 * f0, 2.0 * df, nf, backend, eps)  # type: ignore[arg-type]
        chi2_c = xp.asarray(chi2_ref[rows])
        if xp is np:
            with np.errstate(divide="ignore", invalid="ignore"):
                power = _offsets_assemble(
                    xp, sw2.real, sw2.imag, swy.real, swy.imag,
                    band_cs, weights, chi2_c,
                )
        else:
            power = _offsets_assemble(
                xp, sw2.real, sw2.imag, swy.real, swy.imag,
                band_cs, weights, chi2_c,
            )
        batch_max = xp.max(power, axis=1)
        null_max[rows] = np.asarray(
            batch_max.get() if hasattr(batch_max, "get") else batch_max,
            dtype=np.float64,
        )
    return null_max


def _null_max_loop(
    mblc: MultiBandLightCurve,
    grid: GridSpec,
    settings: GLSSettings,
    backend: str,
    n_bootstrap: int,
    seed: int,
) -> FloatArray:
    """Null peak powers by explicit recomputation (perband/flex models, torch).

    Each resample rebuilds the multi-band light curve with within-band
    resampled ``(value, error)`` pairs and reruns the native power path.
    """
    rng = np.random.default_rng(seed)
    finite = mblc.finite()
    quiet = settings.model_copy(update={"mb_fap_bootstrap": 0})
    null_max = np.empty(n_bootstrap, dtype=np.float64)
    for r in range(n_bootstrap):
        bands: dict[str, LightCurve] = {}
        for name, lc in finite.bands.items():
            if lc.n == 0:
                continue
            idx = rng.integers(0, lc.n, size=lc.n)
            bands[name] = LightCurve(
                time=lc.time,
                value=lc.value[idx],
                error=None if lc.error is None else lc.error[idx],
                domain=lc.domain,
            )
        resampled = MultiBandLightCurve.from_light_curves(bands, meta=finite.meta)
        pg = gls_multiband_power(grid, resampled, quiet, backend)
        null_max[r] = float(pg.power.max()) if pg.power.size else 0.0
    return null_max


def bootstrap_null_max(
    mblc: MultiBandLightCurve,
    grid: GridSpec,
    settings: GLSSettings,
    backend: str,
    n_bootstrap: int,
    seed: int,
) -> FloatArray:
    """Null distribution of the multi-band peak power on ``grid``.

    Dispatches to the batched-NUFFT fast path (offsets model on
    finufft/cufinufft) or the explicit loop (everything else).
    """
    if (
        settings.mb_model == "offsets"
        and backend in ("finufft", "cufinufft")
        and grid.uniform
    ):
        prep = _prep_multiband(mblc, settings)
        f0, df, nf = grid.uniform_frequency_params()
        return _null_max_offsets_nufft(
            prep, f0, df, nf, backend, settings.nufft_eps, n_bootstrap, seed
        )
    return _null_max_loop(mblc, grid, settings, backend, n_bootstrap, seed)


def multiband_fap(
    data: Any,
    settings: GLSSettings | None = None,
    *,
    grid: GridSpec | None = None,
    backend: str = "auto",
    n_bootstrap: int = 500,
    seed: int = 0,
) -> MultibandFAP:
    """Calibrate the multi-band GLS null distribution by within-band bootstrap.

    Parameters
    ----------
    data : various
        Anything :func:`cuperiod.to_input` accepts; a single-band
        :class:`~cuperiod.LightCurve` is treated as a one-band set (giving an
        honest bootstrap FAP for the plain GLS as well).
    settings : GLSSettings, optional
        GLS settings; ``mb_model`` selects the calibrated model.
    grid : GridSpec, optional
        Frequency grid; defaults to the method's grid for the stacked bands.
        The false-alarm level of the *maximum* depends on the searched grid, so
        pass the same grid used for the search.
    backend : str, default "auto"
        Backend selector, as :func:`cuperiod.periodogram`.
    n_bootstrap : int, default 500
        Number of within-band resamples (the smallest resolvable FAP is
        ``1 / (n_bootstrap + 1)``).
    seed : int, default 0
        Resampling seed.

    Returns
    -------
    MultibandFAP
        The null sample with ``fap(power)`` and ``level(fap)`` accessors.

    Examples
    --------
    >>> calib = multiband_fap(mblc, n_bootstrap=1000)          # doctest: +SKIP
    >>> pg = cup.periodogram(mblc, "GLS")                      # doctest: +SKIP
    >>> calib.fap(pg.best_periods(1)[0].power)                 # doctest: +SKIP
    >>> calib.level(0.01)  # 1% false-alarm power threshold    # doctest: +SKIP
    """
    from cuperiod.api import to_input
    from cuperiod.methods.gls import GLSMethod

    lc = to_input(data)
    if isinstance(lc, LightCurve):
        lc = MultiBandLightCurve.from_light_curves({"band": lc}, meta=lc.meta)
    method = GLSMethod()
    coerced = method.coerce_settings(settings)
    assert isinstance(coerced, GLSSettings)
    resolved = method.resolve_backend(backend)
    if resolved == "astropy":
        raise ValueError(
            "multiband_fap needs a native backend (finufft/cufinufft/torch)"
        )
    if grid is None:
        from cuperiod.api import _multiband_grid

        grid = _multiband_grid(method, lc, coerced)
    if grid.size == 0:
        raise InsufficientDataError("multiband_fap: empty trial grid")
    null_max = bootstrap_null_max(lc, grid, coerced, resolved, n_bootstrap, seed)
    return MultibandFAP(
        null_max=null_max,
        n_bootstrap=n_bootstrap,
        seed=seed,
        model=coerced.mb_model,
        backend=resolved,
    )


__all__ = ["MultibandFAP", "bootstrap_null_max", "multiband_fap"]
