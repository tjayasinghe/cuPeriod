"""Transit Least Squares (limb-darkened matched-filter transit search).

A from-scratch search in the spirit of TLS (Hippke & Heller 2019): instead of BLS's
rectangular box it correlates the folded light curve with a realistic *limb-darkened*
transit template, which matches real transit/eclipse shapes far better and so improves
the signal detection efficiency (SDE) for shallow signals.

For each trial period the (flux) light curve is phase-folded and binned; the template —
a quadratic limb-darkening profile for a central transit, resampled to each trial
duration — is slid across phase, and at every position the matched-filter depth and its
signal residue ``SR = (sum w g y')^2 / sum w g^2`` are evaluated in one batched
correlation. The per-period maximum SR is normalized across the grid into the SDE, which
is *maximized* at the transit period. Inputs are flux (a transit is a dip); magnitudes
are converted automatically.

This implementation is CPU (numpy) and vectorizes over the period grid; a GPU kernel and
the Ofir (2014) optimal-frequency grid are planned refinements.
"""

from __future__ import annotations

from typing import ClassVar, Final, Literal

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import TLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

#: Smallest admissible per-bin weight (an empty side is skipped, not divided by).
_W_EPS: Final = 1e-300


def limb_darkened_template(n: int, u1: float, u2: float) -> FloatArray:
    """Normalized central-transit flux-deficit profile over ``n`` samples.

    Quadratic limb darkening ``I(mu) = 1 - u1(1-mu) - u2(1-mu)^2`` sampled along the
    central chord; the profile is 1 at mid-transit and tapers toward the limb. Used as
    the matched-filter template (peak-normalized to 1).

    Parameters
    ----------
    n : int
        Number of template samples (transit duration in phase bins).
    u1, u2 : float
        Quadratic limb-darkening coefficients.

    Returns
    -------
    numpy.ndarray
        The deficit profile of length ``n``.
    """
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    x = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5  # bin centers in [-0.5, 0.5)
    z = np.clip(2.0 * np.abs(x), 0.0, 1.0)  # central-transit projected separation
    mu = np.sqrt(1.0 - z * z)
    g = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    g = np.where(z < 1.0, g, 0.0)
    peak = float(g.max())
    return g / peak if peak > 0.0 else g


def _duration_bins(settings: TLSSettings) -> list[int]:
    """Transit widths in phase bins (deduped, ascending)."""
    n_bins = settings.n_phase_bins
    fracs = np.geomspace(
        settings.duration_min_frac, settings.duration_max_frac, settings.n_durations
    )
    widths = sorted({int(round(f * n_bins)) for f in fracs})
    return [w for w in widths if 1 <= w < n_bins]


def _period_grid(baseline: float, settings: TLSSettings) -> FloatArray:
    """Ascending trial periods from a frequency grid capped by the baseline."""
    max_period = min(settings.max_period_days, baseline / settings.min_transits)
    if max_period <= settings.min_period_days:
        return np.zeros(0, dtype=np.float64)
    # Spacing tied to the transit width so TLS's narrow peaks are well sampled.
    df = settings.grid_duration_frac / (settings.oversample * baseline)
    freq = np.arange(1.0 / max_period, 1.0 / settings.min_period_days, df)
    if freq.size == 0:
        return np.zeros(0, dtype=np.float64)
    return (1.0 / freq[::-1]).copy()


def _matched_filter(
    tau: FloatArray,
    yw: FloatArray,
    w: FloatArray,
    periods: FloatArray,
    *,
    n_bins: int,
    dur_bins: list[int],
    templates: dict[int, FloatArray],
    period_batch: int,
) -> dict[str, FloatArray]:
    """Per-period best matched-filter signal residue and transit parameters."""
    n_periods = int(periods.size)
    out = {
        "sr": np.zeros(n_periods),
        "depth": np.zeros(n_periods),
        "duration": np.zeros(n_periods),
        "t0": np.zeros(n_periods),
    }
    n_points = int(tau.size)
    for start in range(0, n_periods, period_batch):
        stop = min(start + period_batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.size)
        phase = np.mod(tau[None, :] / pb[:, None], 1.0)
        bin_idx = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
        rows = np.repeat(np.arange(n_p), n_points)
        flat = (rows * n_bins + bin_idx.ravel()).astype(np.int64)
        a_flat = np.zeros(n_p * n_bins)  # sum w*y' per bin
        b_flat = np.zeros(n_p * n_bins)  # sum w per bin
        np.add.at(a_flat, flat, np.broadcast_to(yw, (n_p, n_points)).ravel())
        np.add.at(b_flat, flat, np.broadcast_to(w, (n_p, n_points)).ravel())
        a = a_flat.reshape(n_p, n_bins)
        b = b_flat.reshape(n_p, n_bins)

        best_sr = np.zeros(n_p)
        best_depth = np.zeros(n_p)
        best_start = np.zeros(n_p, dtype=np.int64)
        best_width = np.zeros(n_p, dtype=np.int64)
        for width in dur_bins:
            g = templates[width]
            g2 = g * g
            a_ext = np.concatenate([a, a[:, : width - 1]], axis=1)
            b_ext = np.concatenate([b, b[:, : width - 1]], axis=1)
            win_a = np.lib.stride_tricks.sliding_window_view(a_ext, width, axis=1)
            win_b = np.lib.stride_tricks.sliding_window_view(b_ext, width, axis=1)
            num = np.einsum("psk,k->ps", win_a, g)
            den = np.einsum("psk,k->ps", win_b, g2)
            safe_den = np.where(den > _W_EPS, den, 1.0)
            sr = np.where(den > _W_EPS, num * num / safe_den, 0.0)
            # a dip means the in-transit weighted residual is negative -> num < 0
            sr = np.where(num < 0.0, sr, 0.0)
            s_best = np.argmax(sr, axis=1)
            rows_p = np.arange(n_p)
            sr_best = sr[rows_p, s_best]
            improve = sr_best > best_sr
            best_sr = np.where(improve, sr_best, best_sr)
            depth = -num[rows_p, s_best] / np.where(
                den[rows_p, s_best] > _W_EPS, den[rows_p, s_best], 1.0
            )
            best_depth = np.where(improve, depth, best_depth)
            best_start = np.where(improve, s_best, best_start)
            best_width = np.where(improve, width, best_width)

        sl = slice(start, stop)
        out["sr"][sl] = best_sr
        out["depth"][sl] = best_depth
        out["duration"][sl] = best_width.astype(np.float64) / n_bins * pb
        centre = (best_start.astype(np.float64) + best_width / 2.0) / n_bins
        out["t0"][sl] = np.mod(centre, 1.0) * pb
    return out


def tls_power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray | None,
    periods: FloatArray,
    *,
    settings: TLSSettings,
) -> dict[str, FloatArray]:
    """TLS matched-filter search over ``periods`` (flux input; a transit is a dip).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and flux of one band.
    dy : numpy.ndarray or None
        Flux errors (inverse-variance weights); ``None`` for uniform weights.
    periods : numpy.ndarray
        Trial periods (days).
    settings : TLSSettings
        Search configuration.

    Returns
    -------
    dict of numpy.ndarray
        ``sde`` (per-period detection efficiency) plus ``sr``/``depth``/``duration``/
        ``t0`` aligned with ``periods``.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    n_periods = int(periods_host.size)
    if n_periods == 0:
        empty = np.zeros(0)
        return {k: empty for k in ("sde", "sr", "depth", "duration", "t0")}

    w = (
        np.ones_like(y)
        if dy is None
        else 1.0 / np.ascontiguousarray(dy, dtype=np.float64) ** 2
    )
    y_mean = float(np.dot(w, y) / w.sum())
    yw = w * (y - y_mean)  # weighted, mean-subtracted
    tau = t - t.min()

    dur_bins = _duration_bins(settings)
    u1, u2 = settings.limb_dark_u1, settings.limb_dark_u2
    templates = {width: limb_darkened_template(width, u1, u2) for width in dur_bins}
    if not dur_bins:
        z = np.zeros(n_periods)
        return {"sde": z, "sr": z, "depth": z, "duration": z, "t0": z}

    res = _matched_filter(
        tau, yw, w, periods_host,
        n_bins=settings.n_phase_bins, dur_bins=dur_bins, templates=templates,
        period_batch=settings.period_batch,
    )
    sr = res["sr"]
    mean, std = float(sr.mean()), float(sr.std())
    sde = (sr - mean) / std if std > 0.0 else np.zeros_like(sr)
    res["sde"] = sde
    return res


class TLSMethod(PeriodogramMethod):
    """Transit Least Squares — limb-darkened matched filter (numpy CPU)."""

    name: ClassVar[str] = "TLS"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = False
    natural_domain: ClassVar[Domain] = Domain.FLUX
    settings_cls: ClassVar[type] = TLSSettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = None
    all_backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def default_grid(self, lc: LightCurve, settings: TLSSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        periods = _period_grid(finite.baseline, settings)
        if periods.size == 0:
            raise InsufficientDataError(
                "TLS: baseline too short for the configured period range"
            )
        return GridSpec(kind="period", values=periods, uniform=False)

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: TLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"TLS: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("TLS: no usable time baseline")
        periods = grid.period
        res = tls_power(
            finite.time, finite.value, finite.error, periods, settings=settings
        )
        extras = {
            "sr": res["sr"],
            "depth": res["depth"],
            "duration": res["duration"],
            "t0": res["t0"],
        }
        return Periodogram.from_spectrum(
            method="TLS",
            backend=backend,
            frequency=1.0 / periods,
            power=res["sde"],
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            extras=extras,
            meta=finite.meta,
        )


register(TLSMethod())

__all__ = ["TLSMethod", "limb_darkened_template", "tls_power"]
