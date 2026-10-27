"""Multi-band Box Least Squares.

Bands of the same star share one trial *period*. Two statistics are available
(``BLSSettings.mb_ephemeris``, or the ``ephemeris=`` argument of
:func:`bls_multiband_power`):

``"independent"`` (default)
    Each band is box-searched on its own over the common period grid (its own best
    duration and mid-transit time at every period), and the per-period statistic is
    the quadrature sum of the bands' best depth-SNRs, ``sum_k max(SNR_k, 0)^2`` -- a
    likelihood-ratio stack of independent per-band searches. Only the period is
    shared: a band's box is not forced onto another band's duration or phase, so the
    stack also rewards noise maxima at unrelated phases, and in sparse data it is
    weaker than the joint search. The reported box geometry (depth, duration,
    mid-transit time) at each period comes from the band detecting it most strongly.

``"shared"``
    A transit is achromatic, so one box -- one duration and one mid-transit time on a
    phase grid common to every band -- is placed for all bands at once, each band
    fitting its own depth inside it, and the summed squared depth-SNRs of the bands
    that dip there are maximised over the boxes (see
    :mod:`cuperiod.multiband._bls_joint`). The reported duration and mid-transit time
    are that shared box's. Supported on the in-house backends (``numba``, ``numpy``,
    ``cupy``, ``torch``); astropy's ``BoxLeastSquares`` searches one band at a time
    and is refused.

Either way the power is ``sum_k max(SNR_k, 0)^2``, ``extras["band_index"]`` names the
band with the largest SNR (an index into ``meta["bands"]``), the reported depth is
that band's, in its own flux units, and ``meta["mb_ephemeris"]`` names the statistic.

The stack is only an SNR stack when every band's depth-SNR is dimensionless, so a
band's errors are used only when *every* contributing band has them (the same rule
as the multi-band GLS). Otherwise every band is searched with a constant error equal
to its own robust scatter (1.4826 × the median absolute deviation of its flux), which
keeps the bands commensurate whatever their flux scale or zero point.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import BLSSettings
from cuperiod.core.errors import BackendUnavailableError, InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod

#: A band needs at least this many finite points to contribute a box search.
_MIN_BAND_POINTS = 3

#: Scale factor turning a median absolute deviation into a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826

#: The multi-band box statistics (see the module docstring).
EPHEMERIS_MODES: Final = ("independent", "shared")


def _robust_scatter(flux: FloatArray) -> float:
    """A band's noise scale for when it has no errors: ``1.4826 * MAD`` of its flux.

    The median absolute deviation ignores the few in-eclipse points, so the scale
    tracks the noise rather than the signal. A band whose flux is mostly one repeated
    value (MAD 0) falls back to the standard deviation, and a constant band to 1 — it
    has no box to detect, so its scale is immaterial.
    """
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(flux - np.median(flux))))
    if not (np.isfinite(sigma) and sigma > 0.0):
        sigma = float(np.std(flux))
    if not (np.isfinite(sigma) and sigma > 0.0):
        sigma = 1.0
    return sigma


def _band_errors(bands: list[LightCurve]) -> list[FloatArray]:
    """Per-band flux errors that make every band's depth-SNR dimensionless.

    The quadrature stack weights a band by ``SNR^2``, so a band searched with unit
    errors would count by its raw flux scale squared (``10**(0.8 * dmag)``) and a
    bright, noisy band would outvote a faint, precise one. Real errors are used when
    every band has them; otherwise all bands get their own constant robust scatter —
    mixing real errors with a scatter estimate would weigh the two kinds of band
    against each other on different footings (the multi-band GLS applies the same
    all-or-nothing rule).
    """
    if all(lc.error is not None for lc in bands):
        return [np.asarray(lc.error, dtype=np.float64) for lc in bands]
    return [np.full(lc.n, _robust_scatter(lc.value)) for lc in bands]


def resolve_ephemeris(settings: Any, ephemeris: str | None = None) -> str:
    """The multi-band box statistic to run: the argument, else the settings' field.

    ``BLSSettings.mb_ephemeris`` when the settings carry one, else ``"independent"``.

    Raises
    ------
    ValueError
        For a name outside :data:`EPHEMERIS_MODES`.
    """
    if ephemeris is not None:
        mode = ephemeris
    elif isinstance(settings, BLSSettings):
        mode = settings.mb_ephemeris
    else:  # a settings-like object from before the field existed
        mode = "independent"
    mode = str(mode).strip().lower()
    if mode not in EPHEMERIS_MODES:
        raise ValueError(
            f"BLS multiband: unknown ephemeris {mode!r}; choose from "
            f"{EPHEMERIS_MODES}"
        )
    return mode


def bls_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: BLSSettings,
    backend: str,
    method: PeriodogramMethod,
    *,
    ephemeris: str | None = None,
) -> Periodogram:
    """Compute the multi-band BLS statistic on a shared period grid.

    Parameters
    ----------
    grid : GridSpec
        Trial grid, searched exactly as given and segmented as single-band BLS
        segments it (:func:`~cuperiod.methods.bls._grid_segments`): the default
        grid at its recorded log segments, any other grid into log segments whose
        box durations track the period.
    mblc : MultiBandLightCurve
        Two or more bands of one star; a band contributes when it has at least 3
        finite points.
    settings : BLSSettings
        BLS settings.
    backend : str
        Concrete backend (``"numba"``, ``"numpy"``, ``"astropy"``, ``"cupy"`` or
        ``"torch:<device>"``).
    method : PeriodogramMethod
        The owning :class:`~cuperiod.methods.bls.BLSMethod` (unused; reserved).
    ephemeris : {"independent", "shared"}, optional
        The box statistic (see the module docstring). ``None`` reads
        ``settings.mb_ephemeris`` when the settings have that field, else
        ``"independent"``.

    Returns
    -------
    Periodogram
        The per-period sum of the bands' squared depth-SNRs, with the box geometry
        of the strongest band (``"independent"``) or of the shared box
        (``"shared"``); ``meta["mb_ephemeris"]`` records which.
        ``extras["band_index"]`` indexes ``meta["bands"]``, the contributing bands.

    Raises
    ------
    InsufficientDataError
        If no band has 3 finite points, the contributing bands hold fewer than
        ``min_detections`` points, the grid is empty, or the baseline is empty.
    BackendUnavailableError
        For ``ephemeris="shared"`` on the ``astropy`` backend.
    ValueError
        For an unknown ``ephemeris``.
    """
    from cuperiod.methods.bls import _grid_segments, _segments_power

    mode = resolve_ephemeris(settings, ephemeris)
    if mode == "shared" and backend == "astropy":
        raise BackendUnavailableError(
            "BLS multiband: ephemeris='shared' needs the in-house box search, and "
            "the 'astropy' backend (what 'auto'/'cpu' pick when numba is not "
            "installed) searches one band at a time; use backend='numba', 'numpy', "
            "'cupy' or 'torch', or install the [fast] extra"
        )
    finite = mblc.finite()
    used = [
        (name, lc.in_domain(Domain.FLUX))
        for name, lc in finite.bands.items()
        if lc.n >= _MIN_BAND_POINTS
    ]
    if not used:
        raise InsufficientDataError("BLS multiband: no band has enough finite points")
    bands = [lc for _, lc in used]
    n_total = sum(lc.n for lc in bands)
    if n_total < settings.min_detections:
        raise InsufficientDataError(
            f"BLS multiband: {n_total} finite points across usable bands "
            f"< min_detections {settings.min_detections}"
        )
    stacked_time = np.concatenate([lc.time for lc in bands])
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("BLS multiband: no usable time baseline")
    segments = _grid_segments(grid, settings)
    if not segments:
        raise InsufficientDataError("BLS multiband: empty trial grid")
    errors = _band_errors(bands)
    # The statistic is recorded either way, so ``meta["mb_ephemeris"]`` always says
    # which one the result holds.
    meta: dict[str, Any] = {
        **dict(finite.meta),
        "bands": tuple(name for name, _ in used),
        "mb_ephemeris": mode,
    }
    if mode == "shared":
        period, power, extras = _shared(bands, errors, segments, settings, backend)
    else:
        period, power, extras = _independent(
            bands, errors, segments, settings, backend, _segments_power
        )
    power = np.nan_to_num(power, nan=0.0, posinf=0.0)
    mean, std = float(power.mean()), float(power.std())
    sde = (power - mean) / std if std > 0.0 else np.zeros_like(power)
    return Periodogram.from_spectrum(
        method="BLS",
        backend=backend,
        frequency=1.0 / period,
        power=power,
        objective_sense="max",
        n_samples=n_total,
        baseline=baseline,
        extras={
            "depth": extras["depth"],
            "depth_snr": extras["depth_snr"],
            "duration": extras["duration"],
            "t0": extras["t0"],
            "sde": sde,
            "band_index": extras["band_index"],
        },
        meta=meta,
    )


def _independent(
    bands: list[LightCurve],
    errors: list[FloatArray],
    segments: list[tuple[FloatArray, FloatArray]],
    settings: BLSSettings,
    backend: str,
    segments_power: Any,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """The independent per-band stack: ``(period, power, extras)``."""
    per_band = [
        segments_power(backend, lc.time, lc.value, err, segments, settings)
        for lc, err in zip(bands, errors, strict=True)
    ]
    parts: dict[str, list[FloatArray]] = {
        name: [] for name in ("period", "power", "depth", "depth_snr", "duration",
                              "t0", "band_index")
    }
    for s, (periods, _) in enumerate(segments):
        seg = [band_segments[s] for band_segments in per_band]
        dsnr = np.vstack([np.clip(b["depth_snr"], 0.0, None) for b in seg])
        depth = np.vstack([b["depth"] for b in seg])
        dur = np.vstack([b["duration"] for b in seg])
        t0 = np.vstack([b["transit_time"] for b in seg])
        combined = np.sum(dsnr**2, axis=0)
        best = np.argmax(dsnr, axis=0)
        cols = np.arange(dsnr.shape[1])
        parts["period"].append(periods)
        parts["power"].append(combined)
        parts["depth_snr"].append(np.sqrt(combined))
        parts["depth"].append(depth[best, cols])
        parts["duration"].append(dur[best, cols])
        parts["t0"].append(t0[best, cols])
        parts["band_index"].append(best.astype(np.float64))
    joined = {name: np.concatenate(values) for name, values in parts.items()}
    period, power = joined.pop("period"), joined.pop("power")
    return period, power, joined


def _shared(
    bands: list[LightCurve],
    errors: list[FloatArray],
    segments: list[tuple[FloatArray, FloatArray]],
    settings: BLSSettings,
    backend: str,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """The joint-ephemeris search: ``(period, power, extras)``."""
    from cuperiod.multiband._bls_joint import joint_bands, joint_search

    prepared = joint_bands(
        [lc.time for lc in bands], [lc.value for lc in bands], errors
    )
    found = joint_search(
        prepared,
        segments,
        settings.bins_per_duration,
        objective=settings.objective,
        backend=backend,
        batch=settings.batch_periods,
        precision=settings.precision,
    )
    period = np.concatenate([np.asarray(p, dtype=np.float64) for p, _ in segments])

    def joined(name: str) -> FloatArray:
        return np.concatenate([f[name] for f in found])

    extras = {
        "depth": joined("depth"),
        "depth_snr": joined("depth_snr"),
        "duration": joined("duration"),
        "t0": joined("transit_time"),
        "band_index": joined("band_index"),
    }
    return period, joined("power"), extras


__all__ = ["EPHEMERIS_MODES", "bls_multiband_power", "resolve_ephemeris"]
