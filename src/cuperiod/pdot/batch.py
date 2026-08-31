"""Batch period-change measurement: :func:`batch_pdot`.

Runs the whole period-change chain over many light curves on the same worker
machinery as :func:`cuperiod.batch_periodograms` — a spawned CPU process pool,
or GPU workers — and writes **three** tables, because a period-change catalogue
is genuinely three-shaped:

``summary``
    One row per light curve. Every number a survey-scale cut needs: the joint
    ``(P, dP/dt)`` search result, the O-C measurement with its significance, how
    much residual the O-C sheds when the ephemeris is allowed to curve, and the
    best-fitting period law over the measured ``P(t)`` track. Failures get a row
    too, with ``status`` carrying the reason, so nothing disappears silently
    from the catalogue.
``oc``
    One row per O-C segment: the timing points behind each star's ``dP/dt``.
    This is what you plot, refit by hand, or hand to someone who wants to check
    the measurement rather than take it.
``profile``
    One row per interval of the model-free ``P(t)`` track.

The chain per light curve is: optional running-median detrend, then
:func:`cuperiod.pdot_search` to localize, :func:`cuperiod.phase_track` to
measure with honest errors, a polynomial ephemeris ladder to ask whether one
constant rate is enough, and — unless switched off —
:func:`cuperiod.period_profile` plus :func:`cuperiod.compare_period_laws` to
answer the same question without assuming any model at all. Any stage may be
skipped; a stage that fails leaves its columns NaN and does not stop the rest.

Every stage is wrapped so that one pathological star cannot kill a survey-scale
run, and the per-star cost is dominated by the search, so ``chunk_size`` is much
smaller here than in the periodogram batch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cuperiod.batch.io import InputItem, load_source, resolve_inputs, write_rows
from cuperiod.batch.runner import (
    BatchSummary,
    _chunked,
    _classify_sink,
    _dir_manifest_guard,
    _finalize_file,
    _part_path,
    _pending_chunks,
    _run_pool,
)
from cuperiod.batch.sizing import cpu_worker_count, pin_worker_threads
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.pdot.detrend import TrendMethod, detrend
from cuperiod.pdot.laws import AUTO_LAWS, PeriodLawFit, compare_period_laws
from cuperiod.pdot.profile import PeriodProfile, period_profile
from cuperiod.pdot.result import PdotSearchResult
from cuperiod.pdot.track import (
    EphemerisFit,
    PhaseTrackResult,
    _build_ephemeris,
    phase_track,
)

#: Table names produced by a run, in the order they are described above.
TABLES: tuple[str, ...] = ("summary", "oc", "profile")

#: Every parameter name any registered period law can produce. The summary table
#: carries the whole union NaN-filled, so each chunk of a directory sink infers
#: the same Arrow schema whatever law happened to win in it — a part where no
#: star chose ``singular`` would otherwise type ``law_b`` null and make the
#: whole dataset unreadable.
_LAW_PARAMS: tuple[str, ...] = (
    "period", "pdot", "dlnp_dt", "b", "amplitude", "t0", "delta",
)

#: Ephemeris degrees the ladder fits, beyond the linear model.
_LADDER_DEGREES: tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class _ChunkConfig:
    """Immutable, picklable settings shared by every chunk task."""

    settings: PdotSettings
    method: str
    model: str | None
    backend: str
    columns: ColumnMap | None
    domain: Domain | None
    period: float | None
    period_map: Mapping[str, float] | None
    detrend_width: float | None
    detrend_method: TrendMethod
    do_profile: bool
    laws: tuple[str, ...]
    points_per_interval: int | None
    period_window: float | None
    store_oc: bool
    store_profile: bool
    plots_dir: str | None
    plot_keys: frozenset[str] | None


# --- row assembly -------------------------------------------------------------


def _nan_block(names: Sequence[str], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{n}": float("nan") for n in names}


_SEARCH_FIELDS: tuple[str, ...] = (
    "period", "pdot", "pdot_s_per_yr", "pdot_over_p", "power", "zero_pdot_power",
    "timescale_yr", "ridge_correlation", "dlnp_dt_per_yr",
)
_TRACK_FIELDS: tuple[str, ...] = (
    "period", "pdot", "pdot_err", "pdot_err_scaled", "pdot_s_per_yr", "f_test_p",
    "chi2_reduced", "delta_bic", "pdot_snr",
)
_LADDER_FIELDS: tuple[str, ...] = (
    "pddot", "pddot_err", "pdot_cubic", "pdot_cubic_err",
    "rms_min_deg2", "rms_min_deg3", "rms_min_deg4", "curvature_ratio",
)
_PROFILE_FIELDS: tuple[str, ...] = (
    "period_first", "period_last", "total_fractional_change", "period_scatter",
)
_LAW_FIELDS: tuple[str, ...] = (
    "bic", "delta_bic", "chi2_reduced", "rms", "total_fractional_change",
    "timescale_first_yr", "timescale_last_yr",
)


def pdot_to_rows(
    key: str,
    *,
    search: PdotSearchResult | None = None,
    track: PhaseTrackResult | None = None,
    ladder: Sequence[EphemerisFit] = (),
    profile: PeriodProfile | None = None,
    laws: Sequence[PeriodLawFit] = (),
    status: str = "ok",
    n_samples: int = 0,
    baseline: float = float("nan"),
    figure: str = "",
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Flatten one light curve's results into the three catalogue tables.

    Public because the column layout *is* the data product: call it directly to
    reproduce :func:`batch_pdot`'s schema for a light curve analysed by hand.

    Parameters
    ----------
    key : str
        The light curve's batch key.
    search, track, profile : optional
        Stage results; a missing stage leaves its columns NaN.
    ladder : sequence of EphemerisFit
        Polynomial ephemeris fits of increasing degree over the same O-C points.
    laws : sequence of PeriodLawFit
        Period-law fits over the ``P(t)`` track, best (lowest BIC) first.
    status : str, default "ok"
        ``"ok"``, or ``"<ExceptionType>: message"`` for a light curve that
        failed. Failed rows still carry ``key`` and whatever stages completed.
    n_samples, baseline : optional
        Light-curve size and span, recorded even when every stage failed.
    figure : str, default ""
        Path of the diagnostic figure written for this star, if any.

    Returns
    -------
    dict
        ``{"summary": (row,), "oc": (...), "profile": (...)}`` — see
        :data:`TABLES`.
    """
    row: dict[str, Any] = {
        "key": key,
        "status": status,
        "n_samples": int(n_samples),
        "baseline": float(baseline),
        "method": "",
        "backend": "",
        "model": "",
        "t_ref": float("nan"),
        "n_segments": 0,
        "n_harmonics": 0,
        "n_intervals": 0,
        "wrap_risk": False,
        "profile_at_window_edge": False,
        "law": "",
        "law_runner_up": "",
        "implied_model": "",
        "figure": figure,
    }
    row.update(_nan_block(_SEARCH_FIELDS, "search_"))
    row.update(_nan_block(_TRACK_FIELDS, "track_"))
    row.update(_nan_block(_LADDER_FIELDS, "ladder_"))
    row.update(_nan_block(_PROFILE_FIELDS, "profile_"))
    row.update(_nan_block(_LAW_FIELDS, "law_"))
    row.update(_nan_block(_LAW_PARAMS, "law_"))
    row.update(_nan_block(_LAW_PARAMS, "law_err_"))

    oc_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    if search is not None:
        best = search.best
        row.update(
            {
                "method": search.method,
                "backend": search.backend,
                "model": best.model,
                "t_ref": search.t_ref,
                "search_period": best.period,
                "search_pdot": best.pdot,
                "search_pdot_s_per_yr": best.pdot_s_per_yr,
                "search_pdot_over_p": best.pdot_over_p,
                "search_power": best.power,
                "search_zero_pdot_power": best.zero_pdot_power,
                "search_timescale_yr": best.timescale_yr,
                "search_ridge_correlation": best.ridge_correlation,
                "search_dlnp_dt_per_yr": best.dlnp_dt_per_yr,
            }
        )

    if track is not None:
        quad = track.quadratic_fit
        scaled = track.pdot_err_scaled
        row.update(
            {
                "model": track.model,
                "n_segments": track.n_segments,
                "n_harmonics": track.template.n_harmonics,
                "wrap_risk": bool(track.wrap_risk),
                "track_period": track.period,
                "track_pdot": track.pdot,
                "track_pdot_err": track.pdot_err,
                "track_pdot_err_scaled": scaled,
                "track_pdot_s_per_yr": track.pdot_s_per_yr,
                "track_f_test_p": track.f_test_p,
                "track_chi2_reduced": quad.chi2_reduced,
                # Positive favours the quadratic (drifting) model.
                "track_delta_bic": track.linear_fit.bic - quad.bic,
                "track_pdot_snr": (
                    abs(track.pdot) / scaled if scaled > 0.0 else float("nan")
                ),
            }
        )
        oc_rows = [
            {
                "key": key,
                "index": i,
                "t_mid": float(t),
                "delay": float(d),
                "delay_err": float(e),
                "delay_err_scaled": float(e) * math.sqrt(max(1.0, quad.chi2_reduced)),
                "n_points": float(n),
                "period": track.period,
                "pdot": track.pdot,
            }
            for i, (t, d, e, n) in enumerate(
                zip(track.t_mid, track.delay, track.delay_err, track.n_points,
                    strict=True)
            )
        ]

    if ladder:
        by_degree = {f.degree: f for f in ladder}
        cubic = by_degree.get(3)
        if cubic is not None:
            row.update(
                {
                    "ladder_pddot": cubic.pddot,
                    "ladder_pddot_err": cubic.pddot_err,
                    "ladder_pdot_cubic": cubic.pdot,
                    "ladder_pdot_cubic_err": cubic.pdot_err_scaled,
                    "implied_model": cubic.implied_model,
                }
            )
        for degree in _LADDER_DEGREES:
            fit = by_degree.get(degree)
            if fit is not None:
                row[f"ladder_rms_min_deg{degree}"] = _oc_rms_minutes(fit, track)
        # How much residual a cubic sheds over the quadratic. Near 1 the
        # constant-rate model is adequate; well below it, the rate is evolving.
        r2 = row.get("ladder_rms_min_deg2", float("nan"))
        r3 = row.get("ladder_rms_min_deg3", float("nan"))
        if np.isfinite(r2) and np.isfinite(r3) and r2 > 0.0:
            row["ladder_curvature_ratio"] = float(r3 / r2)

    if profile is not None:
        periods = profile.period
        row.update(
            {
                "n_intervals": profile.n_intervals,
                "profile_at_window_edge": profile.any_at_window_edge,
                "profile_period_first": float(periods[0]),
                "profile_period_last": float(periods[-1]),
                "profile_total_fractional_change": profile.total_fractional_change(),
                "profile_period_scatter": float(np.std(periods)),
            }
        )
        profile_rows = [{"key": key, **p.to_dict()} for p in profile.points]

    if laws:
        best_law = laws[0]
        row["law"] = best_law.law
        row["law_bic"] = best_law.bic
        row["law_chi2_reduced"] = best_law.chi2_reduced
        row["law_rms"] = best_law.rms
        if len(laws) > 1:
            row["law_runner_up"] = laws[1].law
            row["law_delta_bic"] = laws[1].bic - best_law.bic
        for name, value in best_law.params.items():
            row[f"law_{name}"] = value
            row[f"law_err_{name}"] = best_law.errors.get(name, float("nan"))
        if profile is not None and profile.n_intervals >= 2:
            t_lo = float(profile.points[0].t_start)
            t_hi = float(profile.points[-1].t_end)
            row["law_total_fractional_change"] = best_law.fractional_change(t_lo, t_hi)
            first, last = best_law.timescale_yr_at([t_lo, t_hi])
            row["law_timescale_first_yr"] = float(first)
            row["law_timescale_last_yr"] = float(last)

    return {
        "summary": (row,),
        "oc": tuple(oc_rows),
        "profile": tuple(profile_rows),
    }


def _oc_rms_minutes(fit: EphemerisFit, track: PhaseTrackResult | None) -> float:
    """RMS of the O-C residuals about ``fit``, in minutes."""
    if track is None or track.t_mid.size == 0:
        return float("nan")
    resid = track.delay - fit.predict(track.t_mid)
    return float(np.sqrt(np.mean(resid * resid)) * 24.0 * 60.0)


# --- the per-light-curve chain ------------------------------------------------


def _ephemeris_ladder(
    track: PhaseTrackResult, degrees: Sequence[int] = _LADDER_DEGREES
) -> list[EphemerisFit]:
    """Polynomial ephemeris fits of increasing degree over one O-C curve.

    A constant ``dP/dt`` is a *quadratic* O-C. If letting the ephemeris curve
    further keeps removing residual, the rate itself is changing and the single
    ``dP/dt`` above is a baseline-weighted average of a moving quantity — the
    failure mode a bare F-test cannot see, because the drift really is
    significant, just not constant.
    """
    w = 1.0 / (track.delay_err * track.delay_err)
    t_ref = float(np.dot(w, track.t_mid) / w.sum())
    dt = track.t_mid - t_ref
    out: list[EphemerisFit] = []
    for degree in degrees:
        if track.t_mid.size <= degree + 1:  # no degrees of freedom left
            continue
        try:
            out.append(
                _build_ephemeris(
                    dt,
                    track.delay,
                    track.delay_err,
                    period_ref=track.period,
                    t_ref=t_ref,
                    degree=degree,
                )
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
    return out


def _analyze_one(
    key: str, lc: LightCurve, cfg: _ChunkConfig
) -> dict[str, tuple[dict[str, Any], ...]]:
    """The whole chain for one light curve; partial results survive a bad stage."""
    finite = lc.finite()
    n_samples, baseline = finite.n, finite.baseline
    work = finite
    if cfg.detrend_width is not None:
        # replace() rather than from_arrays(): the domain (magnitude vs flux)
        # and metadata must survive, and detrending changes only the values.
        work = replace(
            finite,
            value=detrend(
                finite.time,
                finite.value,
                cfg.detrend_width,
                method=cfg.detrend_method,
            ),
        )

    seed = cfg.period
    if cfg.period_map is not None and key in cfg.period_map:
        seed = float(cfg.period_map[key])

    from cuperiod.pdot.scan import pdot_search

    search = pdot_search(
        work,
        period=seed,
        method=cfg.method,
        model=cfg.model,
        settings=cfg.settings,
        backend=cfg.backend,
    )
    track: PhaseTrackResult | None = None
    ladder: list[EphemerisFit] = []
    try:
        track = phase_track(
            work,
            period=search.best.period,
            pdot=search.best.pdot,
            model=cfg.model,
            settings=cfg.settings,
        )
        ladder = _ephemeris_ladder(track)
    except Exception:  # noqa: BLE001 - a star too sparse to track still has a search
        track = None

    profile: PeriodProfile | None = None
    laws: list[PeriodLawFit] = []
    if cfg.do_profile:
        try:
            profile = period_profile(
                work,
                period=search.best.period,
                points_per_interval=cfg.points_per_interval,
                period_window=cfg.period_window,
                settings=cfg.settings,
                backend=cfg.backend,
            )
            if cfg.laws:
                laws = compare_period_laws(
                    profile.t_mid,
                    profile.period,
                    profile.period_err,
                    laws=cfg.laws,
                )
        except Exception:  # noqa: BLE001 - too few intervals is normal, not fatal
            profile = None

    figure = ""
    if cfg.plots_dir is not None and (
        cfg.plot_keys is None or key in cfg.plot_keys
    ):
        figure = _write_figure(key, finite, search, track, profile, laws, cfg)

    return pdot_to_rows(
        key,
        search=search,
        track=track,
        ladder=ladder,
        profile=profile,
        laws=laws,
        n_samples=n_samples,
        baseline=baseline,
        figure=figure,
    )


def _write_figure(
    key: str,
    lc: LightCurve,
    search: PdotSearchResult | None,
    track: PhaseTrackResult | None,
    profile: PeriodProfile | None,
    laws: Sequence[PeriodLawFit],
    cfg: _ChunkConfig,
) -> str:
    """Write one diagnostic figure; a plotting failure never fails the analysis."""
    try:
        from cuperiod.pdot.plots import pdot_figure

        path = Path(cfg.plots_dir or ".") / f"{_safe_name(key)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pdot_figure(
            light_curve=(lc.time, lc.value, lc.error),
            search=search,
            track=track,
            profile=profile,
            law=laws[0] if laws else None,
            title=key,
            path=path,
        )
        return str(path)
    except Exception:  # noqa: BLE001 - a missing matplotlib must not fail a run
        return ""


def _safe_name(key: str) -> str:
    """A filesystem-safe stem for a catalogue key."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in key]
    return ("".join(keep) or "lightcurve")[:120]


def _process_chunk(
    items: Sequence[InputItem], cfg: _ChunkConfig
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, str]]]:
    """Analyse every light curve in a chunk; collect per-table rows and errors."""
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    errors: list[tuple[str, str]] = []
    for key, source in items:
        try:
            lc = load_source(source, columns=cfg.columns, domain=cfg.domain)
            if isinstance(lc, MultiBandLightCurve):
                raise ValueError(
                    "period-change measurement needs single-band light curves"
                )
            produced = _analyze_one(key, lc, cfg)
        except Exception as exc:  # one bad light curve must not kill the batch
            message = f"{type(exc).__name__}: {exc}"
            errors.append((key, message))
            produced = pdot_to_rows(key, status=message)
        for name in TABLES:
            tables[name].extend(produced[name])
    if not cfg.store_oc:
        tables["oc"] = []
    if not cfg.store_profile:
        tables["profile"] = []
    return tables, errors


# --- the driver ---------------------------------------------------------------


def batch_pdot(
    inputs: Any,
    *,
    period: float | Mapping[str, float] | None = None,
    method: str = "GLS",
    model: str | None = None,
    settings: PdotSettings | None = None,
    detrend_width: float | None = None,
    detrend_method: TrendMethod = "median",
    profile: bool = True,
    laws: Sequence[str] | None = AUTO_LAWS,
    points_per_interval: int | None = None,
    period_window: float | None = None,
    backend: str = "auto",
    device: str = "cpu",
    workers: int | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    sink: str | Path | None = None,
    oc_sink: str | Path | None = None,
    profile_sink: str | Path | None = None,
    plots: str | Path | None = None,
    plot_keys: Sequence[str] | None = None,
    resume: bool = True,
    chunk_size: int = 32,
) -> BatchSummary:
    """Measure period changes for many light curves.

    Parameters
    ----------
    inputs : various
        Anything :func:`cuperiod.batch_periodograms` accepts: an iterable of
        light curves or ``(key, lc)`` pairs, a glob string, a directory, or a
        ``(DataFrame, group_column)`` tuple.
    period : float or mapping, optional
        Seed period in days: one value for every star, or a ``{key: period}``
        mapping to feed catalogue periods in per star. ``None`` lets each search
        find its own, which is slower and can lock onto an alias — pass known
        periods when you have them.
    method : str, default "GLS"
        Periodogram wrapped by the joint search. BLS and TLS are not supported.
    model : str, optional
        Period-evolution model for the search warp (:mod:`cuperiod.pdot.models`).
    settings : PdotSettings, optional
        Applied to every stage for every light curve.
    detrend_width : float, optional
        Running-median window in days applied before anything else. Real survey
        photometry usually needs one: a slow brightness trend aliases straight
        onto the 1 c/d comb and can dominate the search. Choose it well above
        the period and well below the trend's timescale. ``None`` disables it.
    detrend_method : {"median", "binned"}, default "median"
        Filter kernel; see :func:`cuperiod.detrend`.
    profile : bool, default True
        Also measure the model-free ``P(t)`` track with
        :func:`cuperiod.period_profile` and fit period laws to it. This is what
        catches a star whose *rate* is changing, which a single ``dP/dt`` cannot
        represent. Switch it off for a faster first pass.
    laws : sequence of str, optional
        Period laws to rank on each track; ``None`` skips the law fitting but
        keeps the track.
    points_per_interval, period_window : optional
        Passed to :func:`cuperiod.period_profile`.
    backend : str, default "auto"
        Backend request for the searches.
    device : {"cpu", "gpu"}, default "cpu"
        Where to run the workers.
    workers : int, optional
        Worker count; ``None`` picks all-but-one core (CPU) or a GPU-sized
        default.
    columns, domain : optional
        Column mapping and brightness-domain handling for file and table inputs.
    sink : str or Path, optional
        Destination for the **summary** table: a ``.parquet``/``.csv`` file, or
        a directory (resumable, one part per chunk). ``None`` returns the rows
        in memory.
    oc_sink, profile_sink : str or Path, optional
        Destinations for the per-segment O-C table and the per-interval ``P(t)``
        table. ``None`` drops them — they are large and are not written unless
        asked for. A directory sink must be a *different* directory from
        ``sink``.
    plots : str or Path, optional
        Directory for per-star diagnostic figures (one PNG per light curve,
        named after the key). Needs matplotlib; a plotting failure is recorded
        as an empty ``figure`` cell and never fails the analysis. Generating
        thousands of figures is slow — pair it with ``plot_keys``.
    plot_keys : sequence of str, optional
        Restrict figures to these keys.
    resume : bool, default True
        Skip chunks already written (directory sinks).
    chunk_size : int, default 32
        Light curves per chunk/task. Much smaller than the periodogram batch's,
        because one light curve here costs a whole grid search.

    Returns
    -------
    BatchSummary
        ``methods`` is reported as ``("PDOT",)``. ``rows`` holds the summary
        table when ``sink is None``; ``n_done`` counts light curves that
        completed the chain, and failures appear both in ``errors`` and as
        summary rows whose ``status`` is not ``"ok"``.

    Examples
    --------
    >>> cup.batch_pdot("lightcurves/*.csv", sink="pdot.parquet",
    ...                oc_sink="oc.parquet")                    # doctest: +SKIP
    >>> cup.batch_pdot(frames, period=known, detrend_width=40.0,
    ...                plots="figures/")                        # doctest: +SKIP
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    if detrend_width is not None and not (
        np.isfinite(detrend_width) and detrend_width > 0.0
    ):
        raise ValueError(
            f"detrend_width must be positive and finite, got {detrend_width}"
        )

    period_value = period if isinstance(period, int | float) else None
    period_map = dict(period) if isinstance(period, Mapping) else None
    cfg = _ChunkConfig(
        settings=settings or PdotSettings(),
        method=method,
        model=model,
        backend="gpu" if device == "gpu" else backend,
        columns=columns,
        domain=domain,
        period=float(period_value) if period_value is not None else None,
        period_map=period_map,
        detrend_width=detrend_width,
        detrend_method=detrend_method,
        do_profile=profile,
        laws=tuple(laws) if laws else (),
        points_per_interval=points_per_interval,
        period_window=period_window,
        store_oc=oc_sink is not None,
        store_profile=profile_sink is not None,
        plots_dir=str(plots) if plots is not None else None,
        plot_keys=frozenset(plot_keys) if plot_keys is not None else None,
    )

    sinks = {
        "summary": _classify_sink(sink),
        "oc": _classify_sink(oc_sink),
        "profile": _classify_sink(profile_sink),
    }
    dirs = [s[1] for s in sinks.values() if s[0] == "dir"]
    if len(dirs) != len({str(d.resolve()) for d in dirs}):
        raise ValueError(
            "sink, oc_sink and profile_sink must be different directories: "
            "part files are named by chunk index only, so sharing a directory "
            "would overwrite one table with another."
        )

    items = resolve_inputs(inputs, columns=columns, domain=domain)
    chunks = _chunked(items, max(1, chunk_size))
    for kind, sink_dir, _ in sinks.values():
        if kind == "dir":
            _dir_manifest_guard(sink_dir, max(1, chunk_size), resume)
    # Resume is driven by whichever table has a directory sink — the summary by
    # preference, since it is the one that always has a row per light curve.
    resume_kind, resume_dir, _ = next(
        (s for s in (sinks[n] for n in TABLES) if s[0] == "dir"), sinks["summary"]
    )
    pending = _pending_chunks(chunks, resume_kind, resume_dir, resume)
    n_skipped = len(items) - sum(len(chunks[i]) for i in pending)

    memory_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    errors: list[tuple[str, str]] = []
    n_done = 0

    def absorb(
        idx: int,
        tables: Mapping[str, list[dict[str, Any]]],
        errs: list[tuple[str, str]],
    ) -> None:
        nonlocal n_done
        n_done += sum(1 for r in tables["summary"] if r.get("status") == "ok")
        errors.extend(errs)
        for name in TABLES:
            kind, sink_dir, _ = sinks[name]
            rows = tables[name]
            if kind == "dir":
                if rows:
                    write_rows(rows, _part_path(sink_dir, idx))
            elif rows:
                memory_rows[name].extend(rows)

    if device == "gpu":
        n_workers = workers if workers is not None else suggest_gpu_workers(method)
        if n_workers <= 1:
            for idx in pending:
                absorb(idx, *_process_chunk(chunks[idx], cfg))
                free_gpu_memory()
        else:
            _run_pool(chunks, pending, cfg, absorb, n_workers, worker=_process_chunk)
    else:
        n_workers = cpu_worker_count(workers)
        if n_workers <= 1:
            for idx in pending:
                absorb(idx, *_process_chunk(chunks[idx], cfg))
        else:
            pin_worker_threads()
            _run_pool(chunks, pending, cfg, absorb, n_workers, worker=_process_chunk)

    for name in TABLES:
        kind, _, sink_file = sinks[name]
        if kind == "file":
            _finalize_file(memory_rows[name], sink_file, resume)

    return BatchSummary(
        n_inputs=len(items),
        n_done=n_done,
        n_failed=len(errors),
        n_skipped=n_skipped,
        methods=("PDOT",),
        device=device,
        sink=str(sink) if sink is not None else None,
        errors=errors,
        rows=(
            memory_rows["summary"] if sinks["summary"][0] == "memory" else None
        ),
    )


__all__ = ["TABLES", "batch_pdot", "pdot_to_rows"]
