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

Resume works per light curve and across all three tables at once: a star is
skipped only when its summary row is ``"ok"`` for the same method *and* every
requested O-C / ``P(t)`` table holds exactly the rows that summary row promises
(``n_segments`` / ``n_intervals``). A star recomputed by a later run replaces its
rows in every table, so an interrupted run, a deleted part file, a sink added
later or a star that failed before all heal on the next call, whatever mix of
file and directory sinks is in use.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, get_args

import numpy as np

from cuperiod.batch.io import (
    InputItem,
    csv_safe,
    load_source,
    resolve_inputs,
    source_label,
    write_rows,
)
from cuperiod.batch.runner import (
    BatchSummary,
    _chunked,
    _classify_sink,
    _dir_manifest_guard,
    _part_path,
    _read_existing_rows,
    _run_pool,
    _source_clash,
    _warn_dead_batch_env,
    _warn_source_clashes,
)
from cuperiod.batch.sizing import (
    cpu_worker_count,
    init_worker_threads,
    numba_threads_per_worker,
    pinned_worker_threads,
)
from cuperiod.core._coerce import as_float_array
from cuperiod.core.backend import cpu_pool_backend
from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.config import PdotSettings
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.methods.base import get_method
from cuperiod.pdot.detrend import TrendMethod, detrend
from cuperiod.pdot.laws import (
    AUTO_LAWS,
    PeriodLawFit,
    available_laws,
    compare_period_laws,
)
from cuperiod.pdot.ltte import LTTESearchResult, ltte_search
from cuperiod.pdot.models import get_model
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

#: The summary column counting each star's rows in a long table: the resume check
#: compares it with the rows actually stored there.
_ROW_COUNT_FIELD: dict[str, str] = {"oc": "n_segments", "profile": "n_intervals"}

#: Directory parts are written long tables first, summary last, so a summary part
#: on disk means its chunk's O-C and ``P(t)`` parts landed before it.
_WRITE_ORDER: tuple[str, ...] = ("oc", "profile", "summary")

#: Tasks per worker the CPU pool aims for when chunks are free to shrink (no
#: directory sink pins their layout): enough that one slow star does not leave the
#: other workers idle at the end, few enough that per-task overhead stays small.
_TASKS_PER_WORKER = 4


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
    do_ltte: bool
    laws: tuple[str, ...]
    points_per_interval: int | None
    period_window: float | None
    store_oc: bool
    store_profile: bool
    plots_dir: str | None
    plot_keys: frozenset[str] | None
    #: Figure stems overriding ``_safe_name`` (letter-case clashes; see
    #: ``_figure_stems``). Usually empty, so chunk tasks stay small.
    figure_stems: Mapping[str, str] = field(default_factory=dict)


# --- row assembly -------------------------------------------------------------


def _nan_block(names: Sequence[str], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{n}": float("nan") for n in names}


def _law_margin(best: PeriodLawFit, runner_up: PeriodLawFit) -> float:
    """The winning law's BIC margin, on the scale the laws were ranked by.

    :func:`~cuperiod.compare_period_laws` ranks on ``bic_scaled`` (chi-squared
    rescaled by the excess scatter), so the margin must too: a plain ``bic``
    difference can come out negative for a lower-ranked law, or ``k**2`` too
    large when the timing errors are understated ``k``-fold. Fits that never went
    through the comparison carry no ``bic_scaled`` and fall back to ``bic``.
    """
    if np.isfinite(best.bic_scaled) and np.isfinite(runner_up.bic_scaled):
        return float(runner_up.bic_scaled - best.bic_scaled)
    return float(runner_up.bic - best.bic)


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
    "bic", "bic_scaled", "delta_bic", "chi2_reduced", "rms",
    "total_fractional_change", "timescale_first_yr", "timescale_last_yr",
)
_LTTE_FIELDS: tuple[str, ...] = (
    "outer_period_yr", "outer_period_err_yr", "amplitude_s", "amplitude_err_s",
    "delta_bic", "f_test_p", "n_cycles", "mass_function", "pdot", "pdot_err",
    "pdot_s_per_yr", "pdot_change", "chi2_reduced",
)


def pdot_to_rows(
    key: str,
    *,
    search: PdotSearchResult | None = None,
    track: PhaseTrackResult | None = None,
    ladder: Sequence[EphemerisFit] = (),
    profile: PeriodProfile | None = None,
    laws: Sequence[PeriodLawFit] = (),
    ltte: LTTESearchResult | None = None,
    status: str = "ok",
    n_samples: int = 0,
    baseline: float = float("nan"),
    figure: str = "",
    source: str = "",
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
        The cubic's ``ladder_pdot_cubic_err`` and ``ladder_pddot_err`` are both
        inflated by ``sqrt(chi2_reduced)`` when that exceeds 1.
    laws : sequence of PeriodLawFit
        Period-law fits over the ``P(t)`` track, best (lowest BIC) first.
    ltte : LTTESearchResult, optional
        The cyclic (light-travel-time) O-C search over the track's delays.
    status : str, default "ok"
        ``"ok"``, or ``"<ExceptionType>: message"`` for a light curve that
        failed. Failed rows still carry ``key`` and whatever stages completed.
    n_samples, baseline : optional
        Light-curve size and span, recorded even when every stage failed.
    figure : str, default ""
        Path of the diagnostic figure written for this star, if any.
    source : str, default ""
        Provenance of the light curve: its file's absolute path, or ``""`` for an
        in-memory one (:func:`cuperiod.batch.io.source_label`). A resumed sink uses
        it to tell a same-named light curve from another folder apart from the one
        it holds.

    Returns
    -------
    dict
        ``{"summary": (row,), "oc": (...), "profile": (...)}`` — see
        :data:`TABLES`.
    """
    row: dict[str, Any] = {
        "key": key,
        "source": source,
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
        "law_at_bound": False,
        "implied_model": "",
        "ltte_at_grid_edge": False,
        "figure": figure,
    }
    row.update(_nan_block(_SEARCH_FIELDS, "search_"))
    row.update(_nan_block(_TRACK_FIELDS, "track_"))
    row.update(_nan_block(_LADDER_FIELDS, "ladder_"))
    row.update(_nan_block(_PROFILE_FIELDS, "profile_"))
    row.update(_nan_block(_LAW_FIELDS, "law_"))
    row.update(_nan_block(_LAW_PARAMS, "law_"))
    row.update(_nan_block(_LAW_PARAMS, "law_err_"))
    row.update(_nan_block(_LTTE_FIELDS, "ltte_"))

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
            # Both ladder errors are inflated by sqrt(chi2_reduced) when the cubic
            # fits worse than its errors say, like every other *_err column of the
            # summary; a formal pddot_err beside a scaled pdot_cubic_err overstated
            # the significance of an evolving rate by exactly that factor.
            row.update(
                {
                    "ladder_pddot": cubic.pddot,
                    "ladder_pddot_err": cubic.pddot_err_scaled,
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
        row["law_at_bound"] = bool(best_law.at_bound)
        row["law_bic"] = best_law.bic
        row["law_bic_scaled"] = best_law.bic_scaled
        row["law_chi2_reduced"] = best_law.chi2_reduced
        row["law_rms"] = best_law.rms
        if len(laws) > 1:
            row["law_runner_up"] = laws[1].law
            row["law_delta_bic"] = _law_margin(best_law, laws[1])
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

    if ltte is not None:
        cyc = ltte.best
        row.update(
            {
                "ltte_outer_period_yr": cyc.outer_period_yr,
                "ltte_outer_period_err_yr": cyc.outer_period_err_yr,
                "ltte_amplitude_s": cyc.amplitude_s,
                "ltte_amplitude_err_s": cyc.amplitude_err_s,
                "ltte_delta_bic": ltte.delta_bic,
                "ltte_f_test_p": ltte.f_test_p,
                "ltte_n_cycles": ltte.n_cycles,
                "ltte_mass_function": cyc.mass_function,
                # The secular rate with the cycle removed, and how far the
                # parabola-only rate sat from it.
                "ltte_pdot": cyc.pdot,
                "ltte_pdot_err": cyc.pdot_err_scaled,
                "ltte_pdot_s_per_yr": cyc.pdot_s_per_yr,
                "ltte_pdot_change": ltte.pdot_change,
                "ltte_chi2_reduced": cyc.chi2_reduced,
                "ltte_at_grid_edge": bool(cyc.at_grid_edge),
            }
        )

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

    # The cyclic O-C: a third body's light-travel time, or a magnetic cycle.
    # Without it a parabola averages over any cycle shorter than the baseline.
    cyclic: LTTESearchResult | None = None
    if cfg.do_ltte and track is not None:
        try:
            cyclic = ltte_search(track, settings=cfg.settings)
        except Exception:  # noqa: BLE001 - too few segments is normal, not fatal
            cyclic = None

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
        figure = _write_figure(
            key, finite, search, track, profile, laws, cyclic, cfg
        )

    return pdot_to_rows(
        key,
        search=search,
        track=track,
        ladder=ladder,
        profile=profile,
        laws=laws,
        ltte=cyclic,
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
    cyclic: LTTESearchResult | None,
    cfg: _ChunkConfig,
) -> str:
    """Write one diagnostic figure; a plotting failure never fails the analysis."""
    try:
        from cuperiod.pdot.plots import pdot_figure

        stem = cfg.figure_stems.get(key) or _safe_name(key)
        path = Path(cfg.plots_dir or ".") / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pdot_figure(
            # The LightCurve itself, not a bare (t, y, dy) tuple: the figure reads
            # the brightness domain from it, and a tuple is drawn as magnitudes —
            # every flux survey's figure came out inverted and labelled 'magnitude'.
            light_curve=lc,
            search=search,
            track=track,
            profile=profile,
            law=laws[0] if laws else None,
            ltte=cyclic.best if cyclic is not None else None,
            title=key,
            path=path,
        )
        return str(path)
    except Exception:  # noqa: BLE001 - a missing matplotlib must not fail a run
        return ""


#: Longest figure stem, digest included; well inside every file system's limit.
_STEM_MAX = 120

#: Device names Windows reserves whatever the extension ("con.png" opens CON).
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _sanitized(key: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)


def _digested_name(key: str) -> str:
    """The sanitized key, cut to fit, with an 8-hex digest of the key appended."""
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{(_sanitized(key) or 'lightcurve')[: _STEM_MAX - 9]}-{digest}"


def _safe_name(key: str) -> str:
    """A filesystem-safe stem for a catalogue key, distinct for distinct keys.

    A key made only of ``[alnum-_.]`` is its own stem, so figures stay findable by
    name. Any other key gets its characters replaced *and* an 8-hex digest of the
    key appended: a path-qualified key such as ``"fieldA/lc_0000"`` would
    otherwise alias a real key ``"fieldA_lc_0000"`` (as ``"a b"`` / ``"a:b"`` /
    ``"a_b"`` did), and one star's figure silently overwrote another's. A key cut
    to fit, or naming a Windows device, is digested for the same reason. Keys that
    differ only in letter case are told apart per run (:func:`_figure_stems`).
    """
    stem = _sanitized(key)
    if (
        stem == key
        and 0 < len(stem) <= _STEM_MAX
        and stem.split(".", 1)[0].lower() not in _RESERVED_STEMS
    ):
        return stem
    return _digested_name(key)


def _figure_stems(keys: Iterable[str]) -> dict[str, str]:
    """Figure stems, for the ``keys`` whose :func:`_safe_name` clash in letter case.

    ``"V1"`` and ``"v1"`` are distinct keys and safe names, but one file on a
    case-insensitive file system (Windows, macOS): the second star's figure
    overwrote the first's. Those keys get the digested stem; every other key keeps
    :func:`_safe_name` (and is absent from the returned mapping, which is usually
    empty).
    """
    plain = {key: _safe_name(key) for key in dict.fromkeys(keys)}
    folded = Counter(stem.casefold() for stem in plain.values())
    return {
        key: _digested_name(key)
        for key, stem in plain.items()
        if folded[stem.casefold()] > 1
    }


def _process_chunk(
    items: Sequence[InputItem], cfg: _ChunkConfig
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, str]]]:
    """Analyse every light curve in a chunk; collect per-table rows and errors."""
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    errors: list[tuple[str, str]] = []
    for key, source in items:
        # Size and span are known as soon as the curve loads, and are recorded on
        # a failed row too, so failures can be triaged by size from the catalogue.
        n_samples, baseline = 0, float("nan")
        try:
            lc = load_source(source, columns=cfg.columns, domain=cfg.domain)
            if isinstance(lc, MultiBandLightCurve):
                raise ValueError(
                    "period-change measurement needs single-band light curves"
                )
            finite = lc.finite()
            n_samples, baseline = finite.n, finite.baseline
            produced = _analyze_one(key, lc, cfg)
        except Exception as exc:  # one bad light curve must not kill the batch
            message = f"{type(exc).__name__}: {exc}"
            errors.append((key, message))
            produced = pdot_to_rows(
                key, status=message, n_samples=n_samples, baseline=baseline
            )
        # Every summary row names its file, so a resumed sink can tell a same-named
        # light curve from another folder apart from the one it holds.
        label = source_label(source)
        tables["summary"].extend(
            {**row, "source": label} for row in produced["summary"]
        )
        for name in TABLES[1:]:
            tables[name].extend(produced[name])
    if not cfg.store_oc:
        tables["oc"] = []
    if not cfg.store_profile:
        tables["profile"] = []
    return tables, errors


# --- argument normalisation ---------------------------------------------------


def _seed_periods(period: Any) -> tuple[float | None, dict[str, float] | None]:
    """Split ``period=`` into one seed for every star or a ``{key: seed}`` map.

    Any real number counts (NumPy scalars and 0-d arrays included: they used to
    fall through both ``isinstance`` checks and silently run every star blind), and
    anything with ``.items()`` is a catalogue (a ``dict`` or a pandas ``Series``),
    its keys stringified as the batch's own keys are. A catalogue entry that is
    missing, NaN or not positive means "no catalogue period": that star runs the
    blind search, as it would without an entry.
    """
    if period is None:
        return None, None
    if isinstance(period, np.ndarray) and period.ndim == 0:
        period = period.item()
    if isinstance(period, bool | np.bool_ | str | bytes):
        raise TypeError(
            "period must be a number of days or a {key: period} mapping, got "
            f"{type(period).__name__} {period!r}"
        )
    if isinstance(period, numbers.Real):
        value = float(period)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"period must be positive and finite, got {value}")
        return value, None
    items = getattr(period, "items", None)
    if not callable(items):
        raise TypeError(
            "period must be a number of days or a {key: period} mapping (dict or "
            f"pandas Series), got {type(period).__name__}"
        )
    if hasattr(period, "index") and hasattr(period, "to_numpy"):  # pandas Series
        keys: list[Any] = list(period.index)
        values = as_float_array(period, name="period")
    else:
        pairs = list(items())
        keys = [k for k, _ in pairs]
        values = as_float_array([v for _, v in pairs], name="period")
    return None, {
        str(k): float(v)
        for k, v in zip(keys, values, strict=True)
        if math.isfinite(v) and v > 0.0
    }


def _law_names(laws: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalise ``laws=`` and reject unknown names before the run starts.

    A bare string is one law (``tuple("singular")`` iterated it per character), and
    an unknown name must raise here: :func:`compare_period_laws` skips names it
    cannot fit, so a typo used to leave a whole catalogue without a fitted law.
    """
    if laws is None:
        return ()
    names = (laws,) if isinstance(laws, str) else tuple(laws)
    normalised = tuple(str(name).strip().lower() for name in names)
    known = available_laws()
    unknown = [name for name, low in zip(names, normalised, strict=True)
               if low not in known]
    if unknown:
        raise ValueError(
            f"unknown period law(s) {unknown}; choose from {list(known)}"
        )
    return normalised


def _key_set(keys: str | Iterable[Any] | None) -> frozenset[str] | None:
    """``plot_keys=`` as batch keys (strings; a bare string is one key)."""
    if keys is None:
        return None
    if isinstance(keys, str):
        return frozenset({keys})
    return frozenset(str(key) for key in keys)


def _pool_chunk_size(n_items: int, chunk_size: int, n_workers: int) -> int:
    """Light curves per task for a CPU pool whose chunk layout is not pinned.

    One task per chunk means ``ceil(n / chunk_size)`` tasks: a batch smaller than
    ``chunk_size`` ran on a single worker however many were asked for. Aim for
    :data:`_TASKS_PER_WORKER` tasks per worker instead, never above ``chunk_size``.
    """
    per_task = math.ceil(n_items / (_TASKS_PER_WORKER * max(1, n_workers)))
    return max(1, min(chunk_size, per_task))


# --- resume -------------------------------------------------------------------

#: ``(kind, directory, file)`` as returned by ``_classify_sink``.
_Sink = tuple[str, Path, Path]


def _part_rows(sink_dir: Path, columns: Sequence[str] | None) -> list[dict[str, Any]]:
    """Rows of every part file in a directory sink (only ``columns``, if given).

    An unreadable part is skipped with a warning: its stars then count as not
    done, are recomputed, and the part is rewritten.
    """
    rows: list[dict[str, Any]] = []
    for part in sorted(sink_dir.glob("part-*.parquet")):
        rows.extend(_part_rows_of(part, columns))
    return rows


def _part_rows_of(
    part: Path, columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Rows of one part file (only ``columns`` it has, if given); empty if unreadable.

    Read through our own handle, closed even when the file is not valid Parquet: a
    handle left to the garbage collector keeps Windows from replacing the part.
    """
    import pyarrow.parquet as pq

    try:
        with part.open("rb") as fh:
            parquet = pq.ParquetFile(fh)
            names = set(parquet.schema_arrow.names)
            cols = None if columns is None else [c for c in columns if c in names]
            rows: list[dict[str, Any]] = parquet.read(columns=cols).to_pylist()
    except Exception as exc:  # noqa: BLE001 - recomputing beats aborting the run
        warnings.warn(
            f"could not read {part} ({exc}); its light curves are computed again",
            UserWarning,
            stacklevel=3,
        )
        return []
    return rows


def _stored_key(key: str, sink: _Sink) -> str:
    """A batch key the way ``sink`` stores it (CSV text cells are formula-escaped)."""
    kind, _, path = sink
    if kind == "file" and path.suffix.lower() == ".csv":
        return str(csv_safe(key))
    return key


def _as_count(value: Any) -> int:
    """A stored row count (int, float or CSV text) as an int; -1 if unreadable."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _done_keys(
    items: Sequence[InputItem],
    tables: Mapping[str, _Sink],
    existing: Mapping[str, list[dict[str, Any]]],
    method: str,
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """The keys whose every requested table already holds a finished result.

    A summary row says the star finished (``status == "ok"``) with this method and
    how many O-C and ``P(t)`` rows it produced. The star is done only if each
    requested long table holds exactly that many rows for it — so a part lost to an
    interrupt, a long table added after the summary was written, or a file sink
    that never got finalised all make it be computed again.

    Also returns ``(key, stored source, input source)`` for the done keys whose
    summary row was written from a different file than this input (a same-named
    light curve from another folder): they stay done, and the caller reports them.
    """
    summary_sink = tables["summary"]
    if summary_sink[0] == "memory":
        return set(), []  # in-memory rows must be returned: everything is computed
    wanted = ["key", "source", "status", "method", *_ROW_COUNT_FIELD.values()]
    summary = (
        existing["summary"]
        if summary_sink[0] == "file"
        else _part_rows(summary_sink[1], wanted)
    )
    promised = {
        str(row.get("key")): row
        for row in summary
        if str(row.get("status")) == "ok" and str(row.get("method")) == method
    }
    if not promised:
        return set(), []
    stored: dict[str, Counter[str]] = {}
    for name in _ROW_COUNT_FIELD:
        if name in tables:
            kind, sink_dir, _ = tables[name]
            rows = existing[name] if kind == "file" else _part_rows(sink_dir, ["key"])
            stored[name] = Counter(str(row.get("key")) for row in rows)
    kind, _, summary_file = summary_sink
    csv_summary = kind == "file" and summary_file.suffix.lower() == ".csv"
    done: set[str] = set()
    clashes: list[tuple[str, str, str]] = []
    for key, src in items:
        row = promised.get(_stored_key(key, summary_sink))
        if row is not None and all(
            counts.get(_stored_key(key, tables[name]), 0)
            == _as_count(row.get(_ROW_COUNT_FIELD[name]))
            for name, counts in stored.items()
        ):
            done.add(key)
            label = source_label(src)
            if _source_clash(row.get("source"), label, csv_sink=csv_summary):
                clashes.append((key, str(row.get("source")), label))
    return done, clashes


def _merge_part(
    part: Path,
    rows: list[dict[str, Any]],
    replaced: set[str],
    order: Mapping[str, int],
    keep_old: bool,
) -> None:
    """Rewrite one directory part: its untouched stars' rows plus this run's rows.

    A chunk is recomputed only for its unfinished stars, so the rows of the others
    are carried over from the old part; rows are kept in input order. A part left
    with no rows at all is removed, so no stale part outlives a recomputation.
    """
    kept: list[dict[str, Any]] = []
    if keep_old and part.exists():
        kept = [r for r in _part_rows_of(part) if str(r.get("key")) not in replaced]
    merged = sorted(
        [*kept, *rows], key=lambda r: order.get(str(r.get("key")), len(order))
    )
    if merged:
        write_rows(merged, part)
    elif part.exists():
        part.unlink()


# --- the driver ---------------------------------------------------------------


def batch_pdot(
    inputs: Any,
    *,
    period: float | Mapping[Any, float] | None = None,
    method: str = "GLS",
    model: str | None = None,
    settings: PdotSettings | None = None,
    detrend_width: float | None = None,
    detrend_method: TrendMethod = "median",
    profile: bool = True,
    laws: str | Sequence[str] | None = AUTO_LAWS,
    ltte: bool = True,
    points_per_interval: int | None = None,
    period_window: float | None = None,
    backend: str | None = "auto",
    device: str = "cpu",
    workers: int | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    sink: str | Path | None = None,
    oc_sink: str | Path | None = None,
    profile_sink: str | Path | None = None,
    plots: str | Path | None = None,
    plot_keys: str | Sequence[Any] | None = None,
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
        Seed period in days: one number for every star (NumPy scalars and 0-d
        arrays included; it must be positive and finite), or a catalogue — a
        ``{key: period}`` mapping or a pandas ``Series`` indexed by key — to feed
        in per-star periods. Catalogue keys are matched as strings, as the batch
        keys are (``7`` matches the key ``"7"``); a missing, NaN or non-positive
        entry leaves that star to the blind search, and a catalogue matching no
        key at all warns. ``None`` lets each search find its own, which is
        slower and can lock onto an alias — pass known periods when you have
        them.
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
    laws : str or sequence of str, optional
        Period laws to rank on each track (names from
        :func:`cuperiod.pdot.laws.available_laws`; a bare string is one law);
        ``None`` skips the law fitting but keeps the track.
    ltte : bool, default True
        Also search each star's O-C for a cyclic (light-travel-time) variation
        with :func:`cuperiod.ltte_search`, reporting the outer period,
        amplitude, and the secular ``dP/dt`` with the cycle removed
        (``ltte_*`` columns). Skipped silently for stars with fewer than
        ``settings.ltte_min_points`` segments.
    points_per_interval, period_window : optional
        Passed to :func:`cuperiod.period_profile`.
    backend : str, default "auto"
        Backend request for the searches. ``"auto"`` (or ``None``) defers to the
        wrapped method's settings backend (``CUPERIOD_<METHOD>_BACKEND``). With
        ``device="cpu"`` it then means the best *CPU* backend — a CPU pool never
        opens a CUDA context per worker — and a GPU-only backend raises
        ``ValueError``. ``device="gpu"`` always requests the GPU.
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
        named after the key; a key that is not a plain file name, or that differs
        from another only in letter case, gets a short digest appended so no
        figure overwrites another). Needs matplotlib; a plotting failure is
        recorded as an empty ``figure`` cell and never fails the analysis.
        Generating thousands of figures is slow — pair it with ``plot_keys``.
    plot_keys : str or sequence, optional
        Restrict figures to these keys (matched as strings; a bare string is
        one key).
    resume : bool, default True
        Skip light curves already measured, for any mix of file and directory
        sinks: a star is skipped when its summary row is ``"ok"`` for this
        method and every requested O-C / ``P(t)`` table holds all of its rows.
        A failed star is retried, and a recomputed star replaces its rows in
        every table (one summary row per key). Changed settings are not
        detected — re-run with ``resume=False`` after changing them. Summary
        rows record their file (``source``), so a key the sink holds from a
        different file is reported (and still skipped). With an in-memory
        summary (``sink=None``) every star is computed.
    chunk_size : int, default 32
        Light curves per chunk/task. Much smaller than the periodogram batch's,
        because one light curve here costs a whole grid search. A directory
        sink uses exactly this (its part files are numbered by chunk);
        otherwise a CPU pool treats it as an upper bound and shrinks chunks so
        every worker gets several tasks.

    Returns
    -------
    BatchSummary
        ``methods`` is reported as ``("PDOT",)``. ``rows`` holds the summary
        table when ``sink is None``; ``n_done`` counts light curves that
        completed the chain in this run, ``n_skipped`` those resume left alone,
        and failures appear both in ``errors`` and as summary rows whose
        ``status`` is not ``"ok"``. Rows and errors are in input order.

    Raises
    ------
    ValueError
        Before any light curve is touched: an unknown ``device``, ``method``
        (or BLS/TLS), ``model`` or law name, a GPU-only backend with
        ``device="cpu"``, a non-positive seed ``period``, a bad
        ``detrend_width``, or directory sinks sharing one directory.
    TypeError
        If ``period`` is neither a number nor a mapping.

    Warns
    -----
    UserWarning
        When the period catalogue matches no key, when input keys repeat (the
        repeats are renamed ``key#2``, ...), or when a resumed sink holds a key
        from a different source file.
    DeprecationWarning
        When a ``CUPERIOD_BATCH_*`` variable is set: nothing reads them.

    Examples
    --------
    >>> cup.batch_pdot("lightcurves/*.csv", sink="pdot.parquet",
    ...                oc_sink="oc.parquet")                    # doctest: +SKIP
    >>> cup.batch_pdot(frames, period=known, detrend_width=40.0,
    ...                plots="figures/")                        # doctest: +SKIP
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    _warn_dead_batch_env()
    if detrend_width is not None and not (
        np.isfinite(detrend_width) and detrend_width > 0.0
    ):
        raise ValueError(
            f"detrend_width must be positive and finite, got {detrend_width}"
        )

    # Everything a worker would only discover per star is checked here, before any
    # input is read or sink created: a typo used to fail (or silently skip a
    # stage for) every light curve one by one, and leave a directory sink full of
    # failure rows that the corrected re-run then skipped.
    from cuperiod.pdot.scan import _UNSUPPORTED

    # A misspelt domain is never read for in-memory inputs, so it would otherwise
    # surface only on the first file or table input, deep in a worker.
    domain = coerce_domain(domain)
    wrapped = get_method(method)
    if wrapped.name in _UNSUPPORTED:
        raise ValueError(
            f"batch_pdot does not support {wrapped.name} (its trial grid carries "
            "box/transit structure); use GLS/PDM/CE/MHAOV/String-Length"
        )
    get_model(model)
    if detrend_method not in get_args(TrendMethod):
        raise ValueError(
            f"detrend_method must be one of {list(get_args(TrendMethod))}, "
            f"got {detrend_method!r}"
        )
    law_names = _law_names(laws)
    seed, seed_map = _seed_periods(period)
    # The backend every worker resolves. On a CPU pool 'auto' must become 'cpu':
    # resolved inside each worker it picks the CUDA fast path on a CUDA box, one
    # GPU context per core. The method's settings backend is the fallback.
    worker_backend = (
        "gpu"
        if device == "gpu"
        else cpu_pool_backend(backend, wrapped.coerce_settings(None))
    )
    cfg = _ChunkConfig(
        settings=settings or PdotSettings(),
        method=wrapped.name,
        model=model,
        backend=worker_backend,
        columns=columns,
        domain=domain,
        period=seed,
        period_map=seed_map,
        detrend_width=detrend_width,
        detrend_method=detrend_method,
        do_profile=profile,
        do_ltte=ltte,
        laws=law_names,
        points_per_interval=points_per_interval,
        period_window=period_window,
        store_oc=oc_sink is not None,
        store_profile=profile_sink is not None,
        plots_dir=str(plots) if plots is not None else None,
        plot_keys=_key_set(plot_keys),
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
    # The tables this run writes: the summary always, a long table only on request.
    tables: dict[str, _Sink] = {"summary": sinks["summary"]}
    if cfg.store_oc:
        tables["oc"] = sinks["oc"]
    if cfg.store_profile:
        tables["profile"] = sinks["profile"]

    items = resolve_inputs(inputs, columns=columns, domain=domain)
    keys = [key for key, _ in items]
    if seed_map is not None and keys and not any(key in seed_map for key in keys):
        warnings.warn(
            f"the period catalogue ({len(seed_map)} usable entries) matches none "
            f"of the {len(items)} input keys, so every star runs the blind search; "
            "catalogue keys are compared as strings with the batch keys",
            UserWarning,
            stacklevel=2,
        )
    if cfg.plots_dir is not None:
        plotted = (k for k in keys if cfg.plot_keys is None or k in cfg.plot_keys)
        cfg = replace(cfg, figure_stems=_figure_stems(plotted))

    size = max(1, chunk_size)
    has_dir = any(kind == "dir" for kind, _, _ in tables.values())
    for kind, sink_dir, _ in tables.values():
        if kind == "dir":
            _dir_manifest_guard(sink_dir, size, resume)

    # Rows already on disk, read once: they decide what is done, and file sinks
    # are rewritten from them at the end.
    existing: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    if resume:
        for name, (kind, _, sink_file) in tables.items():
            if kind == "file" and sink_file.exists():
                existing[name] = _read_existing_rows(sink_file)
    done: set[str] = set()
    if resume:
        done, clashes = _done_keys(items, tables, existing, wrapped.name)
        _warn_source_clashes(clashes)

    if device == "gpu":
        n_workers = workers if workers is not None else suggest_gpu_workers(method)
    else:
        n_workers = cpu_worker_count(workers)
    if has_dir:
        # Part files are numbered by chunk of the *whole* input list, so chunks
        # keep that layout; within a chunk only the unfinished stars are computed.
        chunks = _chunked(items, size)
    else:
        todo = [item for item in items if item[0] not in done]
        if device == "cpu" and n_workers > 1:
            size = _pool_chunk_size(len(todo), size, n_workers)
        chunks = _chunked(todo, size)
    work = [[item for item in chunk if item[0] not in done] for chunk in chunks]
    pending = [idx for idx, chunk in enumerate(work) if chunk]
    n_skipped = len(items) - sum(len(work[idx]) for idx in pending)
    recomputed = {key for idx in pending for key, _ in work[idx]}

    # Buffered per chunk and flattened in chunk order at the end: a pool finishes
    # chunks in completion order, and rows/errors should follow the inputs.
    buffered: dict[str, dict[int, list[dict[str, Any]]]] = {n: {} for n in TABLES}
    errors_by_chunk: dict[int, list[tuple[str, str]]] = {}
    n_done = 0

    def absorb(
        idx: int,
        produced: Mapping[str, list[dict[str, Any]]],
        errs: list[tuple[str, str]],
    ) -> None:
        nonlocal n_done
        n_done += sum(1 for r in produced["summary"] if r.get("status") == "ok")
        errors_by_chunk[idx] = errs
        order = {key: pos for pos, (key, _) in enumerate(chunks[idx])}
        for name in _WRITE_ORDER:  # summary last: its part marks the chunk landed
            if name not in tables:
                continue
            kind, sink_dir, _ = tables[name]
            if kind == "dir":
                _merge_part(
                    _part_path(sink_dir, idx),
                    produced[name],
                    recomputed,
                    order,
                    keep_old=resume,
                )
            else:
                buffered[name][idx] = produced[name]

    if n_workers <= 1 or not pending:
        for idx in pending:
            absorb(idx, *_process_chunk(work[idx], cfg))
            if device == "gpu":
                free_gpu_memory()
    elif device == "gpu":
        _run_pool(work, pending, cfg, absorb, n_workers, worker=_process_chunk)
    else:
        pool_size = min(n_workers, len(pending))
        # BLAS/OpenMP pinned to one thread only while the pool exists (the parent
        # session keeps its threads afterwards); numba gets its share per worker.
        with pinned_worker_threads():
            _run_pool(
                work,
                pending,
                cfg,
                absorb,
                pool_size,
                worker=_process_chunk,
                initializer=init_worker_threads,
                initargs=(numba_threads_per_worker(pool_size),),
            )

    rows = {
        name: [r for idx in sorted(by_chunk) for r in by_chunk[idx]]
        for name, by_chunk in buffered.items()
    }
    errors = [e for idx in sorted(errors_by_chunk) for e in errors_by_chunk[idx]]

    for name, sink_spec in tables.items():
        kind, _, sink_file = sink_spec
        if kind != "file" or (not pending and sink_file.exists()):
            continue  # an up-to-date file is left untouched
        # Rows of every recomputed star are replaced, never appended beside: a
        # failed row (method "") used to survive next to its later "ok" row.
        replaced = {_stored_key(key, sink_spec) for key in recomputed}
        kept = [r for r in existing[name] if str(r.get("key")) not in replaced]
        write_rows([*kept, *rows[name]], sink_file)

    return BatchSummary(
        n_inputs=len(items),
        n_done=n_done,
        n_failed=len(errors),
        n_skipped=n_skipped,
        methods=("PDOT",),
        device=device,
        sink=str(sink) if sink is not None else None,
        errors=errors,
        rows=rows["summary"] if tables["summary"][0] == "memory" else None,
    )


__all__ = ["TABLES", "batch_pdot", "pdot_to_rows"]
