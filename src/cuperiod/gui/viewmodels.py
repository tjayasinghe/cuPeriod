"""Frozen view models of an interactive pre-whitening session (no Qt).

A :class:`~cuperiod.prewhiten.session.PreWhitenSession` is confined to the compute
worker, so widgets never read it. After every job the worker condenses the head state
into one immutable :class:`SessionView` (rows of the frequency table, the spectra, the
light curve with its model and mask, the history list) and only that object crosses to
the GUI thread. Everything here is plain data: building a view is the worker's job,
painting it is the widgets'.

Two kinds of view exist (section 4.1 of the design):

* a **light** view after an edit or an Auto progress step: built from the state and
  its cached fit only, errors only where a covariance is already at hand (else NaN);
* a **full** view on Auto completion, ``finalize`` and explicit requests, carrying the
  :class:`~cuperiod.prewhiten.result.PreWhitenResult` with the configured
  uncertainty estimator.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import FloatArray, IntArray

if TYPE_CHECKING:
    from cuperiod.core.config import PreWhitenSettings
    from cuperiod.prewhiten.result import PreWhitenResult
    from cuperiod.prewhiten.session.report import RowStats
    from cuperiod.prewhiten.session.session import PreWhitenSession
    from cuperiod.prewhiten.session.state import (
        Candidate,
        Component,
        LogEntry,
        SessionSnapshot,
        SessionState,
    )
    from cuperiod.prewhiten.spectrum import AmplitudeSpectrum

BoolArray = npt.NDArray[np.bool_]

MarkerShape = Literal["free", "fixed", "tied", "inactive"]

_nan = float("nan")


@dataclass(frozen=True)
class SessionKey:
    """Identity of a live session in the GUI: one per source and band choice.

    ``data_id`` is the content fingerprint of the source's curve: a file reloaded with
    other numbers under the same name is another session, never a continuation of the
    old one (the cache keys of the one-shot results follow the same rule).
    """

    source_id: str
    band: str = ""
    data_id: str = ""


@dataclass(frozen=True, eq=False)
class ComponentView:
    """One row of the frequency table (every component, inactive ones included).

    ``amplitude`` and ``phase`` are signed, as the state holds them (a negative
    amplitude under a fixed phase is shown with a minus sign, flag
    ``negative_amplitude``); errors are NaN when no covariance is at hand.
    """

    id: int
    label: str
    rank: int | None
    frequency: float
    frequency_error: float
    period: float
    amplitude: float
    amplitude_error: float
    phase: float
    phase_error: float
    amplitude_rms: float | None
    snr: float
    fap: float
    delta_bic: float
    spectrum_amplitude: float
    amplitude_ratio: float
    blended: bool
    active: bool
    fix_frequency: bool
    fix_amplitude: bool
    fix_phase: bool
    tie: str | None
    variation: str
    combination: str | None
    origin: str
    flags: frozenset[str]
    note: str
    passes: bool


@dataclass(frozen=True, eq=False)
class SubsetView:
    """One subset of one partition (Subsets dock)."""

    partition: str
    index: int
    label: str
    n_points: int
    n_used: int
    t_start: float
    t_stop: float
    zero_point: float
    zero_point_error: float
    weight: float
    rms: float
    # Additive: the Subsets dock's tri-state "use" box reads these instead of
    # inferring them from n_used.
    excluded: bool = False
    n_deleted: int = 0


@dataclass(frozen=True)
class CandidateView:
    """What Step would try next (the spectrum's candidate marker)."""

    frequency: float
    period: float
    amplitude: float
    snr_estimate: float
    hint: str


@dataclass(frozen=True)
class HistoryItem:
    """One log entry of the History dock.

    "Copy as Python" is a controller request (``copy_entry_script(seq)``) rendered on
    the worker, not a field, so the view models do not depend on the script module.
    """

    seq: int
    group: int
    op: str
    text: str
    kind: str
    reachable: bool


@dataclass(frozen=True, eq=False)
class SolutionBundle:
    """What the controller caches per result key: the result and its session.

    ``tag`` is the submitter's own label of the Auto run that produced it (the
    controller passes the cache key it asked for), echoed back untouched.
    """

    result: PreWhitenResult
    snapshot: SessionSnapshot
    tag: object = None


@dataclass(frozen=True)
class MarkerState:
    """How the spectrum draws one component marker."""

    id: int
    frequency: float
    shape: MarkerShape
    selected: bool


@dataclass(frozen=True, eq=False)
class SessionView:
    """Everything a widget may paint about one session, as of one head state.

    ``result`` is the full result on a full view (``light=False``: Auto completion,
    ``finalize``, *Compute errors*). On a light per-edit view it carries the *light*
    result (``build_result(..., light=True)``) exactly when the committed change moved
    the set of active frequencies, which is when the controller emits
    ``solution_ready``; it is ``None`` on every other light view, Auto progress views
    included.
    """

    key: object
    serial: int
    id_epoch: int
    light: bool
    rows: tuple[ComponentView, ...]
    summary_html: str
    result: PreWhitenResult | None
    data_spectrum: AmplitudeSpectrum
    residual_spectrum: AmplitudeSpectrum
    window: AmplitudeSpectrum
    threshold: FloatArray | None
    candidate: CandidateView | None
    time: FloatArray
    value: FloatArray
    model: FloatArray
    residuals: FloatArray
    deleted: BoolArray
    point_codes: IntArray | None
    partitions: tuple[str, ...]
    subsets: tuple[SubsetView, ...]
    offsets_by: str | None
    variation_by: str | None
    weights_by: str | None
    warnings: tuple[str, ...]
    history: tuple[HistoryItem, ...]
    cursor: int
    can_undo: bool
    can_redo: bool
    pristine: bool
    dirty: bool
    status: str
    stop_reason: str
    project_path: str | None
    backend: str
    # Additive (not in the 4.1 listing): what the controller compares the Options form
    # with (D17) without touching the worker-confined session.
    settings: PreWhitenSettings | None = None
    requested_backend: str = ""
    active_frequencies: FloatArray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    #: The id the next added component gets (an Add-field tie names it).
    next_id: int = 0
    #: An Auto progress view (light, no candidate): selection rules wait for the end.
    progress: bool = False
    #: Codes of every partition (row highlights in the Subsets dock).
    partition_codes: Mapping[str, IntArray] = field(default_factory=dict)
    #: The session's per-point label columns (``Split by ▸ Column…``).
    label_columns: tuple[str, ...] = ()

    def row(self, id: int) -> ComponentView | None:  # noqa: A002 - the table's name
        """The row of component ``id`` (``None`` when there is none)."""
        for r in self.rows:
            if r.id == id:
                return r
        return None


# -- pure builders (run on the worker; no Qt) ------------------------------------------
def component_flags(component: Component, extra: Sequence[str] = ()) -> frozenset[str]:
    """The flags of a table row: the constraint flags plus any the fit reported."""
    flags = set(extra)
    if component.fix_frequency:
        flags.add("fixed_f")
    if component.fix_amplitude:
        flags.add("fixed_a")
    if component.fix_phase:
        flags.add("fixed_phi")
    if component.tie is not None:
        flags.discard("fixed_f")
        flags.add("tied")
    if component.amplitude < 0.0:
        flags.add("negative_amplitude")
    if not component.active:
        flags.add("inactive")
    return frozenset(flags)


def component_views(
    state: SessionState,
    rows: Sequence[RowStats],
    *,
    result: PreWhitenResult | None = None,
) -> tuple[ComponentView, ...]:
    """The table rows of ``state`` (table order), stats from ``report.rows``.

    ``result`` (full or light), when given, supplies the fit's own flags
    (``degenerate``, ``at_bound``, ...) for the active rows.
    """
    stats = {r.id: r for r in rows}
    fit_flags: dict[int, tuple[str, ...]] = {}
    if result is not None:
        active_ids = [c.id for c in state.active]
        if len(active_ids) == len(result.components):
            for cid, sinusoid in zip(active_ids, result.components, strict=True):
                fit_flags[cid] = sinusoid.flags
    out: list[ComponentView] = []
    rank = 0
    for c in state.components:
        s = stats.get(c.id)
        if c.active:
            rank += 1
        out.append(
            ComponentView(
                id=c.id,
                label=c.label,
                rank=rank if c.active else None,
                frequency=float(c.frequency),
                frequency_error=_nan if s is None else float(s.frequency_error),
                period=1.0 / c.frequency if c.frequency != 0.0 else math.inf,
                amplitude=float(c.amplitude),
                amplitude_error=_nan if s is None else float(s.amplitude_error),
                phase=float(c.phase),
                phase_error=_nan if s is None else float(s.phase_error),
                amplitude_rms=None if s is None else s.amplitude_rms,
                snr=_nan if s is None else float(s.snr),
                fap=float(c.fap),
                delta_bic=float(c.delta_bic),
                spectrum_amplitude=_nan if s is None else float(s.spectrum_amplitude),
                amplitude_ratio=_nan if s is None else float(s.amplitude_ratio),
                blended=False if s is None else bool(s.blended),
                active=bool(c.active),
                fix_frequency=bool(c.fix_frequency),
                fix_amplitude=bool(c.fix_amplitude),
                fix_phase=bool(c.fix_phase),
                tie=None if c.tie is None else c.tie.text(),
                variation=str(c.variation),
                combination=None if s is None else s.combination,
                origin=str(c.origin),
                flags=component_flags(c, fit_flags.get(c.id, ())),
                note=c.note,
                passes=False if s is None else bool(s.passes),
            )
        )
    return tuple(out)


def marker_states(
    rows: Sequence[ComponentView], selected: Sequence[int] = ()
) -> tuple[MarkerState, ...]:
    """The spectrum markers of ``rows``; shape never relies on colour alone."""
    chosen = set(selected)
    out = []
    for r in rows:
        shape: MarkerShape
        if not r.active:
            shape = "inactive"
        elif r.tie is not None:
            shape = "tied"
        elif r.fix_frequency:
            shape = "fixed"
        else:
            shape = "free"
        out.append(MarkerState(r.id, r.frequency, shape, r.id in chosen))
    return tuple(out)


def candidate_view(candidate: Candidate | None) -> CandidateView | None:
    """The spectrum's "next peak" marker, with a one-line hint."""
    if candidate is None:
        return None
    f = float(candidate.frequency)
    snr = float(candidate.snr_estimate)
    snr_text = f"{snr:.1f}" if math.isfinite(snr) else "–"
    return CandidateView(
        frequency=f,
        period=1.0 / f if f != 0.0 else math.inf,
        amplitude=float(candidate.spectrum_amplitude),
        snr_estimate=snr,
        hint=f"Next peak {f:.6f} c/d (S/N ≈ {snr_text}); Step (F7) tries it",
    )


def history_items(
    entries: Sequence[LogEntry], reachable: Sequence[bool]
) -> tuple[HistoryItem, ...]:
    """History rows for ``entries`` (``reachable[i]`` for entry ``i``)."""
    from cuperiod.prewhiten.session import actions

    return tuple(
        HistoryItem(
            seq=e.seq,
            group=e.group,
            op=e.op,
            text=actions.describe(e),
            kind=str(e.report.kind),
            reachable=bool(ok),
        )
        for e, ok in zip(entries, reachable, strict=True)
    )


def _session_history(
    session: PreWhitenSession,
) -> tuple[tuple[LogEntry, ...], int]:
    """Every log entry (the redo tail included) and the cursor."""
    return tuple(session.log_entries), int(session.history_cursor)


def _subset_views(
    session: PreWhitenSession,
    state: SessionState,
    used_mask: BoolArray,
    residuals: FloatArray,
    result: PreWhitenResult | None,
) -> tuple[SubsetView, ...]:
    time = session.data.time
    zp_errors: dict[str, float] = {}
    if result is not None and result.subsets is not None:
        sol = result.subsets
        if sol.offsets_by is not None:
            for label, err in zip(sol.offset_labels, sol.offset_error, strict=False):
                zp_errors[label] = float(err)
    deleted = np.asarray(state.mask.deleted, dtype=np.bool_)
    out: list[SubsetView] = []
    for p in state.partitions:
        codes = np.asarray(p.codes)
        excluded = set(state.excluded_subsets(p.name))
        offsets = (
            state.subset_offsets
            if p.name == state.offsets_by and state.subset_offsets is not None
            else None
        )
        for i, label in enumerate(p.labels):
            members = codes == i
            n_points = int(members.sum())
            in_use = members & used_mask
            n_used = int(in_use.sum())
            n_deleted = int((members & deleted).sum()) if deleted.size else 0
            span = time[members]
            r = residuals[in_use]
            zero_point = (
                float(offsets[i]) if offsets is not None and i < len(offsets) else _nan
            )
            out.append(
                SubsetView(
                    partition=p.name,
                    index=i,
                    label=label,
                    n_points=n_points,
                    n_used=n_used,
                    t_start=float(span.min()) if span.size else _nan,
                    t_stop=float(span.max()) if span.size else _nan,
                    zero_point=zero_point,
                    zero_point_error=zp_errors.get(label, _nan)
                    if offsets is not None
                    else _nan,
                    weight=float(p.weights[i]) if i < len(p.weights) else 1.0,
                    rms=float(np.sqrt(np.mean(r * r))) if r.size else _nan,
                    excluded=i in excluded,
                    n_deleted=n_deleted,
                )
            )
    return tuple(out)


def _summary_html(
    state: SessionState,
    *,
    n_points: int,
    n_used: int,
    backend: str,
    status: str,
) -> str:
    k_active = len(state.active)
    k_all = len(state.components)
    stats = state.stats
    counts = (
        "<b>no</b> components yet"  # not "0 active of 0 components"
        if k_all == 0
        else f"<b>{k_active}</b> active of {k_all} component{'s' if k_all != 1 else ''}"
    )
    parts = [
        counts,
        f"rms {stats.rms:.4g}",
        f"BIC {stats.bic:.6g}" if math.isfinite(stats.bic) else "BIC –",
        f"{n_used} of {n_points} points in use",
        f"backend {html.escape(backend)}",
    ]
    if status:
        parts.append(html.escape(status))
    return " · ".join(parts)


def _threshold(
    spectrum: AmplitudeSpectrum, settings: PreWhitenSettings
) -> FloatArray | None:
    """The S/N acceptance line over the residual spectrum (None without an S/N rule)."""
    if "snr" not in settings.stop_criteria:
        return None
    noise = spectrum.noise_curve(
        window=settings.snr_window, estimator=settings.noise_estimator
    )
    return np.asarray(noise, dtype=np.float64) * float(settings.snr_threshold)


def build_session_view(
    session: PreWhitenSession,
    *,
    key: object,
    id_epoch: int,
    result: PreWhitenResult | None = None,
    previous_frequencies: FloatArray | None = None,
    progress: bool = False,
    dirty: bool = False,
    project_path: str | None = None,
    warnings: Sequence[str] = (),
    status: str = "",
    requested_backend: str = "",
    threshold: bool = True,
) -> SessionView:
    """Condense ``session``'s head into a :class:`SessionView` (worker thread only).

    Parameters
    ----------
    session : PreWhitenSession
        The live session; must not be inside another call on this thread.
    key, id_epoch :
        The bridge's key of the session and its current id epoch.
    result : PreWhitenResult, optional
        A full result: the view is then full (``light=False``).
    previous_frequencies : array, optional
        The active frequencies of the previous light view. When given and different
        from the head's, the light result is built for ``solution_ready``.
    progress : bool
        An Auto progress view: light, no result, no candidate preview.
    threshold : bool
        Compute the S/N line (``noise_curve``) of the residual spectrum.
    """
    from cuperiod.prewhiten.session import report

    state = session.state
    kernel = session.kernel
    derived = kernel.derive(state)
    data = session.data
    light = result is None
    frequencies = np.asarray(state.active_frequencies, dtype=np.float64)
    if (
        light
        and not progress
        and previous_frequencies is not None
        and not _same_frequencies(previous_frequencies, frequencies)
    ):
        result = report.build_result(kernel, state, derived, light=True)
    rows = component_views(state, report.rows(kernel, state, derived), result=result)

    used = kernel.used(state)
    used_mask = np.zeros(data.n, dtype=np.bool_)
    used_mask[np.asarray(used.index)] = True
    fit = derived.fit
    # Model at every point: the headline model (plus the zero point the fit removed),
    # replaced by the exact model (subset zero points, per-subset amplitudes) where
    # the fit actually used the point.
    model = np.asarray(fit.model(data.time), dtype=np.float64) + kernel.zero_point(
        state
    )
    model[np.asarray(used.index)] = np.asarray(used.value) - np.asarray(fit.residuals)
    residuals = np.asarray(data.value, dtype=np.float64) - model

    data_spectrum = kernel.data_spectrum(state)
    residual_spectrum = derived.residual_spectrum
    window = kernel.window(state)
    line = _threshold(residual_spectrum, state.settings) if threshold else None
    candidate = None if progress else candidate_view(session.candidate())

    entries, cursor = _session_history(session)
    reach = [session.reachable(e.seq) for e in entries]
    point_partition = state.offsets_by or (
        state.partitions[0].name if state.partitions else None
    )
    codes = (
        np.asarray(state.partition(point_partition).codes, dtype=np.int64)
        if point_partition is not None
        else None
    )
    stop_reason = state.stop_reason
    status_text = status or (
        f"stopped: {stop_reason}" if state.status == "stopped" and stop_reason else ""
    )
    return SessionView(
        key=key,
        serial=state.serial,
        id_epoch=id_epoch,
        light=light,
        rows=rows,
        summary_html=_summary_html(
            state,
            n_points=data.n,
            n_used=int(used_mask.sum()),
            backend=session.backend,
            status=status_text,
        ),
        result=result,
        data_spectrum=data_spectrum,
        residual_spectrum=residual_spectrum,
        window=window,
        threshold=line,
        candidate=candidate,
        time=np.asarray(data.time, dtype=np.float64),
        value=np.asarray(data.value, dtype=np.float64),
        model=model,
        residuals=residuals,
        deleted=np.asarray(state.mask.deleted, dtype=np.bool_),
        point_codes=codes,
        partitions=tuple(p.name for p in state.partitions),
        subsets=_subset_views(session, state, used_mask, residuals, result),
        offsets_by=state.offsets_by,
        variation_by=state.variation_by,
        weights_by=state.weights_by,
        warnings=tuple(warnings),
        history=history_items(entries, reach),
        cursor=cursor,
        can_undo=session.can_undo,
        can_redo=session.can_redo,
        pristine=session.is_pristine,
        dirty=dirty,
        status=status_text,
        stop_reason=stop_reason,
        project_path=project_path,
        backend=session.backend,
        settings=state.settings,
        requested_backend=requested_backend,
        active_frequencies=frequencies,
        next_id=int(state.next_id),
        progress=bool(progress),
        partition_codes={
            p.name: np.asarray(p.codes, dtype=np.int64) for p in state.partitions
        },
        label_columns=tuple(data.labels),
    )


def _same_frequencies(a: FloatArray, b: FloatArray) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a, b))


__all__ = [
    "CandidateView",
    "ComponentView",
    "HistoryItem",
    "MarkerShape",
    "MarkerState",
    "SessionKey",
    "SessionView",
    "SolutionBundle",
    "SubsetView",
    "build_session_view",
    "candidate_view",
    "component_flags",
    "component_views",
    "history_items",
    "marker_states",
]
