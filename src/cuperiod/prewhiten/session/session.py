"""The public facade: :class:`PreWhitenSession`.

One mutating entry point, :meth:`PreWhitenSession.do`, validates an operation's
arguments, runs its reducer (plus the implicit Calculate a manual edit implies),
commits the new state, logs the action and makes it undoable. Every typed method
(``step``, ``add``, ``fix``, ``delete_points``, ...) is a one-line call of ``do``.
Bad input never raises out of ``do``: it comes back as a ``"refused"`` report and
leaves the session untouched (the error contract); only programming errors
propagate.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import ValidationError

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PreWhitenSettings, SessionSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import (
    MultiBandLightCurve,
    raise_if_unusable,
    stack_bands,
)
from cuperiod.prewhiten.constraints import ConstraintError, Variation
from cuperiod.prewhiten.engine import _as_light_curve, default_prewhiten_grid
from cuperiod.prewhiten.fit import MultiSineFit, Refinement
from cuperiod.prewhiten.result import PreWhitenResult
from cuperiod.prewhiten.session import actions
from cuperiod.prewhiten.session import stepper as _stepper
from cuperiod.prewhiten.session.history import History
from cuperiod.prewhiten.session.kernel import ExactUniformGrid, SessionKernel
from cuperiod.prewhiten.session.state import (
    Candidate,
    Component,
    CreationArgs,
    Derived,
    LogEntry,
    ProjectError,
    SessionData,
    SessionError,
    SessionSnapshot,
    SessionState,
    StepReport,
    VariationResult,
    content_sha256,
    parse_ref,
)
from cuperiod.prewhiten.spectrum import AmplitudeSpectrum, SpectrumEngine
from cuperiod.prewhiten.subsets import Partition

if TYPE_CHECKING:
    from cuperiod.core._typing import FloatArray

ComponentRef = int | str

#: Exceptions a reducer may raise for bad input or an unusable state: ``do()`` rolls
#: back and reports them as ``refused``. Anything else is a programming error.
_REFUSABLE: tuple[type[BaseException], ...] = (
    ValidationError,
    SessionError,
    ConstraintError,
    ValueError,
    np.linalg.LinAlgError,
)

#: The modules whose reducers register the manual and data ops. They are imported at
#: the first do(), not with this module, so ``import cuperiod`` stays light; a
#: missing one is a broken install and raises (no op is silently unavailable).
_OP_MODULES = ("ops_components", "ops_data")
_ops_loaded = False


def _load_ops() -> None:
    """Import the reducer modules once."""
    global _ops_loaded
    if _ops_loaded:
        return
    for name in _OP_MODULES:
        importlib.import_module(f"cuperiod.prewhiten.session.{name}")
    _ops_loaded = True


def _delegate(module: str) -> Any:
    """A heavier module of the session package, imported when first needed."""
    return importlib.import_module(f"cuperiod.prewhiten.session.{module}")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refs(ids: ComponentRef | Iterable[ComponentRef]) -> list[int]:
    if isinstance(ids, (int, str, np.integer)):
        return [parse_ref(ids)]  # type: ignore[arg-type]
    return [parse_ref(i) for i in ids]


def _floats(values: Any) -> Any:
    """Plain floats where possible; anything else is left for the model to refuse."""
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return values


def _raw_refs(ids: Any) -> Any:
    """A wrapper's ids as a list, left unparsed: the op's argument model parses them,
    so a bad reference is refused by ``do()`` instead of raising here (D21)."""
    if isinstance(ids, (int, str, np.integer)):
        return [ids]
    try:
        return list(ids)
    except TypeError:
        return ids


def _filter_labels(
    labels: Mapping[str, Any] | None, usable: np.ndarray, n_rows: int
) -> dict[str, np.ndarray]:
    """Per-point label columns, filtered by the same rule as ``LightCurve.finite``."""
    out: dict[str, np.ndarray] = {}
    for name, column in (labels or {}).items():
        array = np.asarray(column)
        if array.shape != (n_rows,):
            raise ValueError(
                f"label column {name!r} has {array.shape[0] if array.ndim else 0} "
                f"rows, the light curve {n_rows}"
            )
        out[str(name)] = array if bool(usable.all()) else array[usable]
    return out


def _unknown_label(label: str, partition: Partition) -> str:
    """Refusal text for a subset label that ``partition`` does not have."""
    shown = ", ".join(repr(x) for x in partition.labels[:8])
    more = ", ..." if partition.n_subsets > 8 else ""
    return (
        f"subset label {label!r} is not in partition {partition.name!r} "
        f"(its labels: {shown}{more})"
    )


class PreWhitenSession:
    """Step-wise, editable pre-whitening of one light curve.

    Same coercion, checks, grid, engine and errors as :func:`cuperiod.prewhiten`, in
    that order. ``prewhiten(data, ...)`` is ``PreWhitenSession(data, ...,
    record=False)`` run to its stopping rule and finalized.

    Parameters
    ----------
    data : various
        A single-band light curve in any form :func:`cuperiod.to_input` accepts.
    settings : PreWhitenSettings, optional
        The requested settings (kept as given; the resolved grid is in the state).
    grid : GridSpec or (f0, df, nf), optional
        A custom uniform frequency grid; defaults to
        :func:`~cuperiod.prewhiten.engine.default_prewhiten_grid`.
    backend : str, optional
        Overrides ``settings.backend``; fixed for the life of the session.
    columns, domain : optional
        Column mapping and brightness domain for table inputs.
    t_ref : float, optional
        Phase epoch; defaults to the weighted mean epoch (frozen at creation).
    labels : Mapping[str, array], optional
        Extra per-point columns aligned with the input rows (``"site"``, ``"band"``),
        filtered alike.
    partitions : sequence of Partition or str
        Root partitions (part of the root state, never logged); a string names a
        label column and means ``Partition.from_labels(labels[name], name=name)``.
    session_settings : SessionSettings, optional
        Undo depth, cache sizes, protection rules.
    record : bool, default True
        Keep the log and the undo history; ``False`` keeps only the head (what
        :func:`cuperiod.prewhiten` and batch workers use).

    Examples
    --------
    >>> session = cup.PreWhitenSession((time, mag, err))           # doctest: +SKIP
    >>> session.step()                                             # doctest: +SKIP
    >>> session.add(12.3456)                                       # doctest: +SKIP
    >>> session.undo()                                             # doctest: +SKIP
    >>> result = session.finalize()                                # doctest: +SKIP
    """

    #: The column mapping of a table input (a project may then reference the file).
    _columns: ColumnMap | None

    def __init__(
        self,
        data: Any,
        *,
        settings: PreWhitenSettings | None = None,
        grid: GridSpec | tuple[float, float, int] | None = None,
        backend: str | None = None,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        t_ref: float | None = None,
        labels: Mapping[str, Any] | None = None,
        partitions: Sequence[Partition | str] = (),
        session_settings: SessionSettings | None = None,
        record: bool = True,
    ) -> None:
        cfg = settings or PreWhitenSettings()
        raw = _as_light_curve(data, columns, domain)
        # Name a placeholder error column (all 0, -1 or NaN) as the cause, rather than
        # the "no usable time baseline" its emptied curve would otherwise report.
        raise_if_unusable(raw, "pre-whitening")
        lc = raw.finite()
        n = lc.n
        if n < cfg.min_detections:
            raise InsufficientDataError(
                f"pre-whitening: {n} finite points < min_detections "
                f"{cfg.min_detections}"
            )
        if lc.baseline <= 0.0:
            raise InsufficientDataError("pre-whitening: no usable time baseline")
        if grid is None:
            trial_grid: GridSpec = default_prewhiten_grid(lc, cfg)
        elif isinstance(grid, GridSpec):
            trial_grid = grid
        else:
            trial_grid = ExactUniformGrid.of(grid)
        requested = backend or cfg.backend
        engine = SpectrumEngine(
            lc.time,
            lc.error,
            grid=trial_grid,
            normalization=cfg.normalization,
            backend=requested,
            device=cfg.device,
            precision=cfg.precision,
            eps=cfg.nufft_eps,
            freq_batch=cfg.direct_freq_batch,
            **({} if t_ref is None else {"t_ref": float(t_ref)}),
        )
        usable = raw._usable()
        label_columns = _filter_labels(labels, usable, raw.n)
        session_data = SessionData(
            time=lc.time,
            value=lc.value,
            error=lc.error,
            domain=lc.domain,
            source_index=np.flatnonzero(usable).astype(np.int64),
            labels=label_columns,
            meta=dict(lc.meta),
            content_sha256=content_sha256(lc.time, lc.value, lc.error),
        )
        roots: list[Partition] = []
        for item in partitions:
            if isinstance(item, Partition):
                if item.n_points != n:
                    raise ValueError(
                        f"partition {item.name!r} has {item.n_points} codes for "
                        f"{n} points"
                    )
                roots.append(item)
            elif item in label_columns:
                roots.append(Partition.from_labels(label_columns[item], name=item))
            else:
                raise ValueError(f"no label column {item!r} for a root partition")
        params = trial_grid.uniform_frequency_params()
        creation = CreationArgs(
            backend=engine.backend,
            t_ref=float(engine.t_ref),
            grid=(float(params[0]), float(params[1]), int(params[2])),
            settings=cfg.model_copy(),
            session_settings=session_settings or SessionSettings(),
            partitions=tuple(roots),
            label_names=tuple(label_columns),
            content_sha256=session_data.content_sha256,
        )
        kernel, root, derived = SessionKernel.open(
            session_data, creation, grid=trial_grid, engine=engine,
            backend_request=requested,
        )
        self._setup(kernel, root, record=record)
        # Remembered so a project can reference (rather than embed) the data file.
        self._columns = columns

    def _setup(
        self,
        kernel: SessionKernel,
        head: SessionState,
        *,
        record: bool,
        history: History | None = None,
    ) -> None:
        self._kernel = kernel
        self._record = bool(record)
        depth = kernel.session_settings.undo_depth
        self._history: History | None = (
            history if history is not None else (History(head, depth=depth))
        ) if self._record else None
        self._head = head
        self._lock = threading.RLock()
        self._pristine = True
        self._saved_serial: int | None = None
        self._columns = None

    # -- alternative constructors ------------------------------------------------------
    @classmethod
    def from_bands(
        cls, data: MultiBandLightCurve, *, centre: bool = True, **kwargs: Any
    ) -> PreWhitenSession:
        """A session on all bands merged, with a root ``band`` partition.

        Each band's mean is removed first (``centre=True``), exactly as the GUI's
        stacked view does; ``use_offsets("band")`` then fits a zero point per band.
        """
        lc, band = stack_bands(data, centre=centre)
        return cls(lc, labels={"band": band}, partitions=["band"], **kwargs)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> PreWhitenSession:
        """A session on a data file written by :meth:`export_script`."""
        lc, labels = _delegate("project").read_session_csv(path)
        return cls(lc, labels=labels, **kwargs)

    # -- thread confinement ------------------------------------------------------------
    @contextmanager
    def _guard(self) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise SessionError("PreWhitenSession used from two threads")
        try:
            yield
        finally:
            self._lock.release()

    def _group(self) -> AbstractContextManager[Any]:
        return self._history.group() if self._history is not None else nullcontext()

    # -- read-only ---------------------------------------------------------------------
    @property
    def state(self) -> SessionState:
        """The head state."""
        return self._head

    @property
    def components(self) -> tuple[Component, ...]:
        return self._head.components

    @property
    def status(self) -> Literal["searching", "stopped"]:
        return self._head.status

    @property
    def history(self) -> tuple[LogEntry, ...]:
        """The effective log, root to head (empty with ``record=False``)."""
        return self._history.effective() if self._history is not None else ()

    @property
    def log_entries(self) -> tuple[LogEntry, ...]:
        """Every log entry, the redo tail after :attr:`history_cursor` included.

        ``log_entries[:history_cursor]`` is :attr:`history`; what follows is what
        :meth:`redo` would re-apply (a History dock shows both).
        """
        return self._history.entries if self._history is not None else ()

    @property
    def history_cursor(self) -> int:
        """How many of :attr:`log_entries` the head has applied."""
        return self._history.cursor if self._history is not None else 0

    @property
    def data(self) -> SessionData:
        """The session's data, as read-only views."""
        return self._kernel.data.readonly()

    @property
    def settings(self) -> PreWhitenSettings:
        """The requested settings of the head."""
        return self._head.settings

    @property
    def session_settings(self) -> SessionSettings:
        return self._kernel.session_settings

    @property
    def t_ref(self) -> float:
        return self._kernel.t_ref

    @property
    def backend(self) -> str:
        return self._kernel.backend

    @property
    def kernel(self) -> SessionKernel:
        """The numerical core (for reducers, views and tests)."""
        return self._kernel

    @property
    def derived(self) -> Derived:
        """The head's fit and residual spectrum (rebuilt if evicted)."""
        with self._guard():
            return self._kernel.derive(self._head)

    @property
    def fit(self) -> MultiSineFit:
        return self.derived.fit

    @property
    def residuals(self) -> FloatArray:
        return self.derived.fit.residuals

    @property
    def can_undo(self) -> bool:
        return self._history is not None and self._history.can_undo

    @property
    def can_redo(self) -> bool:
        return self._history is not None and self._history.can_redo

    @property
    def is_pristine(self) -> bool:
        """The log holds only automatic entries (step, polish, prune, renumber)."""
        if self._history is None:
            return self._pristine
        return all(e.op in actions.PRISTINE_OPS for e in self._history.effective())

    @property
    def is_dirty(self) -> bool:
        """Changed since the last save (or since creation when never saved)."""
        if self._saved_serial is None:
            return bool(self.history)
        return self._head.serial != self._saved_serial

    def spectrum(self) -> AmplitudeSpectrum:
        """The data spectrum of the points in use, zero points removed."""
        with self._guard():
            return self._kernel.data_spectrum(self._head)

    def residual_spectrum(self) -> AmplitudeSpectrum:
        """The spectrum of the head's residuals."""
        return self.derived.residual_spectrum

    def window(self) -> AmplitudeSpectrum:
        """The spectral window of the points in use."""
        with self._guard():
            return self._kernel.window(self._head)

    def candidate(self) -> Candidate | None:
        """What :meth:`step` would try next (a preview; nothing changes)."""
        with self._guard():
            from cuperiod.prewhiten.engine import _peak_fap

            state = self._head
            kernel = self._kernel
            derived = kernel.derive(state)
            scan = derived.scan
            parts = [derived.fit.frequency, np.asarray(state.blocked, dtype=np.float64)]
            if state.inactive_frequencies.size:
                parts.append(state.inactive_frequencies)
            index = scan.peak_index(
                exclude=np.concatenate(parts), separation=kernel.separation(state)
            )
            if index is None:
                return None
            frequency, amplitude = scan.refine_peak(index)
            cfg = state.settings
            noise = scan.noise_at(
                frequency, window=cfg.snr_window, estimator=cfg.noise_estimator
            )
            used = kernel.used(state)
            return Candidate(
                frequency=float(frequency),
                index=int(index),
                spectrum_amplitude=float(scan.amplitude[index]),
                snr_estimate=(
                    float(amplitude) / noise
                    if np.isfinite(noise) and noise > 0.0
                    else float("nan")
                ),
                fap=_peak_fap(
                    used.time, used.error_eff, _stepper._power_at(scan, index),
                    f_max=kernel.f_max(state),
                ),
            )

    def result(
        self,
        *,
        uncertainty: str | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PreWhitenResult:
        """Report the head as it is (pure: the state does not change)."""
        with self._guard():
            return self._result(uncertainty=uncertainty, should_stop=should_stop)

    def _result(
        self,
        *,
        uncertainty: str | None = None,
        should_stop: Callable[[], bool] | None = None,
        light: bool = False,
    ) -> PreWhitenResult:
        from cuperiod.prewhiten.session.report import build_result

        state = self._head
        return build_result(
            self._kernel, state, self._kernel.derive(state),
            uncertainty=uncertainty, should_stop=should_stop, light=light,
        )

    def variations(
        self,
        ids: Iterable[ComponentRef] | None = None,
        *,
        mode: Variation = "both",
        by: str | None = None,
        improve_frequencies: bool = False,
        should_stop: Callable[[], bool] | None = None,
    ) -> VariationResult:
        """Per-subset amplitude/phase with a constancy test (a query, no change)."""
        with self._guard():
            module = _delegate("variation")
            state = self._head
            selected = None if ids is None else _refs(list(ids))
            result: VariationResult = module.compute(
                self._kernel, state, self._kernel.derive(state), selected,
                mode=mode, by=by, improve_frequencies=improve_frequencies,
                should_stop=should_stop,
            )
            return result

    def subset_rms_weights(self, name: str) -> tuple[float, ...]:
        """Subset weights ``1 / rms_s^2`` of the head's residuals (mean one).

        A query: pass the numbers to :meth:`set_subset_weights` to use them (the log
        then stores the numbers, so a replay does not depend on the residuals). A
        subset with no point in use gets weight 1.

        Raises
        ------
        SessionError
            When there is no partition called ``name``.
        """
        with self._guard():
            from cuperiod.prewhiten.session.ops_data import residual_rms_weights

            state = self._head
            return residual_rms_weights(
                self._kernel, state, self._kernel.derive(state), name
            )

    # -- the one mutating entry point --------------------------------------------------
    def do(self, op: str, **args: Any) -> StepReport:
        """Validate, reduce, commit and log one operation.

        Returns a ``"refused"`` report (and changes nothing) for invalid arguments or
        an action the state cannot take, ``"noop"`` for one that changes nothing.
        """
        with self._guard():
            return self._do(op, args)

    def group(self) -> AbstractContextManager[Any]:
        """Actions inside the block undo as one."""
        return self._group()

    def _do(self, op: str, raw: Mapping[str, Any]) -> StepReport:
        _load_ops()
        try:
            parsed = actions.parse(op, raw)
        except SessionError as exc:
            return StepReport(kind="refused", message=str(exc), hint=exc.hint)
        reduce = actions.reducer(op)
        if reduce is None:
            return StepReport(kind="refused", message=f"op {op!r} is not available")
        state = self._head
        kernel = self._kernel
        try:
            derived = kernel.derive(state)
            new, new_derived, report = reduce(kernel, state, derived, parsed)
            if report.committed and actions.OP_CLASS[op] == "manual":
                new, new_derived = self._manual_effects(new, new_derived)
        except _REFUSABLE as exc:
            # A refusal that names a remedy carries it as a code too (SessionError's
            # hint), so a front end need not parse the keyword advice in the message.
            return StepReport(
                kind="refused",
                message=str(exc) or type(exc).__name__,
                hint=getattr(exc, "hint", None),
            )
        if not report.committed:
            return report
        self._commit(op, parsed, state, derived, new, new_derived, report)
        return report

    def _manual_effects(
        self, new: SessionState, new_derived: Derived | None
    ) -> tuple[SessionState, Derived | None]:
        """A manual edit: implicit Calculate, then searching with a fresh budget."""
        if new_derived is None and self._kernel.session_settings.recalculate_on_edit:
            new, new_derived = self._kernel.recalculate(new)
        new = replace(
            new,
            status="searching",
            stop_reason="",
            blocked=(),
            budget_origin=new.n_iterations,
        )
        return new, new_derived

    def _commit(
        self,
        op: str,
        parsed: actions.OpArgs,
        state: SessionState,
        derived: Derived,
        new: SessionState,
        new_derived: Derived | None,
        report: StepReport,
    ) -> None:
        kernel = self._kernel
        if new.serial <= state.serial or new is state:
            new = replace(new, serial=kernel.next_serial())
        cached = kernel.cached(new)
        if new_derived is None and actions.OP_CLASS[op] == "neutral":
            new_derived = derived  # parameters of the model did not change
        if new_derived is not None and cached is not new_derived:
            kernel.remember(new, new_derived)
        if self._history is not None:
            args = (
                dict(report.resolved)
                if report.resolved is not None
                else actions.canonical(parsed)
            )
            entry = LogEntry(
                seq=self._history.next_seq(),
                group=self._history.allocate_group(),
                op=op,
                args=args,
                report=replace(report, resolved=None, changed=report.changed),
                timestamp=_now(),
            )
            self._history.push(entry, new)
        elif op not in actions.PRISTINE_OPS:
            self._pristine = False
        self._head = new

    # -- the automatic path ------------------------------------------------------------
    def step(
        self, *, seed: float | None = None, index: int | None = None
    ) -> StepReport:
        """One extraction attempt (one iteration of the automatic loop).

        With ``seed`` (and ``index``, the grid index whose power gives the FAP) the
        peak search is skipped and the trial fit starts at ``seed``: how replay and
        exported scripts repeat a logged attempt bitwise on any backend.
        """
        return self.do("step", seed=seed, index=index)

    def advance(self) -> StepReport:
        """:meth:`step` until the outcome is not ``"blocked"``."""
        with self._guard(), self._group():
            while True:
                report = self._do("step", {})
                if report.kind != "blocked":
                    return report

    def run(
        self,
        *,
        max_steps: int | None = None,
        seeds: Sequence[tuple[float, int] | None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_step: Callable[[StepReport], None] | None = None,
    ) -> StepReport:
        """:meth:`step` until the search stops (one undo group).

        With ``seeds``, exactly ``len(seeds)`` attempts, attempt ``i`` seeded by
        ``seeds[i]`` (``None`` = search). ``should_stop`` is polled before each step;
        a stopped run resumes where it left off.
        """
        with self._guard(), self._group():
            last = StepReport(kind="noop", message="nothing to do")
            if seeds is not None:
                for pair in seeds:
                    if should_stop is not None and should_stop():
                        break
                    raw = {} if pair is None else {"seed": pair[0], "index": pair[1]}
                    last = self._do("step", raw)
                    if on_step is not None:
                        on_step(last)
                return last
            count = 0
            while self._head.status != "stopped":
                if max_steps is not None and count >= max_steps:
                    break
                if should_stop is not None and should_stop():
                    break
                last = self._do("step", {})
                count += 1
                if on_step is not None:
                    on_step(last)
                if last.kind in ("refused", "noop"):
                    break
            return last

    def polish(self) -> StepReport:
        """Final simultaneous (or cyclic) fit of every frequency."""
        return self.do("polish")

    def prune(self) -> StepReport:
        """The v1.3 S/N re-check; renumbers when it removed something."""
        return self.do("prune")

    def renumber(self) -> StepReport:
        """Compact ids to ``1..K`` in table order; ties are rewritten."""
        return self.do("renumber")

    def finalize(self) -> PreWhitenResult:
        """Polish and prune per the settings, then :meth:`result` (one undo group)."""
        with self._guard(), self._group():
            cfg = self._head.settings
            if cfg.final_refine and self._head.active:
                self._do("polish", {})
            if cfg.prune and "snr" in cfg.stop_criteria and self._head.active:
                self._do("prune", {})
            return self._result()

    # -- the manual loop ---------------------------------------------------------------
    def add_peak(
        self, *, seed: float | None = None, index: int | None = None
    ) -> StepReport:
        """Add the tallest admissible peak; the stopping rules inform, never block."""
        return self.do("add_peak", seed=seed, index=index)

    def add(
        self,
        frequency: float,
        *,
        snap: bool = True,
        refine: Refinement | None = None,
        allow_close: bool = False,
    ) -> StepReport:
        """Add a component at (``snap``: the residual peak nearest) ``frequency``."""
        with self._guard():
            target: Any = frequency
            try:
                value = float(frequency)
            except (TypeError, ValueError):
                pass  # the argument model refuses it
            else:
                target = value
                if snap and np.isfinite(value) and value > 0.0:
                    target = self._snap(value)
            return self._do(
                "add",
                {"frequency": target, "refine": refine, "allow_close": allow_close},
            )

    def _snap(self, frequency: float) -> float:
        """The residual-spectrum local maximum within ``snap_rayleigh / T``."""
        state = self._head
        kernel = self._kernel
        radius = kernel.session_settings.snap_rayleigh / kernel.baseline(state)
        scan = kernel.derive(state).scan
        freq, amp = scan.frequency, scan.amplitude
        lo = int(np.searchsorted(freq, frequency - radius, side="left"))
        hi = int(np.searchsorted(freq, frequency + radius, side="right"))
        if hi - lo < 1:
            return frequency
        index = lo + int(np.argmax(amp[lo:hi]))
        refined, _ = scan.refine_peak(index)
        return float(refined)

    def remove(self, *ids: ComponentRef, cascade: bool = False) -> StepReport:
        return self.do("remove", ids=_raw_refs(ids), cascade=cascade)

    def set_active(
        self, ids: ComponentRef | Iterable[ComponentRef], active: bool
    ) -> StepReport:
        return self.do("set_active", ids=_raw_refs(ids), active=active)

    def fix(
        self,
        ids: ComponentRef | Iterable[ComponentRef],
        *,
        frequency: bool | None = None,
        amplitude: bool | None = None,
        phase: bool | None = None,
    ) -> StepReport:
        return self.do(
            "fix", ids=_raw_refs(ids), frequency=frequency, amplitude=amplitude,
            phase=phase,
        )

    def set_values(
        self,
        id: ComponentRef,  # noqa: A002 - the op's argument name
        *,
        frequency: float | None = None,
        amplitude: float | None = None,
        phase: float | None = None,
    ) -> StepReport:
        return self.do(
            "set_values", id=id, frequency=frequency, amplitude=amplitude,
            phase=phase,
        )

    def tie(
        self,
        id: ComponentRef,  # noqa: A002 - the op's argument name
        expression: str | None,
        *,
        allow_close: bool = False,
    ) -> StepReport:
        """Tie a frequency to others (``"F1 + F2"``, ``"2F1"``); ``None`` unties."""
        return self.do(
            "tie", id=id, expression=expression, allow_close=allow_close
        )

    def set_variation(
        self,
        ids: ComponentRef | Iterable[ComponentRef],
        mode: Variation,
        *,
        hold_frequency: bool = True,
    ) -> StepReport:
        return self.do(
            "set_variation", ids=_raw_refs(ids), mode=mode,
            hold_frequency=hold_frequency,
        )

    def calculate(self) -> StepReport:
        """Amplitudes, phases and zero points, with their errors; frequencies held."""
        return self.do("calculate")

    def improve(self, ids: Iterable[ComponentRef] | None = None) -> StepReport:
        """Non-linear fit of all (``None``) or the selected frequencies."""
        return self.do("improve", ids=None if ids is None else _raw_refs(ids))

    def set_note(self, id: ComponentRef, note: str) -> StepReport:  # noqa: A002
        return self.do("set_note", id=id, note=note)

    # -- data --------------------------------------------------------------------------
    def _runs(self, indices: Any) -> list[tuple[int, int]]:
        """Index runs from ints, a boolean mask or ``[(start, stop), ...]``.

        Raises
        ------
        SessionError
            For anything else (the wrappers turn it into a refusal).
        """
        from cuperiod.prewhiten.session.state import PointMask

        try:
            if isinstance(indices, (list, tuple)) and indices and all(
                isinstance(r, (list, tuple)) and len(r) == 2 for r in indices
            ):
                return [(int(a), int(b)) for a, b in indices]
            n = self._kernel.data.n
            return PointMask.none(n).without(np.asarray(indices)).runs()
        except (TypeError, ValueError, OverflowError) as exc:
            raise SessionError(f"not a point selection: {exc}") from None

    def delete_points(self, indices: Any) -> StepReport:
        """Leave points out of the fit (ints, a boolean mask, or index runs)."""
        try:
            runs = self._runs(indices)
        except SessionError as exc:
            return StepReport(kind="refused", message=str(exc))
        return self.do("delete_points", runs=runs)

    def restore_points(self, indices: Any | None = None) -> StepReport:
        """Bring deleted points back (``None``: all of them)."""
        if indices is None:
            return self.do("restore_points", runs=None)
        try:
            runs = self._runs(indices)
        except SessionError as exc:
            return StepReport(kind="refused", message=str(exc))
        return self.do("restore_points", runs=runs)

    def exclude_subsets(
        self, name: str, subsets: Iterable[int | str], *, excluded: bool = True
    ) -> StepReport:
        """Leave whole subsets out of (or back into) the fit.

        ``subsets`` are subset indices or labels of partition ``name``.
        """
        try:
            partition = self._head.partition(name)
            indices: list[int] = []
            for s in subsets:
                if isinstance(s, str):
                    if s not in partition.labels:
                        raise SessionError(_unknown_label(s, partition))
                    indices.append(partition.labels.index(s))
                else:
                    indices.append(int(s))
        except (SessionError, ValueError, TypeError, OverflowError) as exc:
            return StepReport(kind="refused", message=str(exc))
        return self.do(
            "exclude_subsets", name=name, subsets=indices, excluded=excluded
        )

    def define_partition(self, partition: Partition) -> StepReport:
        return self.do(
            "define_partition",
            name=partition.name,
            labels=list(partition.labels),
            runs=[list(r) for r in partition.runs()],
            weights=list(partition.weights),
            origin=partition.origin.model_dump(mode="json"),
        )

    def drop_partition(self, name: str) -> StepReport:
        return self.do("drop_partition", name=name)

    def set_subset_weights(self, name: str, weights: Sequence[float]) -> StepReport:
        return self.do("set_subset_weights", name=name, weights=_floats(weights))

    def use_offsets(self, name: str | None) -> StepReport:
        return self.do("use_offsets", name=name)

    def use_variation(self, name: str | None) -> StepReport:
        return self.do("use_variation", name=name)

    def use_weights(self, name: str | None) -> StepReport:
        return self.do("use_weights", name=name)

    def configure(
        self, settings: PreWhitenSettings | None = None, **updates: Any
    ) -> StepReport:
        """Replace the requested settings (``settings`` and/or field ``updates``)."""
        base = settings if settings is not None else self._head.settings
        try:
            merged = base.model_copy(update=updates) if updates else base
            PreWhitenSettings.model_validate(merged.model_dump())
        except (ValidationError, ValueError) as exc:
            return StepReport(kind="refused", message=str(exc))
        return self.do("configure", settings=merged.model_dump(mode="json"))

    # -- history / persistence ---------------------------------------------------------
    def undo(self) -> bool:
        """Move back one action (or group); ``False`` when nothing is reachable."""
        with self._guard():
            if self._history is None:
                return False
            state = self._history.undo()
            if state is None:
                return False
            self._head = state
            return True

    def redo(self) -> bool:
        """Move forward one action (or group)."""
        with self._guard():
            if self._history is None:
                return False
            state = self._history.redo()
            if state is None:
                return False
            self._head = state
            return True

    def goto(self, seq: int) -> StepReport:
        """Move to the state after log entry ``seq`` (``0`` = root)."""
        with self._guard():
            if self._history is None:
                return StepReport(kind="refused", message="this session keeps no log")
            if self._history.reachable(seq) is False:
                return StepReport(
                    kind="refused",
                    message=(
                        "not in memory: this project was opened from a file; use "
                        "Verify project to replay its log"
                    ),
                )
            state = self._history.goto(seq)
            assert state is not None
            self._head = state
            return StepReport(kind="edited", message=f"at entry {seq}")

    def reachable(self, seq: int) -> bool:
        """Whether the state after entry ``seq`` is still in memory."""
        return self._history is not None and self._history.reachable(seq)

    def snapshot(self, *, with_states: bool = False) -> SessionSnapshot:
        """The session without its data (``with_states``: plus the undo states)."""
        with self._guard():
            history = self._history
            return SessionSnapshot(
                creation=self._kernel.creation,
                head=self._head,
                log=history.effective() if history is not None else (),
                states=history.aligned()
                if with_states and history is not None
                else None,
                saved_serial=self._saved_serial,
            )

    @classmethod
    def restore(
        cls,
        snapshot: SessionSnapshot,
        data: Any,
        *,
        backend: str | None = None,
        labels: Mapping[str, Any] | None = None,
        columns: ColumnMap | None = None,
    ) -> PreWhitenSession:
        """Re-attach a snapshot to its data: states as stored, no replay.

        ``labels`` are the per-point label columns (aligned with ``data``'s rows) and
        ``columns`` the column mapping of a table input, as given to the constructor.
        ``backend=None`` uses the recorded backend when this machine has it, else the
        best CPU backend (with a :class:`SessionWarning`).

        Raises
        ------
        ProjectError
            When the data are not the data the snapshot was taken on.
        """
        creation = snapshot.creation
        if backend is None:
            # The recorded backend when this machine has it, else the best CPU one.
            backend = _delegate("project").choose_backend(creation.backend, None)
        probe = cls(
            data,
            settings=creation.settings,
            grid=creation.grid,
            backend=backend,
            columns=columns,
            t_ref=creation.t_ref,
            labels=labels,
            partitions=creation.partitions,
            session_settings=creation.session_settings,
            record=False,
        )
        if (
            creation.content_sha256
            and probe._kernel.data.content_sha256 != creation.content_sha256
        ):
            raise ProjectError(
                "the data differ from the data this session was created on "
                "(content hash mismatch)"
            )
        kernel = probe._kernel
        kernel.creation = replace(creation, backend=kernel.backend)
        states = [s for s in (snapshot.states or ()) if s is not None]
        kernel.reserve_serials(
            max([snapshot.head.serial, *(s.serial for s in states)])
        )
        kernel.forget()
        history = History.from_snapshot(
            snapshot, depth=kernel.session_settings.undo_depth
        )
        session = cls.__new__(cls)
        session._setup(kernel, snapshot.head, record=True, history=history)
        session._saved_serial = snapshot.saved_serial
        session._columns = columns
        return session

    def _adopt_head(self, state: SessionState, derived: Derived | None = None) -> None:
        """Put ``state`` in place of the head (not an action: nothing is logged).

        A project load re-commits the restored head's statistics this way; the state
        it replaces stays unreachable.
        """
        with self._guard():
            if self._history is not None:
                self._history.replace_head(state)
            if derived is not None:
                self._kernel.remember(state, derived)
            self._head = state
            if self._saved_serial is not None:
                self._saved_serial = state.serial

    def _restamp_last(self, timestamp: str) -> None:
        """Give the newest log entry ``timestamp`` (a replay keeps the logged times)."""
        if self._history is not None:
            self._history.restamp_last(timestamp)

    def save(
        self,
        path: str | Path,
        *,
        embed_data: Literal["auto", "always", "never"] = "auto",
        title: str | None = None,
        ui: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a ``*.cupw.json`` project file (atomically).

        ``embed_data="auto"`` references the data file when it sits in the project's
        directory and still holds exactly these data (and the session has no label
        columns); otherwise the data are embedded. ``title`` and ``ui`` (free-form GUI
        state) are stored as given.
        """
        _delegate("project").save(
            self, path, embed_data=embed_data, title=title, ui=ui
        )
        self._saved_serial = self._head.serial

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        data: Any | None = None,
        verify: bool = False,
        allow_modified_data: bool = False,
        backend: str | None = None,
    ) -> PreWhitenSession:
        """Open a project file at its stored head (see :func:`project.load`).

        ``verify=True`` replays the log first and raises :class:`ProjectError` when
        the stored head does not follow from it.
        """
        session: PreWhitenSession = _delegate("project").load(
            path, data=data, verify=verify, allow_modified_data=allow_modified_data,
            backend=backend,
        )
        return session

    @classmethod
    def replay(
        cls, data: Any, log: Sequence[LogEntry | Mapping[str, Any]], **create: Any
    ) -> PreWhitenSession:
        """A new session on ``data`` with every logged entry applied in order.

        ``create`` are the constructor's keyword arguments. Steps use their logged
        seeds (bitwise on any backend); an entry whose outcome differs from the log
        raises :class:`ProjectError` naming it.
        """
        session: PreWhitenSession = _delegate("project").replay(data, log, **create)
        return session

    def to_script(
        self, *, data_expr: str | None = None, title: str | None = None
    ) -> str:
        """The effective log as a Python script using the public API.

        The script reloads the data from their source file; ``data_expr`` (a Python
        expression, inserted as code) supplies them instead, and is required when
        the data have no reusable source file.
        """
        text: str = _delegate("script").render(self, data_expr=data_expr, title=title)
        return text

    def export_script(self, path: str | Path, *, title: str | None = None) -> Path:
        """Write :meth:`to_script` to ``path`` (and, when needed, its data file)."""
        out: Path = _delegate("script").export(self, path, title=title)
        return out

    def release(self) -> None:
        """Drop device-side engine state (GPU plans); it is rebuilt on demand."""
        with self._guard():
            self._kernel.release_engines()

    def __repr__(self) -> str:
        head = self._head
        return (
            f"PreWhitenSession({len(head.active)} active of {len(head.components)} "
            f"components, n={self._kernel.data.n}, status={head.status!r}, "
            f"backend={self.backend!r})"
        )


__all__ = ["ComponentRef", "PreWhitenSession"]
