"""Application state and the controller that mediates between widgets and compute.

:class:`AppState` is a plain dataclass snapshot of what the app is showing. The
:class:`AppController` is the single hub: widgets call its setters and listen to its
signals; it owns the :class:`~cuperiod.gui.compute.ComputeManager` and the
:class:`~cuperiod.gui.models.ResultCache`, runs (or cache-hits) periodograms *and*
pre-whitening solutions, derives peaks, and tracks the selected period that drives the
phased view. Views never talk to each other — only to the controller — so the data flow
stays one-directional.

The two analyses share every input path (the loaded curve, the source browser, the
folded view) and differ only in what they compute and which result signal they emit, so
switching costs nothing and neither can disturb the other's cached results.

A result is applied only to the run the user is waiting for (the *pending* key), and
only once. Loading another curve supersedes that run: its result is still cached under
its own key (which carries a fingerprint of the data, not just the source name), but
it can never land on the newly loaded curve.

Pre-whitening is interactive (design section 4.2): every pre-whitening run is a
:class:`~cuperiod.prewhiten.PreWhitenSession` owned by the private
:class:`~cuperiod.gui.session_bridge.SessionBridge`, one per source and band. Compute
(Auto) is that session run to its stopping rule, so its numbers are those of
:func:`cuperiod.prewhiten`; Step, Improve and every edit act on the same session, and
nothing the user did to a session is ever dropped by switching source, band or
analysis. The controller caches a finished untouched Auto as a
:class:`~cuperiod.gui.viewmodels.SolutionBundle` (the result *and* its session), so a
revisit shows the very same session again without a re-run.
"""

from __future__ import annotations

import math
import time as _time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, stack_bands
from cuperiod.core.result import Peak, Periodogram, peak_defaults_from_settings
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.dialogs import DialogProvider, DiscardChoice
from cuperiod.gui.fold import epoch_for_peak
from cuperiod.gui.meta import (
    GUI_SAMPLES_PER_PEAK,
    PEAK_ONLY_FIELDS,
    alias_diverse_default,
    auto_max_frequency,
    reaches_short_periods,
    resolved_prewhiten_backend,
    supports_multiband,
)
from cuperiod.gui.models import (
    COMBINED_BAND,
    STACKED_BAND,
    LoadedCurve,
    ResultCache,
    ResultKey,
    SourceItem,
    data_fingerprint,
    settings_hash,
)
from cuperiod.gui.qt import QObject, QtCore, QtGui, Signal
from cuperiod.gui.session_bridge import SessionBridge
from cuperiod.gui.viewmodels import (
    ComponentView,
    SessionKey,
    SessionView,
    SolutionBundle,
)
from cuperiod.prewhiten.engine import default_maximum_frequency
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid

if TYPE_CHECKING:
    from cuperiod.prewhiten.session.session import PreWhitenSession
    from cuperiod.prewhiten.session.state import StepReport
    from cuperiod.prewhiten.subsets import Partition

Mode = Literal["single", "batch"]

#: Which analysis the app is running.
Analysis = Literal["periodogram", "prewhiten"]

#: Registry-style key under which pre-whitening runs are cached.
PREWHITEN_KEY = "PREWHITEN"

#: Denser default frequency oversampling (kept for compatibility; the value lives in
#: :data:`cuperiod.gui.meta.GUI_SAMPLES_PER_PEAK`, which the options form displays).
_DEFAULT_SAMPLES_PER_PEAK = GUI_SAMPLES_PER_PEAK

#: Memory budget of each result cache. An automatic grid scales with baseline times
#: maximum frequency: a multi-year ground-based periodogram is ~36 MB and a
#: pre-whitening result (three spectra) ~90 MB, so a count cap alone let a batch
#: browse retain several GB. The budget keeps ~14 such periodograms or ~5 solutions;
#: anything evicted is simply recomputed on revisit.
CACHE_BYTES: Final = 512 * 1024**2

#: Source ids of curves that came out of a project file: ``"project:<path>"``.
PROJECT_PREFIX: Final = "project:"

#: The project file suffix (spec 2.10) and the file-dialog filters.
PROJECT_SUFFIX: Final = ".cupw.json"
PROJECT_FILTER: Final = "cuPeriod projects (*.cupw.json);;All files (*)"
LIGHT_CURVE_FILTER: Final = (
    "Light curves (*.csv *.ecsv *.txt *.dat *.fits *.fit *.parquet);;All files (*)"
)

#: How long the lazy pre-whitening preview waits after the last trigger (D22).
PREVIEW_DEBOUNCE_MS: Final = 300

#: Resolved spectrum backends that hold no GPU (the lazy preview prefers them).
_CPU_BACKENDS: Final = frozenset({"numpy", "finufft", "torch:cpu"})

#: Report kinds of an attempted step / add_peak: the pick cursor then moves on to the
#: next candidate (design 4.3).
_ATTEMPT_KINDS: Final = frozenset({"accepted", "blocked", "rejected", "stopped"})


@dataclass
class AppState:
    """A snapshot of what the app is currently showing."""

    theme: str = "dark"
    mode: Mode = "single"
    analysis: Analysis = "periodogram"
    method: str = "GLS"
    backend: str = "auto"
    n_peaks: int = 10
    alias_diverse: bool = False
    band: str = ""
    source_id: str = ""
    data_id: str = ""
    current_lc: LoadedCurve | None = None
    current_pg: Periodogram | None = None
    current_solution: PreWhitenResult | None = None
    selected_peak: Peak | None = None
    selected_period: float | None = None
    sources: list[SourceItem] | None = None
    current_source_index: int = 0
    last_compute_ms: float = 0.0
    last_from_cache: bool = False
    # Interactive pre-whitening (design 4.2).
    session_view: SessionView | None = None
    project_path: str | None = None
    selected_components: tuple[int, ...] = ()
    cursor_frequency: float | None = None
    #: Why the displayed session's lazy preview failed ("" when it did not): the
    #: state 4.9 calls "preview failed", in which the session commands are disabled.
    preview_error: str = ""


@dataclass
class _Intent:
    """What the next settled view of a session should do to selection and cursor."""

    reset: bool = False  # the session was (re)displayed: first component, no cursor
    auto_done: bool = False  # Auto completed: first component, cursor on the candidate
    attempt: bool = False  # a step / add_peak ran: cursor on the candidate


@dataclass(frozen=True)
class _Request:
    """The latest pre-whitening request of the Options form (for lazy opens)."""

    backend: str
    settings: PreWhitenSettings
    band: str


@dataclass(frozen=True)
class _Input:
    """What a session of the current curve analyses.

    ``curve`` is the single-band curve (for a stacked multiband input, the centred
    merge, with the per-point band names in ``labels``); ``bands`` is the multiband
    curve a stacked session is built from (``PreWhitenSession.from_bands``).
    """

    curve: LightCurve
    bands: MultiBandLightCurve | None = None
    labels: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Fresh:
    """A fresh Auto handed to ``ComputeManager.submit_prewhiten`` (case 3 of 4.2)."""

    key: SessionKey
    rkey: ResultKey
    data: _Input


@dataclass
class _SaveWait:
    """Sessions being saved before ``then`` may run (Save in a discard prompt)."""

    pending: set[SessionKey]
    then: Callable[[], None]
    names: tuple[str, ...] = field(default=())


class _PartitionRequest(Protocol):
    """What the Subsets dock asks for (``widgets.subsets_panel.PartitionRequest``)."""

    def resolve(
        self,
        time: FloatArray,
        *,
        partitions: Mapping[str, Partition],
        labels: Mapping[str, Any] | None = None,
    ) -> Partition: ...


def _is_empty(lc: LoadedCurve) -> bool:
    """A curve with no point at all (a header-only file, all bands empty)."""
    if isinstance(lc, MultiBandLightCurve):
        return all(band.n == 0 for band in lc.bands.values())
    return lc.n == 0


def _project_file(path: str) -> str:
    """``path`` with the project suffix when the user typed none."""
    return path if path.lower().endswith(".json") else path + PROJECT_SUFFIX


def _describe_loaded(session: PreWhitenSession) -> tuple[LightCurve, PreWhitenSettings]:
    """The curve and requested settings of a loaded project (runs on the worker)."""
    data = session.data
    lc = LightCurve.from_arrays(
        np.array(data.time),
        np.array(data.value),
        None if data.error is None else np.array(data.error),
        domain=data.domain,
    )
    return lc, session.settings


class AppController(QObject):
    """Owns state, compute pool, and cache; the sole mediator between widgets."""

    lc_loaded = Signal(object)  # LoadedCurve
    lc_cleared = Signal()  # no curve is active any more (e.g. a new batch)
    compute_started = Signal()
    periodogram_ready = Signal(object)  # Periodogram
    solution_ready = Signal(object)  # PreWhitenResult
    compute_failed = Signal(str)
    peaks_ready = Signal(object)  # list[Peak]
    period_changed = Signal(float)  # drives the spectrum line
    fold_changed = Signal(float, float)  # (period, t0) drives the phased view
    selection_cleared = Signal()  # no peaks -> clear the spectrum band + phased fold
    busy_changed = Signal(bool)
    sources_changed = Signal(object)  # list[str] labels (batch mode)
    source_selected = Signal(int)
    source_failed = Signal(int)  # index of a batch source whose loader raised
    analysis_changed = Signal(str)  # "periodogram" | "prewhiten"
    # -- interactive pre-whitening (design 4.2) ----------------------------------------
    session_changed = Signal(object)  # SessionView of the displayed session
    session_progress = Signal(object)  # StepReport, during Auto / Step
    session_message = Signal(str, str)  # (severity, text): refusals, warnings, notices
    session_busy = Signal(bool)  # a quick job ran > 300 ms (non-blocking indicator)
    variations_ready = Signal(object)  # VariationResult
    components_selected = Signal(object)  # tuple[int, ...], current first
    cursor_changed = Signal(float)  # pick cursor (NaN hides it); never moves the fold
    project_changed = Signal(str, bool)  # (project path or "", dirty)
    # Additive (not in the 4.2 listing).
    status_message = Signal(str)  # one-line notices for the status bar
    preview_failed = Signal(str)  # "Spectrum unavailable: <reason>" placeholder text
    settings_restored = Signal(object)  # PreWhitenSettings of an opened project
    entry_script_ready = Signal(int, str)  # History "Copy as Python": (seq, text)
    close_approved = Signal()  # the user resolved the unsaved sessions; close again

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        dialogs: DialogProvider | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = AppState()
        self._cache: ResultCache[Periodogram] = ResultCache(max_bytes=CACHE_BYTES)
        self._solutions: ResultCache[SolutionBundle] = ResultCache(
            maxsize=32, max_bytes=CACHE_BYTES
        )
        self._compute = ComputeManager(self)
        self._pending_key: ResultKey | None = None
        # The key a worker is executing right now (from the task's ``started``
        # signal, not from submit: a queued task that gets superseded never starts).
        self._running_key: ResultKey | None = None
        # Peak-selection defaults of the pending periodogram run (its settings'
        # separation / alias / harmonic fields), applied to the result on delivery.
        self._peak_defaults: dict[str, Any] = {}
        # The peaks on show, so a spectrum-marker click (which carries only a
        # period) can recover the peak and its transit epoch.
        self._peaks: list[Peak] = []
        self._compute.signals.started.connect(self._on_started)
        self._compute.signals.finished.connect(self._on_finished)
        self._compute.signals.solution_ready.connect(self._on_solution)
        self._compute.signals.failed.connect(self._on_failed)
        # -- sessions --------------------------------------------------------------
        self._dialogs = dialogs
        # What the user knows each source as (source id -> label), for prompts: a
        # demo's key is "asassn_446677590359", its label the star's name.
        self._labels: dict[str, str] = {}
        self._bridge = SessionBridge(self._compute, parent=self)
        self._compute.set_session_runner(self._run_fresh)
        self._shown: SessionKey | None = None
        self._intents: dict[SessionKey, _Intent] = {}
        # (settings hash, requested backend) each live session was opened with, as
        # far as the GUI thread knows (the settled view says it exactly).
        self._session_sig: dict[SessionKey, tuple[str, str]] = {}
        self._auto_inflight: dict[SessionKey, ResultKey] = {}
        self._auto_t0: dict[SessionKey, float] = {}
        self._fresh: _Fresh | None = None
        self._request: _Request | None = None
        self._gpu_used = False
        self._loading: dict[SessionKey, str] = {}
        self._verifying: set[SessionKey] = set()
        self._save_waits: list[_SaveWait] = []
        self._project_state: tuple[str, bool] = ("", False)
        self._close_approved = False
        self._closing_sync = False
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._on_preview_timer)
        bridge = self._bridge
        bridge.view_ready.connect(self._on_view)
        bridge.progress.connect(self._on_progress)
        bridge.message.connect(self.session_message)
        bridge.failed.connect(self._on_session_failed)
        bridge.preview_failed.connect(self._on_preview_failed)
        bridge.busy.connect(self.busy_changed)
        bridge.quick_busy.connect(self.session_busy)
        bridge.auto_finished.connect(self._on_auto_finished)
        bridge.finalized.connect(self._on_finalized)
        bridge.query_ready.connect(self._on_query)
        bridge.reported.connect(self._on_reported)
        bridge.saved.connect(self._on_saved)
        bridge.loaded.connect(self._on_loaded)
        bridge.settled.connect(self._on_settled)

    # -- dialogs -----------------------------------------------------------------------
    @property
    def dialogs(self) -> DialogProvider | None:
        """The provider every prompt goes through (``None``: nothing is asked)."""
        return self._dialogs

    def set_dialog_provider(self, provider: DialogProvider | None) -> None:
        """Route the controller's prompts (unsaved sessions, files) to ``provider``."""
        self._dialogs = provider

    # -- inputs ------------------------------------------------------------------------
    def set_light_curve(
        self, lc: LoadedCurve, source_id: str, *, label: str = ""
    ) -> None:
        """Set the active light curve (clears the previous spectrum/selection).

        Any run still pending is superseded *before* ``lc_loaded`` is emitted: its
        result is cached under its own key but never applied to this curve, and the
        busy state is released so Compute is usable at once. The order matters, as
        batch mode starts the next run from an ``lc_loaded`` slot.

        In pre-whitening the curve's session (kept by the bridge, e.g. a dirty one
        from an earlier visit) is displayed again right after ``lc_loaded``; a session
        never seen before opens lazily (D22).

        ``label`` is what the user knows the source as (a demo's or batch source's
        name rather than its key); prompts and File ▸ Unsaved sessions use it.
        """
        if label:
            self._labels[source_id] = label
        self.state.current_lc = lc
        self.state.source_id = source_id
        self.state.data_id = data_fingerprint(lc)
        self._clear_results()
        self._supersede_pending()
        self.lc_loaded.emit(lc)
        self._sync_session_display(force=True)

    def clear_light_curve(self) -> None:
        """Drop the active curve and its results; emits ``lc_cleared``."""
        self.state.current_lc = None
        self.state.source_id = ""
        self.state.data_id = ""
        self._clear_results()
        self._supersede_pending()
        self.lc_cleared.emit()
        self._show(None)

    def _clear_results(self) -> None:
        self.state.current_pg = None
        self.state.current_solution = None
        self.state.selected_peak = None
        self.state.selected_period = None
        self.state.selected_components = ()
        self.state.cursor_frequency = None
        # Before lc_loaded goes out: the window syncs its commands from it there.
        self.state.preview_error = ""
        self._peaks = []

    def _supersede_pending(self) -> None:
        """Forget the pending run and release the busy state.

        Queued tasks are cancelled (a running one cannot be; it completes and is
        cached under its own key). No completion handler fires for a forgotten run,
        so the busy state must be released here or Compute would stay disabled.
        Session work is not touched: it belongs to its session (D15).
        """
        self._pending_key = None
        self._compute.cancel_all()
        self.busy_changed.emit(False)

    def set_analysis(self, analysis: Analysis) -> None:
        """Switch between the periodogram and pre-whitening analyses.

        Only the *pending* run is superseded; both caches survive, so flipping back to a
        previously computed analysis of the same source redisplays it instantly. A
        pre-whitening session stays alive (with whatever is still queued for it) and
        is displayed again on the way back.
        """
        if analysis == self.state.analysis:
            return
        self.state.analysis = analysis
        self._pending_key = None
        self._peaks = []
        # Dropping the pending key means no completion handler will ever fire for the
        # in-flight run, so the busy state has to be released here or the Compute button
        # stays disabled for the rest of the session.
        self.busy_changed.emit(False)
        self.analysis_changed.emit(analysis)
        self._sync_session_display(force=False)

    # -- batch mode --------------------------------------------------------------------
    def load_sources(self, items: list[SourceItem]) -> None:
        """Enter batch mode with ``items`` (labels via ``sources_changed``).

        The previous curve, its results and its source index are dropped first
        (``lc_cleared``): if the new batch's first source then fails to load, nothing
        of the old batch may stay on screen or be analysed under the new batch's name.
        """
        self.state.sources = list(items)
        self.state.mode = "batch"
        self.state.current_source_index = 0
        self.clear_light_curve()
        self.sources_changed.emit([item.label for item in items])

    def exit_batch_mode(self) -> None:
        """Leave batch mode (the active curve, if any, stays loaded)."""
        self.state.mode = "single"
        self.state.sources = None
        self.state.current_source_index = 0

    def select_source(self, index: int) -> None:
        """Load and activate source ``index`` (the light curve, then auto-compute).

        A source whose curve has no point at all takes the failure path too: there is
        nothing to show or analyse, and saying so beats an empty plot.
        """
        sources = self.state.sources
        if not sources:
            return
        index = max(0, min(index, len(sources) - 1))
        previous_index = self.state.current_source_index
        item = sources[index]
        try:
            lc = item.loader()
        except Exception as exc:  # noqa: BLE001 - surface load failures to the UI
            self._source_failed(
                index, previous_index, f"Could not load {item.label}: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        if _is_empty(lc):
            self._source_failed(
                index, previous_index, f"{item.label} contains no data rows."
            )
            return
        self.state.current_source_index = index
        self.source_selected.emit(index)
        self.set_light_curve(lc, item.key, label=item.label)

    def _source_failed(self, index: int, previous_index: int, message: str) -> None:
        # Leave the state on the previously-loaded source and snap the browser
        # highlight back, so a failed selection doesn't strand the UI showing one
        # source's plots while a different (failed) source stays highlighted.
        # ``source_failed`` names the failed row, so the browser can mark it and
        # step over it instead of retrying it on every Next.
        self.state.current_source_index = previous_index
        self.source_failed.emit(index)
        self.source_selected.emit(previous_index)
        self.compute_failed.emit(message)

    def run(
        self,
        method: str,
        backend: str,
        settings: BaseSettings,
        n_peaks: int,
        band: str = "",
    ) -> None:
        """Compute (or cache-hit) the periodogram for the active light curve.

        ``band`` selects a multiband input: a band name for a single band,
        :data:`~cuperiod.gui.models.STACKED_BAND` to merge bands, or
        :data:`~cuperiod.gui.models.COMBINED_BAND` (or ``""``) to use the whole
        multiband curve (for multiband-capable methods; otherwise it falls back to a
        stacked single curve). The legacy strings ``"stacked"``/``"combined"`` still
        work when no real band carries that name.

        The peak-selection fields of ``settings`` (separation, alias tolerance,
        harmonic order) are applied to the finished spectrum and are not part of the
        cache key, so changing one re-selects peaks without a recompute; ``n_peaks``
        (the Peaks spin) overrides ``settings.n_peaks``.
        """
        lc_input = self._resolve_input(method, band)
        if lc_input is None:
            return
        settings = self._tune_auto_grid(lc_input, settings)
        self.state.method = method
        self.state.backend = backend
        self.state.n_peaks = n_peaks
        self.state.band = band
        self.state.alias_diverse = alias_diverse_default(method)
        # The Peaks spin, not settings.n_peaks, is the peak count the views ask for.
        self._peak_defaults = {**peak_defaults_from_settings(settings), "n": n_peaks}
        key = ResultKey(
            self._source_tag(band),
            method,
            settings_hash(settings, exclude=PEAK_ONLY_FIELDS),
            backend,
            self.state.data_id,
        )
        self._pending_key = key
        cached = self._cache.get(key)
        if cached is not None:
            self._apply_result(key, cached, 0.0, from_cache=True)
            return
        self.busy_changed.emit(True)
        self.compute_started.emit()
        if self._join_running(key):
            return
        self._compute.submit(key, lc_input, method, settings, backend)

    def run_prewhiten(
        self, backend: str, settings: PreWhitenSettings, band: str = ""
    ) -> None:
        """Auto: extract (or cache-hit) the frequency solution of the active curve.

        Pre-whitening is single-band by definition, so a multiband curve is reduced the
        same way a single-band method does: the selected band, or all bands stacked
        (each centred on its own mean). The spectra are always stored: the GUI draws
        them, and a solution without them would leave an empty plot.

        The three cases of design 4.2: a session the user has edited (not pristine)
        is continued — configured to the form, run to its stopping rule and finalized
        (a backend change is refused); else a cached solution is shown at once with
        its exact session; else a fresh session runs, exactly :func:`cup.prewhiten`
        (handed to ``ComputeManager.submit_prewhiten``, as the one-shot run was).
        """
        if self.state.current_lc is None:
            return
        data = self._input(band)
        tuned = self._tuned(data, settings)
        self.state.backend = backend
        self.state.band = band
        self.state.method = PREWHITEN_KEY
        self._request = _Request(backend, settings, band)
        self._preview_timer.stop()
        rkey = ResultKey(
            self._source_tag(band),
            PREWHITEN_KEY,
            settings_hash(tuned),
            backend,
            self.state.data_id,
        )
        key = self._session_key(band)
        self._show(key)
        if self._auto_inflight.get(key) == rkey:
            return  # the very same Auto is queued or running: wait for it
        bridge = self._bridge
        if bridge.is_open(key) and not bridge.is_pristine(key):
            self._start_auto(key, tuned, backend, rkey)  # case 1: continue
            return
        bundle = self._solutions.get(rkey)
        if bundle is not None:
            self._show_cached(key, rkey, bundle, data)  # case 2
            return
        if bridge.is_open(key) and self._live_sig(key) == (rkey.settings_hash, backend):
            # Pristine with these very settings (stepped, or a preview): stepping on
            # is what a restart would compute, so keep its history.
            self._start_auto(key, tuned, backend, rkey)
            return
        # case 3: a fresh run (replaces a pristine live session)
        self._fresh = _Fresh(key, rkey, data)
        self._compute.submit_prewhiten(rkey, data.curve, tuned, backend)

    def _join_running(self, key: ResultKey) -> bool:
        """Wait for the identical run already executing instead of queueing a copy.

        Returning to a source before its first run has been delivered used to queue a
        duplicate that recomputed the same result and re-applied it later, resetting
        the user's peak selection. Anything queued behind the running task is
        cancelled, exactly as a fresh submit would supersede it.
        """
        if key != self._running_key:
            return False
        self._compute.cancel_all()
        return True

    def _source_tag(self, band: str) -> str:
        """The cache key's source part: the source id, qualified by the band choice."""
        return self.state.source_id + (f"::{band}" if band else "")

    def _resolve_input(self, method: str, band: str) -> LoadedCurve | None:
        """Pick the curve to analyse from the (possibly multiband) active curve.

        The all-bands tokens are tested *before* the band names, so a real band named
        ``combined`` or ``stacked`` cannot hijack the all-bands choice.
        """
        lc = self.state.current_lc
        if not isinstance(lc, MultiBandLightCurve):
            return lc
        if band == COMBINED_BAND:
            return lc if supports_multiband(method) else self._stack(lc)
        if band == STACKED_BAND:
            return self._stack(lc)
        if band in lc.bands:
            return lc.bands[band]
        if band == "stacked" or not supports_multiband(method):
            return self._stack(lc)
        return lc

    @staticmethod
    def _stack(mblc: MultiBandLightCurve) -> LightCurve:
        """Merge all bands into one time-sorted single-band curve."""
        return stack_bands(mblc, centre=False)[0]

    @staticmethod
    def _stack_centred(mblc: MultiBandLightCurve) -> LightCurve:
        """Merge all bands after removing each band's own mean (``stack_bands``)."""
        return stack_bands(mblc, centre=True)[0]

    def _tune_auto_grid(self, lc: LoadedCurve, settings: BaseSettings) -> BaseSettings:
        """Improve the default frequency grid for frequency-grid methods.

        (a) Raises an auto max-frequency so sub-day periods aren't missed on sparse data
        (see :func:`cuperiod.gui.meta.auto_max_frequency`; the frequency-domain
        analyses — GLS, MHAOV, pre-whitening — use the engine's
        :func:`~cuperiod.prewhiten.default_maximum_frequency`, whose higher floor keeps
        δ Scuti / HADS frequencies in band) and (b) densifies ``samples_per_peak`` to
        :data:`~cuperiod.gui.meta.GUI_SAMPLES_PER_PEAK` when the caller left it unset
        (a programmatic ``GLSSettings()``). A value that was *set* -- by the options
        form, which shows the GUI default itself, or by an environment override -- is
        used exactly as given, including a deliberate 5.
        """
        fields = type(settings).model_fields
        updates: dict[str, Any] = {}
        time = self._time_of(lc)
        if (
            "maximum_frequency" in fields
            and getattr(settings, "maximum_frequency", None) is None
            and time.size >= 2
        ):
            nyquist_factor = int(getattr(settings, "nyquist_factor", 5))
            auto = (
                default_maximum_frequency(time, nyquist_factor)
                if reaches_short_periods(settings)
                else auto_max_frequency(time, nyquist_factor)
            )
            updates["maximum_frequency"] = auto
        if (
            "samples_per_peak" in fields
            and "samples_per_peak" not in settings.model_fields_set
        ):
            current = getattr(settings, "samples_per_peak", None)
            if current is not None and current < GUI_SAMPLES_PER_PEAK:
                updates["samples_per_peak"] = GUI_SAMPLES_PER_PEAK
        return settings.model_copy(update=updates) if updates else settings

    @staticmethod
    def _time_of(lc: LoadedCurve | None) -> FloatArray:
        """Concatenated *finite* time array for a single- or multi-band curve.

        Non-finite times (FITS gap cadences) are left out, so the fold epoch, the
        baseline and the auto-grid hint derived from it are all finite.
        """
        if isinstance(lc, MultiBandLightCurve):
            times = [band.time for band in lc.bands.values() if band.n]
            time = np.concatenate(times) if times else np.empty(0, dtype=np.float64)
        elif isinstance(lc, LightCurve):
            time = lc.time
        else:
            return np.empty(0, dtype=np.float64)
        finite = np.isfinite(time)
        return time if bool(finite.all()) else time[finite]

    # -- compute results ---------------------------------------------------------------
    def _on_started(self, key: ResultKey) -> None:
        self._running_key = key

    def _task_done(self, key: ResultKey) -> None:
        if key == self._running_key:
            self._running_key = None

    def _on_finished(self, key: ResultKey, pg: Periodogram, elapsed_ms: float) -> None:
        # Cache even superseded results so revisiting is instant, but only apply the
        # one the user is currently waiting for.
        self._task_done(key)
        self._cache.put(key, pg)
        if key == self._pending_key:
            self._apply_result(key, pg, elapsed_ms, from_cache=False)

    def _on_solution(
        self, key: ResultKey, result: PreWhitenResult, elapsed_ms: float
    ) -> None:
        # A one-shot PreWhitenTask (only when no session runner is attached): it has
        # no session, so there is no bundle to cache.
        self._task_done(key)
        if key == self._pending_key:
            self._apply_solution(result, elapsed_ms, from_cache=False)

    def _on_failed(self, key: ResultKey, message: str) -> None:
        self._task_done(key)
        if key == self._pending_key:
            self._pending_key = None
            self.busy_changed.emit(False)
            self.compute_failed.emit(message)

    def _apply_solution(
        self,
        result: PreWhitenResult,
        elapsed_ms: float,
        *,
        from_cache: bool,
        busy: bool | None = False,
    ) -> None:
        """Show ``result`` as the solution (v1.3): the first component is selected.

        ``busy`` is the busy state to announce; ``None`` leaves it to the session
        bridge (an Auto job reports its own end).
        """
        self._pending_key = None  # applied once: a late duplicate is only cached
        self._peaks = []
        self.state.current_solution = result
        self.state.current_pg = None
        self.state.method = PREWHITEN_KEY
        self.state.last_compute_ms = elapsed_ms
        self.state.last_from_cache = from_cache
        if busy is not None:
            self.busy_changed.emit(busy)
        self.solution_ready.emit(result)
        if result.components:
            self.select_component(result.components[0])
        else:
            self.state.selected_peak = None
            self.state.selected_period = None
            self.selection_cleared.emit()

    def select_component(self, component: Sinusoid) -> None:
        """Select an extracted sinusoid as the active period (drives the fold)."""
        self.state.selected_peak = None
        self.select_period(component.period)

    def _apply_result(
        self, key: ResultKey, pg: Periodogram, elapsed_ms: float, *, from_cache: bool
    ) -> None:
        self._pending_key = None  # applied once: a late duplicate is only cached
        # The run's own peak-selection settings, not those the spectrum was cached
        # with (they are not part of the cache key).
        pg = pg.with_peak_defaults(self._peak_defaults)
        self.state.current_pg = pg
        self.state.last_compute_ms = elapsed_ms
        self.state.last_from_cache = from_cache
        self.busy_changed.emit(False)
        self.periodogram_ready.emit(pg)
        peaks = pg.best_periods(
            self.state.n_peaks, alias_diverse=self.state.alias_diverse
        )
        self._peaks = list(peaks)
        self.peaks_ready.emit(peaks)
        if peaks:
            self.select_peak(peaks[0])
        else:
            self.state.selected_peak = None
            self.state.selected_period = None
            self.selection_cleared.emit()

    # -- selection ---------------------------------------------------------------------
    def select_peak(self, peak: Peak) -> None:
        """Select ``peak`` as the active period (drives the line and phased view)."""
        self.state.selected_peak = peak
        self.state.selected_period = peak.period
        epoch = epoch_for_peak(peak, self._lc_time())
        self.period_changed.emit(peak.period)
        self.fold_changed.emit(peak.period, epoch)

    def select_period(self, period: float) -> None:
        """Select an arbitrary ``period`` (e.g. from dragging the spectrum line).

        A period that is exactly one of the listed peaks' (a spectrum-marker click,
        or a drag that snapped onto a peak's grid sample) selects that peak, so a box
        search's transit epoch is kept just as when the peak is picked in the table.
        """
        for peak in self._peaks:
            if peak.period == period:
                self.select_peak(peak)
                return
        self.state.selected_peak = None
        self.state.selected_period = period
        epoch = epoch_for_peak(None, self._lc_time())
        self.period_changed.emit(period)
        self.fold_changed.emit(period, epoch)

    def current_time(self) -> FloatArray:
        """Finite times of the active curve (bands concatenated); empty if none."""
        return self._time_of(self.state.current_lc)

    # -- one-shot Stop -----------------------------------------------------------------
    @property
    def finishing(self) -> bool:
        """A stopped one-shot run still occupies the worker ("finishing…")."""
        return self._compute.finishing

    def cancel_compute(self) -> None:
        """Stop: the displayed session's long job, else the one-shot run (Esc).

        A session job stops after its current step and keeps what it committed. A
        one-shot periodogram cannot be interrupted, so it is superseded: the window is
        released at once and the run finishes in the background (its result is cached
        but not shown). Running the same configuration again joins it.
        """
        key = self._shown
        if key is not None and self._bridge.cancel(key):
            self.status_message.emit(
                "Stopping after the current step; what was found is kept."
            )
            return
        pending = self._pending_key
        self._pending_key = None
        stopped = self._compute.cancel_running()
        self.busy_changed.emit(False)
        if pending is not None or stopped is not None:
            self.status_message.emit(
                "Stopped. The running computation finishes in the background; its "
                "result is cached but not shown."
            )

    # -- lifecycle ---------------------------------------------------------------------
    def shutdown(self) -> None:
        """Cancel pending compute; wait briefly only if nothing is running.

        Called from the window's ``closeEvent``. A task that has started cannot be
        interrupted, so waiting for it would only freeze the closing window; queued
        tasks are cancelled and an idle pool drains at once. (The console entry point
        ends the process if a worker is still running when the event loop returns.)
        A long session job is asked to stop after its current step.
        """
        self._preview_timer.stop()
        for key in tuple(self._bridge.keys()):
            self._bridge.cancel(key)
        self._compute.cancel_all()
        if not self._compute.active:
            self._compute.wait_for_done(2000)

    def _lc_time(self) -> FloatArray:
        """Finite times of the active curve (bands concatenated), for the epoch."""
        return self._time_of(self.state.current_lc)

    # =================================================================================
    # Interactive pre-whitening (design 4.2)
    # =================================================================================
    # -- keys and inputs ---------------------------------------------------------------
    def _band_token(self, band: str) -> str:
        """The band a session of the current curve analyses ("" for a single band)."""
        lc = self.state.current_lc
        if not isinstance(lc, MultiBandLightCurve):
            return ""
        if band not in (COMBINED_BAND, STACKED_BAND) and band in lc.bands:
            return band
        return STACKED_BAND

    def _session_key(self, band: str) -> SessionKey:
        """The session key of the current curve for ``band``.

        A project's curve is identified by its path (the bridge keyed its load
        before the curve existed on this thread); any other source by its name and
        content fingerprint.
        """
        source = self.state.source_id
        data_id = "" if source.startswith(PROJECT_PREFIX) else self.state.data_id
        return SessionKey(source, self._band_token(band), data_id)

    def _input(self, band: str) -> _Input:
        """What a session of the current curve analyses for ``band``."""
        lc = self.state.current_lc
        if isinstance(lc, MultiBandLightCurve):
            token = self._band_token(band)
            if token != STACKED_BAND:
                return _Input(lc.bands[token])
            merged, names = stack_bands(lc, centre=True)
            return _Input(merged, bands=lc, labels={"band": names})
        assert isinstance(lc, LightCurve)
        return _Input(lc)

    def _tuned(self, data: _Input, settings: PreWhitenSettings) -> PreWhitenSettings:
        """The settings a session runs with: spectra kept, the GUI's auto grid."""
        if not settings.store_spectra:
            settings = settings.model_copy(update={"store_spectra": True})
        tuned = self._tune_auto_grid(data.curve, settings)
        assert isinstance(tuned, PreWhitenSettings)
        return tuned

    @staticmethod
    def _factory(
        data: _Input, settings: PreWhitenSettings, backend: str
    ) -> Callable[[], PreWhitenSession]:
        """A worker-side constructor of a new session on ``data``."""

        def make() -> PreWhitenSession:
            from cuperiod.prewhiten.session.session import PreWhitenSession

            if data.bands is not None:
                return PreWhitenSession.from_bands(
                    data.bands, centre=True, settings=settings, backend=backend
                )
            return PreWhitenSession(data.curve, settings=settings, backend=backend)

        return make

    def _lazy_backend(self, requested: str, settings: PreWhitenSettings) -> str:
        """The backend of a lazily opened session (D22): a CPU one unless a GPU
        session already ran in this application run, so browsing creates no CUDA
        context."""
        if self._gpu_used:
            return requested
        resolved = resolved_prewhiten_backend(requested, settings)
        if resolved is None or not resolved[1]:
            return requested
        cpu = resolved_prewhiten_backend("cpu", settings)
        return cpu[0] if cpu is not None else "numpy"

    def _live_sig(self, key: SessionKey) -> tuple[str, str] | None:
        """``(settings hash, requested backend)`` of ``key``'s live session.

        The settled view says it exactly; while jobs are queued, the record of the
        last open stands in.
        """
        view = self._bridge.latest_view(key)
        if (
            self._bridge.is_idle(key)
            and view is not None
            and view.settings is not None
            and view.requested_backend
        ):
            return settings_hash(view.settings), view.requested_backend
        return self._session_sig.get(key)

    # -- display -----------------------------------------------------------------------
    def _sync_session_display(self, *, force: bool) -> None:
        """Display the current curve's session in pre-whitening, nothing otherwise."""
        if self.state.analysis == "prewhiten" and self.state.current_lc is not None:
            self._show(self._session_key(self.state.band), force=force)
            self._arm_preview()
        else:
            self._preview_timer.stop()
            self._show(None)

    def _show(self, key: SessionKey | None, *, force: bool = False) -> None:
        """Make ``key`` the displayed session (``force``: re-emit its view)."""
        if key == self._shown and not force:
            return
        self._shown = key
        self.state.session_view = None
        self.state.preview_error = ""  # a failure belonged to what was on show
        if key is None:
            self.state.selected_components = ()
        else:
            self._intents.setdefault(key, _Intent()).reset = True
        self._bridge.display(key)  # emits the kept view of a reattached session
        if self._bridge.is_busy:
            self.busy_changed.emit(True)
        self._emit_project()

    def display_session(self, band: str = "") -> None:
        """Show the current curve's session for ``band`` (the band picker moved)."""
        self.state.band = band
        self._sync_session_display(force=False)

    @property
    def displayed_session(self) -> SessionKey | None:
        """The key of the session on show (``None`` outside pre-whitening)."""
        return self._shown

    @property
    def session_bridge(self) -> SessionBridge:
        """The bridge (tests and diagnostics; widgets go through the controller)."""
        return self._bridge

    def session_name(self, key: SessionKey) -> str:
        """A short human name of ``key`` (prompts, File ▸ Unsaved sessions).

        The source's label when it was loaded with one (a demo or batch source),
        else its file name; the band follows in brackets.
        """
        name = self._labels.get(key.source_id) or self._source_file_name(key)
        band = "all bands" if key.band == STACKED_BAND else key.band
        return f"{name} [{band}]" if band else name

    @staticmethod
    def _source_file_name(key: SessionKey) -> str:
        """The file name part of ``key``'s source id (a project's prefix dropped)."""
        source = key.source_id
        if source.startswith(PROJECT_PREFIX):
            source = source[len(PROJECT_PREFIX) :]
        return Path(source).name or source

    def dirty_sessions(self) -> tuple[tuple[SessionKey, str], ...]:
        """The sessions with unsaved changes, with their names (File ▸ Sessions)."""
        return tuple((k, self.session_name(k)) for k in self._bridge.dirty_keys())

    def is_session_dirty(self, key: SessionKey | None = None) -> bool:
        """Whether ``key`` (default: the displayed session) has unsaved changes."""
        key = self._shown if key is None else key
        return key is not None and self._bridge.is_dirty(key)

    def activate_session(self, key: SessionKey) -> bool:
        """Show a kept session again (File ▸ Sessions); False when its curve is gone.

        The session lives in the bridge but its curve must be the current one, so a
        batch source is selected by its key; a curve opened from a file has to be
        opened again (its session then reattaches by itself).
        """
        if key.source_id == self.state.source_id:
            self.set_analysis("prewhiten")
            self.display_session(key.band)
            return True
        for index, item in enumerate(self.state.sources or ()):
            if item.key == key.source_id:
                self.set_analysis("prewhiten")
                self.state.band = key.band
                self.select_source(index)
                return True
        self.status_message.emit(
            f"Open {self.session_name(key)} again to continue its session."
        )
        return False

    # -- lazy open (D22) ---------------------------------------------------------------
    def schedule_preview(
        self, backend: str, settings: PreWhitenSettings, band: str = ""
    ) -> None:
        """Remember the form's request and open the displayed session after a pause.

        Only in single mode and in pre-whitening; the session opens with no step, on
        a CPU backend unless a GPU session already ran, and a failure only shows
        placeholder text (``preview_failed``).
        """
        self._request = _Request(backend, settings, band)
        if band != self.state.band and self.state.analysis == "prewhiten":
            self.display_session(band)
        else:
            self._arm_preview()

    def _arm_preview(self) -> None:
        key = self._shown
        if (
            self._request is None
            or key is None
            or self.state.analysis != "prewhiten"
            or self.state.mode == "batch"
            or self.state.current_lc is None
            or self._bridge.is_open(key)
        ):
            self._preview_timer.stop()
            return
        self._preview_timer.start()

    def _on_preview_timer(self) -> None:
        key = self._shown
        request = self._request
        if (
            key is None
            or request is None
            or self.state.analysis != "prewhiten"
            or self.state.mode == "batch"
            or self.state.current_lc is None
            or self._bridge.is_open(key)
        ):
            return
        self._open_session(key, request, preview=True)

    def _open_session(
        self, key: SessionKey, request: _Request, *, preview: bool, lazy: bool = True
    ) -> None:
        """Queue the opening of ``key``'s session (no step) from ``request``."""
        data = self._input(self.state.band)
        settings = self._tuned(data, request.settings)
        backend = (
            self._lazy_backend(request.backend, settings) if lazy else request.backend
        )
        self._session_sig[key] = (settings_hash(settings), backend)
        factory = self._factory(data, settings, backend)
        if key == self._shown:
            self.state.preview_error = ""  # a new attempt; it says again if it fails
        self._bridge.open(key, factory, preview=preview, backend=backend)

    def _on_preview_failed(self, key: object, reason: str) -> None:
        if key == self._shown:
            text = f"Spectrum unavailable: {reason}"
            self.state.preview_error = text
            self.preview_failed.emit(text)

    # -- the addressed session ---------------------------------------------------------
    def _active_key(self) -> SessionKey | None:
        """The session the user's action addresses (opened lazily on first use)."""
        key = self._shown
        if (
            key is None
            and self.state.current_lc is not None
            and self.state.analysis == "prewhiten"
        ):
            key = self._session_key(self.state.band)
            self._show(key)
        if key is None:
            self.session_message.emit("warning", "No pre-whitening session is open.")
            return None
        if not self._bridge.is_open(key):
            if self.state.current_lc is None:
                self.session_message.emit(
                    "warning", "No pre-whitening session is open."
                )
                return None
            self._preview_timer.stop()
            request = self._request or _Request(
                "auto", PreWhitenSettings(), self.state.band
            )
            self._open_session(key, request, preview=False)
        return key

    # -- Auto / Step / Improve ---------------------------------------------------------
    def _start_auto(
        self,
        key: SessionKey,
        settings: PreWhitenSettings,
        backend: str,
        rkey: ResultKey,
        *,
        factory: Callable[[], PreWhitenSession] | None = None,
        restart: bool = False,
    ) -> None:
        self._auto_inflight[key] = rkey
        self._auto_t0[key] = _time.perf_counter()
        self._bridge.auto(
            key, settings, backend, factory=factory, restart=restart, tag=rkey
        )
        if key == self._shown:
            self.compute_started.emit()

    def _run_fresh(
        self,
        rkey: ResultKey,
        lc: LoadedCurve,
        settings: PreWhitenSettings,
        backend: str,
    ) -> None:
        """``ComputeManager.submit_prewhiten`` as a fresh session's Auto (case 3)."""
        fresh, self._fresh = self._fresh, None
        if fresh is not None and fresh.rkey == rkey:
            key, data = fresh.key, fresh.data
        else:  # submitted by someone else: a plain session on what was given
            data = (
                _Input(lc)
                if isinstance(lc, LightCurve)
                else _Input(*self._stacked(lc))
            )
            key = SessionKey(rkey.source_id, "", rkey.data_id)
        self._session_sig[key] = (rkey.settings_hash, backend)
        self._start_auto(
            key,
            settings,
            backend,
            rkey,
            factory=self._factory(data, settings, backend),
            restart=True,
        )

    @staticmethod
    def _stacked(
        mblc: MultiBandLightCurve,
    ) -> tuple[LightCurve, MultiBandLightCurve, Mapping[str, Any]]:
        merged, names = stack_bands(mblc, centre=True)
        return merged, mblc, {"band": names}

    def _show_cached(
        self, key: SessionKey, rkey: ResultKey, bundle: SolutionBundle, data: _Input
    ) -> None:
        """Case 2: the cached result at once, and the session it came from (D20)."""
        self._apply_solution(
            bundle.result, 0.0, from_cache=True, busy=self._bridge.is_busy
        )
        self._intents.setdefault(key, _Intent()).auto_done = True
        bridge = self._bridge
        view = bridge.latest_view(key)
        live = (
            bridge.is_open(key)
            and bridge.is_idle(key)
            and bridge.is_pristine(key)
            and view is not None
            and view.serial == bundle.snapshot.head.serial
            and self._live_sig(key) == (rkey.settings_hash, rkey.backend)
        )
        if live:
            if self.state.session_view is not view:
                bridge.display(key)  # re-emits it: the selection follows the solution
            return
        self._session_sig[key] = (rkey.settings_hash, rkey.backend)
        bridge.restore(
            key,
            bundle.snapshot,
            data.curve,
            labels=data.labels,
            requested_backend=rkey.backend,
        )

    def session_step(
        self, backend: str, settings: PreWhitenSettings, band: str = ""
    ) -> None:
        """Step: one iteration of the extraction on the displayed session (D17).

        Never restarts: a session with an empty log is re-opened with the form's
        settings and backend; otherwise a different form is applied by ``configure``
        (an edit) and a backend change is refused.
        """
        if self.state.current_lc is None:
            return
        self._prepare_attempt(backend, settings, band)
        data = self._input(band)
        tuned = self._tuned(data, settings)
        key = self._session_key(band)
        sig = self._live_sig(key)
        manual = sig is not None and sig[0] != settings_hash(tuned)
        if not self._bridge.is_open(key):
            self._session_sig[key] = (settings_hash(tuned), backend)
        self._bridge.step(
            key,
            tuned,
            backend,
            factory=self._factory(data, tuned, backend),
            manual=manual,
        )

    def session_improve(
        self,
        backend: str,
        settings: PreWhitenSettings,
        band: str = "",
        *,
        ids: Sequence[int] | None = None,
        expect_id_epoch: int | None = None,
    ) -> None:
        """Improve all (``ids=None``) or the given components, after a configure to
        the form when it differs (D17: Improve never restarts)."""
        if self.state.current_lc is None:
            return
        self._prepare_attempt(backend, settings, band)
        data = self._input(band)
        tuned = self._tuned(data, settings)
        key = self._session_key(band)
        if not self._bridge.is_open(key):
            self._session_sig[key] = (settings_hash(tuned), backend)
        self._bridge.improve(
            key,
            tuned,
            backend,
            ids=ids,
            factory=self._factory(data, tuned, backend),
            expect_id_epoch=expect_id_epoch,
        )

    def _prepare_attempt(
        self, backend: str, settings: PreWhitenSettings, band: str
    ) -> None:
        self.state.backend = backend
        self.state.band = band
        self._request = _Request(backend, settings, band)
        self._preview_timer.stop()
        self._show(self._session_key(band))

    def session_cancel(self) -> None:
        """Stop the displayed session's long job after its current step."""
        key = self._shown
        if key is not None and self._bridge.cancel(key):
            self.status_message.emit(
                "Stopping after the current step; what was found is kept."
            )

    # -- edits -------------------------------------------------------------------------
    def session_do(
        self, op: str, *, expect_id_epoch: int | None = None, **args: Any
    ) -> None:
        """Apply one op to the displayed session (FIFO on the worker, never dropped).

        Id-addressed ops carry ``expect_id_epoch`` (the view's ``id_epoch``) and are
        refused when the component list changed meanwhile. ``"finalize"`` is polish
        plus prune.
        """
        key = self._active_key()
        if key is not None:
            self._bridge.submit(key, op, args, expect_id_epoch=expect_id_epoch)

    def session_do_group(
        self,
        actions: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        expect_id_epoch: int | None = None,
    ) -> None:
        """Apply several ops that undo as one."""
        key = self._active_key()
        if key is not None:
            self._bridge.submit_group(key, actions, expect_id_epoch=expect_id_epoch)

    def add_frequency(self, frequency: float, exact: bool = False) -> None:
        """Add a component at ``frequency`` (``SpectrumView.add_requested``).

        Plain: at the residual peak nearest ``frequency``, refined as the settings
        say. ``exact``: at ``frequency`` itself, not refined.
        """
        key = self._active_key()
        if key is None:
            return
        f = float(frequency)

        def call(session: PreWhitenSession) -> StepReport:
            if exact:
                return session.add(f, snap=False, refine="none")
            return session.add(f)

        self._bridge.submit_call(key, "add", call)

    def delete_points(self, indices: Sequence[int] | Any) -> None:
        """Leave points out (indices into ``SessionView.time``); one undo step."""
        key = self._active_key()
        if key is None:
            return
        chosen = np.asarray(indices, dtype=np.int64).copy()

        def call(session: PreWhitenSession) -> StepReport:
            return session.delete_points(chosen)

        self._bridge.submit_call(key, "delete points", call)

    def restore_points(self, indices: Sequence[int] | Any | None = None) -> None:
        """Bring deleted points back (``None``: every one of them)."""
        key = self._active_key()
        if key is None:
            return
        chosen = None if indices is None else np.asarray(indices, dtype=np.int64).copy()

        def call(session: PreWhitenSession) -> StepReport:
            return session.restore_points(chosen)

        self._bridge.submit_call(key, "restore points", call)

    def define_partition(self, request: _PartitionRequest) -> None:
        """Define or replace a partition from a Subsets-dock request.

        The request is resolved on the worker against the session's own time stamps,
        partitions and label columns (only the worker has them); a request that
        cannot be resolved is reported, not raised.
        """
        key = self._active_key()
        if key is None:
            return

        def call(session: PreWhitenSession) -> StepReport:
            partition = request.resolve(
                session.data.time,
                partitions={p.name: p for p in session.state.partitions},
                labels=session.data.labels,
            )
            return session.define_partition(partition)

        self._bridge.submit_call(key, "define partition", call)

    def weights_from_rms(self, name: str) -> None:
        """Weight partition ``name``'s subsets by 1/rms² of their residuals and use
        them (one undo step; the log stores the numbers)."""
        key = self._active_key()
        if key is None:
            return

        def call(session: PreWhitenSession) -> list[StepReport]:
            weights = session.subset_rms_weights(name)
            first = session.set_subset_weights(name, weights)
            if not first.committed and first.kind == "refused":
                return [first]
            return [first, session.use_weights(name)]

        self._bridge.submit_call(key, "weights from rms", call)

    def session_undo(self) -> None:
        key = self._active_key()
        if key is not None:
            self._bridge.undo(key)

    def session_redo(self) -> None:
        key = self._active_key()
        if key is not None:
            self._bridge.redo(key)

    def session_goto(self, seq: int) -> None:
        """Move the displayed session to the state after log entry ``seq``."""
        key = self._active_key()
        if key is not None:
            self._bridge.goto(key, int(seq))

    # -- queries -----------------------------------------------------------------------
    def query_variations(
        self,
        ids: Sequence[int] | None,
        mode: str,
        by: str | None,
        improve_frequencies: bool = False,
    ) -> None:
        """Per-subset amplitude/phase of ``ids`` (``variations_ready``)."""
        key = self._active_key()
        if key is not None:
            self._bridge.query_variations(key, ids, mode, by, improve_frequencies)

    def compute_errors(self) -> None:
        """The full result with the configured estimator (a full view; long)."""
        key = self._active_key()
        if key is not None:
            self._bridge.compute_errors(key)

    def copy_entry_script(self, seq: int) -> None:
        """Log entry ``seq`` as Python, to the clipboard (``entry_script_ready``)."""
        key = self._active_key()
        if key is not None:
            self._bridge.entry_script(key, int(seq))

    def _on_query(self, kind: str, payload: object) -> None:
        if kind == "variations":
            self.variations_ready.emit(payload)
        elif kind == "entry_script" and isinstance(payload, tuple):
            seq, text = payload
            app = QtGui.QGuiApplication.instance()
            if isinstance(app, QtGui.QGuiApplication):
                clipboard = app.clipboard()
                if clipboard is not None:
                    clipboard.setText(str(text))
            self.entry_script_ready.emit(int(seq), str(text))
            self.session_message.emit("info", f"Copied log entry {seq} as Python.")
        elif kind == "errors":
            self.status_message.emit("Uncertainties computed.")

    # -- selection and cursor ----------------------------------------------------------
    def select_components(self, ids: Sequence[int]) -> None:
        """Select components by id (the table, a marker click); the fold follows the
        first one, the cursor does not move."""
        chosen = tuple(int(i) for i in ids)
        view = self.state.session_view
        if view is not None:
            chosen = tuple(i for i in chosen if view.row(i) is not None)
        if not chosen:
            self.state.selected_components = ()
            self.components_selected.emit(())
            return
        self._select_ids(chosen, view)

    def set_cursor(self, frequency: float | None) -> None:
        """The pick cursor moved in the spectrum (no echo)."""
        if frequency is not None:
            frequency = float(frequency)
            if not (math.isfinite(frequency) and frequency > 0.0):
                frequency = None
        self.state.cursor_frequency = frequency

    def _move_cursor(self, frequency: float | None) -> None:
        self.state.cursor_frequency = frequency
        self.cursor_changed.emit(math.nan if frequency is None else float(frequency))

    def _select_ids(self, ids: tuple[int, ...], view: SessionView | None) -> None:
        self.state.selected_components = ids
        self.components_selected.emit(ids)
        row = view.row(ids[0]) if view is not None else None
        if (
            row is not None
            and math.isfinite(row.period)
            and row.period > 0.0
            and self.state.selected_period != row.period
        ):
            self.state.selected_peak = None
            self.select_period(row.period)

    def _clear_component_selection(self) -> None:
        had = bool(self.state.selected_components)
        self.state.selected_components = ()
        if had:
            self.components_selected.emit(())
        if had or self.state.selected_period is not None:
            self.state.selected_peak = None
            self.state.selected_period = None
            self.selection_cleared.emit()

    def _apply_selection(
        self, prev: SessionView | None, view: SessionView, intent: _Intent
    ) -> None:
        """The selection and pick-cursor rules of design 4.3 for a settled view."""
        first: ComponentView | None = next((r for r in view.rows if r.active), None)
        selected = self.state.selected_components
        keep = object()
        target: object = keep
        to_candidate = intent.attempt or intent.auto_done
        same_epoch = prev is not None and prev.id_epoch == view.id_epoch
        if intent.reset or intent.auto_done:
            target = first.id if first is not None else None
        else:
            new_ids: set[int] = set()
            if prev is not None and same_epoch:
                new_ids = {r.id for r in view.rows} - {r.id for r in prev.rows}
            current = selected[0] if selected else None
            row = view.row(current) if current is not None else None
            if new_ids:
                target = max(new_ids)  # added, stepped: the new component
                to_candidate = True
            elif prev is not None and not same_epoch:
                # A history move, prune, renumber: ids may mean other components now.
                old = prev.row(current) if current is not None else None
                same = (
                    row is not None
                    and old is not None
                    and row.frequency == old.frequency
                    and row.active
                )
                if not same:
                    target = first.id if first is not None else None
            elif current is not None and (row is None or not row.active):
                target = first.id if first is not None else None
        if isinstance(target, int):
            self._select_ids((target,), view)
        elif target is None:
            self._clear_component_selection()
        else:
            ids = tuple(i for i in selected if view.row(i) is not None)
            if ids != selected:
                self.state.selected_components = ids
                self.components_selected.emit(ids)
            if ids and prev is not None and same_epoch:
                row, old = view.row(ids[0]), prev.row(ids[0])
                if (
                    row is not None
                    and old is not None
                    and row.frequency != old.frequency
                ):
                    self.select_period(row.period)  # the fold follows an improved one
        if to_candidate and view.candidate is not None:
            self._move_cursor(view.candidate.frequency)
        elif intent.reset and not to_candidate:
            self._move_cursor(None)

    # -- bridge signals ----------------------------------------------------------------
    def _on_view(self, key: object, view: object) -> None:
        if not isinstance(view, SessionView) or not isinstance(key, SessionKey):
            return
        if key != self._shown:
            return
        prev = self.state.session_view
        if prev is not None and prev.key != key:
            prev = None
        self.state.session_view = view
        self.state.preview_error = ""  # the session is open after all
        self._note_backend(view)
        if view.progress:
            self.session_changed.emit(view)
            self._emit_project()
            return
        intent = self._intents.pop(key, None) or _Intent()
        if view.result is not None:
            self.state.current_solution = view.result
            self.state.current_pg = None
            self.state.method = PREWHITEN_KEY
            if view.light:
                # The committed change moved the active frequencies (4.2 (b)).
                self.solution_ready.emit(view.result)
        self.session_changed.emit(view)
        self._apply_selection(prev, view, intent)
        self._emit_project()

    def _note_backend(self, view: SessionView) -> None:
        backend = view.backend
        if backend and backend not in _CPU_BACKENDS:
            self._gpu_used = True

    def _on_progress(self, key: object, report: object) -> None:
        if not isinstance(key, SessionKey):
            return
        self._intents.setdefault(key, _Intent()).attempt = True
        if key == self._shown:
            self.session_progress.emit(report)

    def _on_reported(self, key: object, reports: object) -> None:
        if not isinstance(key, SessionKey) or not isinstance(reports, tuple):
            return
        if any(getattr(r, "kind", "") in _ATTEMPT_KINDS for r in reports):
            self._intents.setdefault(key, _Intent()).attempt = True

    def _on_auto_finished(self, key: object, bundle: object) -> None:
        if not isinstance(key, SessionKey) or not isinstance(bundle, SolutionBundle):
            return
        rkey = bundle.tag
        if isinstance(rkey, ResultKey) and self._cacheable(bundle, rkey):
            # Cached displayed or not, superseded or not, as v1.3 cached results.
            self._solutions.put(rkey, bundle)
        if self._auto_inflight.get(key) == rkey:
            del self._auto_inflight[key]
        started = self._auto_t0.pop(key, None)
        if key != self._shown:
            return
        elapsed = 0.0 if started is None else (_time.perf_counter() - started) * 1e3
        self._intents.setdefault(key, _Intent()).auto_done = True
        self._apply_solution(bundle.result, elapsed, from_cache=False, busy=None)

    @staticmethod
    def _cacheable(bundle: SolutionBundle, rkey: ResultKey) -> bool:
        """An untouched run of exactly the cached settings (only then is a cache hit
        the same solution a fresh run would give)."""
        from cuperiod.prewhiten.session.actions import PRISTINE_OPS

        snapshot = bundle.snapshot
        pristine = all(entry.op in PRISTINE_OPS for entry in snapshot.log)
        return pristine and (
            settings_hash(snapshot.creation.settings) == rkey.settings_hash
        )

    def _on_finalized(self, key: object, result: object) -> None:
        if key == self._shown and isinstance(result, PreWhitenResult):
            self.state.current_solution = result
            self.state.method = PREWHITEN_KEY
            self.solution_ready.emit(result)

    def _on_settled(self, key: object) -> None:
        if isinstance(key, SessionKey):
            self._auto_inflight.pop(key, None)

    def _on_session_failed(self, key: object, text: str) -> None:
        if not isinstance(key, SessionKey):
            return
        self._auto_t0.pop(key, None)
        self._abort_saves(key)
        path = self._loading.pop(key, None)
        if path is not None:
            self._project_failed(key, path, text)
            return
        if key == self._shown:
            self.compute_failed.emit(text)
        else:
            self.session_message.emit("error", f"{self.session_name(key)}: {text}")

    def _emit_project(self) -> None:
        key = self._shown
        path = (self._bridge.project_path(key) or "") if key is not None else ""
        dirty = key is not None and self._bridge.is_dirty(key)
        self.state.project_path = path or None
        if (path, dirty) != self._project_state:
            self._project_state = (path, dirty)
            self.project_changed.emit(path, dirty)

    # -- unsaved sessions --------------------------------------------------------------
    def confirm_then(
        self,
        action: Callable[[], None],
        *,
        keys: Iterable[SessionKey] | None = None,
    ) -> bool:
        """Run ``action`` once no unsaved session among ``keys`` would be lost.

        ``keys`` defaults to every dirty session. With none dirty (or no dialog
        provider to ask) ``action`` runs at once. Otherwise
        :meth:`DialogProvider.confirm_discard` asks: *Save* writes each session
        (asking for a file where it has none) and runs ``action`` once all are
        saved; *Discard* drops them first; *Cancel* drops ``action``. Returns whether
        ``action`` has run by the time this returns.
        """
        candidates = self._bridge.dirty_keys() if keys is None else tuple(keys)
        dirty = [k for k in candidates if self._bridge.is_dirty(k)]
        provider = self._dialogs
        if not dirty or provider is None:
            action()
            return True
        ran: list[bool] = []

        def run() -> None:
            ran.append(True)
            action()

        def answered(choice: DiscardChoice) -> None:
            if choice == "discard":
                for k in dirty:
                    self._bridge.discard(k)
                    self._intents.pop(k, None)
                self._emit_project()
                run()
            elif choice == "save":
                self._save_all(dirty, run)
            else:
                self.status_message.emit("Cancelled; nothing was discarded.")

        provider.confirm_discard([self.session_name(k) for k in dirty], answered)
        return bool(ran)

    def _save_all(self, keys: Sequence[SessionKey], then: Callable[[], None]) -> None:
        """Save every session of ``keys`` (asking for missing files), then ``then``."""
        queue = list(keys)
        paths: dict[SessionKey, str] = {}
        provider = self._dialogs

        def advance() -> None:
            while queue:
                key = queue[0]
                known = self._bridge.project_path(key)
                if known:
                    paths[key] = known
                    queue.pop(0)
                    continue
                if provider is None:
                    return

                def chosen(path: str | None, key: SessionKey = key) -> None:
                    if not path:
                        self.status_message.emit("Not saved; nothing was discarded.")
                        return
                    paths[key] = _project_file(path)
                    queue.pop(0)
                    advance()

                provider.save_file(
                    f"Save {self.session_name(key)}",
                    PROJECT_FILTER,
                    self._suggested_project(key),
                    chosen,
                )
                return
            wait = _SaveWait(set(paths), then)
            self._save_waits.append(wait)
            for key, path in paths.items():
                self._bridge.save(key, path)

        advance()

    def _on_saved(self, key: object, path: str) -> None:
        if not isinstance(key, SessionKey):
            return
        for wait in list(self._save_waits):
            wait.pending.discard(key)
            if not wait.pending:
                self._save_waits.remove(wait)
                wait.then()
        if key == self._shown:
            self._emit_project()

    def _abort_saves(self, key: SessionKey) -> None:
        for wait in list(self._save_waits):
            if key in wait.pending:
                self._save_waits.remove(wait)
                self.status_message.emit(
                    f"{self.session_name(key)} was not saved; nothing was discarded."
                )

    def confirm_close(self) -> bool:
        """Whether the window may close now (no unsaved session would be lost).

        ``False`` while the user is being asked: the answer arrives later and, once
        the sessions are saved or discarded, ``close_approved`` is emitted so the
        window closes again (this then returns ``True``).
        """
        if self._close_approved or not self._bridge.dirty_keys():
            return True
        self._closing_sync = True
        try:
            done = self.confirm_then(self._approve_close)
        finally:
            self._closing_sync = False
        return done

    def _approve_close(self) -> None:
        self._close_approved = True
        if not self._closing_sync:  # answered later: the window must close again
            self.close_approved.emit()

    # -- sessions and projects ---------------------------------------------------------
    def new_session(
        self,
        backend: str | None = None,
        settings: PreWhitenSettings | None = None,
        band: str | None = None,
    ) -> None:
        """Replace the displayed session by a fresh one (asks when it is unsaved).

        The fresh session uses the given form values, else the last ones.
        """
        if band is not None:
            self.state.band = band
        if backend is not None or settings is not None:
            last = self._request
            self._request = _Request(
                backend if backend is not None else (last.backend if last else "auto"),
                settings
                if settings is not None
                else (last.settings if last else PreWhitenSettings()),
                self.state.band,
            )
        if self.state.current_lc is None:
            self.session_message.emit("warning", "Open a light curve first.")
            return
        key = self._session_key(self.state.band)

        def fresh() -> None:
            self._bridge.discard(key)
            self._intents.pop(key, None)
            self._auto_inflight.pop(key, None)
            request = self._request or _Request(
                "auto", PreWhitenSettings(), self.state.band
            )
            self._open_session(key, request, preview=False, lazy=False)
            self._show(key, force=True)

        self.confirm_then(fresh, keys=[key])

    def open_project(self, path: str | Path, *, verify: bool = False) -> None:
        """Open a ``*.cupw.json`` project as the current source (``project:<path>``).

        Leaves batch mode, switches to pre-whitening, fills the Options form with the
        project's settings (``settings_restored``) and displays its session. Other
        sessions, dirty ones included, stay alive. A project whose data file cannot
        be found asks for it through the dialog provider. ``verify`` (File ▸ Verify
        project) replays the project's log first and refuses a head that does not
        follow from it.
        """
        target = str(path)
        key = SessionKey(PROJECT_PREFIX + target, "", "")
        self.confirm_then(
            lambda: self._load_project(key, target, None, verify=verify), keys=[key]
        )

    def _load_project(
        self, key: SessionKey, path: str, data: Any, *, verify: bool = False
    ) -> None:
        if self._bridge.has_session(key):
            self._bridge.discard(key)  # reopened: the file's version replaces it
        self._loading[key] = path
        if verify:
            self._verifying.add(key)
        else:
            self._verifying.discard(key)

        def loader() -> PreWhitenSession:
            from cuperiod.prewhiten.session.session import PreWhitenSession

            return PreWhitenSession.load(path, data=data, verify=verify)

        self._bridge.load(key, loader, project_path=path, describe=_describe_loaded)
        self.status_message.emit(f"Opening {Path(path).name}…")

    def _on_loaded(self, key: object, payload: object) -> None:
        if not isinstance(key, SessionKey):
            return
        path = self._loading.pop(key, None)
        if path is None or not isinstance(payload, tuple):
            return
        lc, settings = payload
        if self.state.mode == "batch":
            self.exit_batch_mode()
        self.state.band = ""
        backend = self._request.backend if self._request is not None else "auto"
        self._request = _Request(backend, settings, "")
        # The curve first, then the analysis: switching first would briefly show
        # the previous curve's session.
        self.set_light_curve(lc, key.source_id)
        self.set_analysis("prewhiten")
        self.settings_restored.emit(settings)
        self._emit_project()
        if key in self._verifying:
            self._verifying.discard(key)
            self.status_message.emit(
                f"Verified {Path(path).name}: replaying its log reproduces the "
                "saved result."
            )

    def _project_failed(self, key: SessionKey, path: str, text: str) -> None:
        self._bridge.discard(key)
        name = Path(path).name
        provider = self._dialogs
        verify = key in self._verifying
        self._verifying.discard(key)
        if "not found" in text and provider is not None:

            def located(data_path: str | None) -> None:
                if data_path:
                    self._load_project(key, path, data_path, verify=verify)
                else:
                    self.compute_failed.emit(f"Could not open {name}: {text}")

            provider.open_file(
                f"Locate the light curve of {name}", LIGHT_CURVE_FILTER, located
            )
            return
        self.compute_failed.emit(f"Could not open {name}: {text}")

    def save_project(
        self, path: str | Path | None = None, *, ask: bool = False
    ) -> None:
        """Save the displayed session to ``path``, its project file, or a file the
        user picks (``ask``: always pick, as *Save as…*)."""
        key = self._shown
        if key is None or not self._bridge.is_open(key):
            self.session_message.emit("warning", "No pre-whitening session to save.")
            return
        target = str(path) if path is not None else None
        if target is None and not ask:
            target = self._bridge.project_path(key)
        if target:
            self._bridge.save(key, _project_file(target))
            return
        provider = self._dialogs
        if provider is None:
            self.session_message.emit(
                "warning", "Choose a file to save the project to."
            )
            return

        def chosen(picked: str | None) -> None:
            if picked:
                self._bridge.save(key, _project_file(picked))

        provider.save_file(
            "Save project", PROJECT_FILTER, self._suggested_project(key), chosen
        )

    def _suggested_project(self, key: SessionKey) -> str:
        known = self._bridge.project_path(key)
        if known:
            return known
        # From the source id, not the label: a label ("Kepler KIC 7532973 — hot-
        # Jupiter transit (flux)") makes a poor file name.
        stem = Path(self._source_file_name(key)).stem or "session"
        return stem + PROJECT_SUFFIX

    def export_script(self, path: str | Path) -> None:
        """Write the displayed session as a Python script (and its data file)."""
        key = self._shown
        if key is None or not self._bridge.is_open(key):
            self.session_message.emit("warning", "No pre-whitening session to export.")
            return
        target = str(path)

        def call(session: PreWhitenSession) -> None:
            session.export_script(target)

        self._bridge.submit_call(
            key,
            "Export script",
            call,
            manual=False,
            notice=f"Exported the script to {target}",
        )

    # -- commands ----------------------------------------------------------------------
    def run_command(self, command_id: str) -> bool:
        """Carry out a registry command that needs nothing from a widget.

        Returns ``False`` for the commands the window handles itself (those that
        read the Options form, a selection, or open a dialog of its own).
        """
        simple: dict[str, Callable[[], None]] = {
            "edit.undo": self.session_undo,
            "edit.redo": self.session_redo,
            "session.stop": self.cancel_compute,
            "session.calculate": lambda: self.session_do("calculate"),
            "session.add_peak": lambda: self.session_do("add_peak"),
            "session.finalize": lambda: self.session_do("finalize"),
            "session.renumber": lambda: self.session_do("renumber"),
            "data.restore_all": lambda: self.restore_points(None),
            "file.new_session": self.new_session,
            "file.save_project": self.save_project,
            "file.save_project_as": lambda: self.save_project(ask=True),
        }
        if command_id == "session.add_at_cursor":
            cursor = self.state.cursor_frequency
            if cursor is None:
                self.session_message.emit(
                    "info", "Click the spectrum (or press N) to place the cursor."
                )
            else:
                self.add_frequency(cursor)
            return True
        handler = simple.get(command_id)
        if handler is None:
            return False
        handler()
        return True


__all__ = [
    "CACHE_BYTES",
    "LIGHT_CURVE_FILTER",
    "PREVIEW_DEBOUNCE_MS",
    "PREWHITEN_KEY",
    "PROJECT_FILTER",
    "PROJECT_PREFIX",
    "PROJECT_SUFFIX",
    "Analysis",
    "AppController",
    "AppState",
    "Mode",
]
