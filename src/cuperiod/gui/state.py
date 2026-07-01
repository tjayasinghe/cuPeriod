"""Application state and the controller that mediates between widgets and compute.

:class:`AppState` is a plain dataclass snapshot of what the app is showing. The
:class:`AppController` is the single hub: widgets call its setters and listen to its
signals; it owns the :class:`~cuperiod.gui.compute.ComputeManager` and the
:class:`~cuperiod.gui.models.ResultCache`, runs (or cache-hits) periodograms, derives
peaks, and tracks the selected period that drives the phased view. Views never talk to
each other — only to the controller — so the data flow stays one-directional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.compute import ComputeManager
from cuperiod.gui.fold import epoch_for_peak
from cuperiod.gui.meta import (
    alias_diverse_default,
    auto_max_frequency,
    supports_multiband,
)
from cuperiod.gui.models import (
    LoadedCurve,
    ResultCache,
    ResultKey,
    SourceItem,
    settings_hash,
)
from cuperiod.gui.qt import QObject, Signal

Mode = Literal["single", "batch"]

#: Denser default frequency oversampling for smoother, better-resolved periodograms
#: (helps the jagged look on a period axis at long periods). Applied only when the user
#: leaves ``samples_per_peak`` at the model default.
_DEFAULT_SAMPLES_PER_PEAK = 10


@dataclass
class AppState:
    """A snapshot of what the app is currently showing."""

    theme: str = "dark"
    mode: Mode = "single"
    method: str = "GLS"
    backend: str = "auto"
    n_peaks: int = 10
    alias_diverse: bool = False
    band: str = ""
    source_id: str = ""
    current_lc: LoadedCurve | None = None
    current_pg: Periodogram | None = None
    selected_peak: Peak | None = None
    selected_period: float | None = None
    sources: list[SourceItem] | None = None
    current_source_index: int = 0
    last_compute_ms: float = 0.0
    last_from_cache: bool = False


class AppController(QObject):
    """Owns state, compute pool, and cache; the sole mediator between widgets."""

    lc_loaded = Signal(object)  # LoadedCurve
    compute_started = Signal()
    periodogram_ready = Signal(object)  # Periodogram
    compute_failed = Signal(str)
    peaks_ready = Signal(object)  # list[Peak]
    period_changed = Signal(float)  # drives the spectrum line
    fold_changed = Signal(float, float)  # (period, t0) drives the phased view
    busy_changed = Signal(bool)
    sources_changed = Signal(object)  # list[str] labels (batch mode)
    source_selected = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = AppState()
        self._cache = ResultCache()
        self._compute = ComputeManager(self)
        self._pending_key: ResultKey | None = None
        self._compute.signals.finished.connect(self._on_finished)
        self._compute.signals.failed.connect(self._on_failed)

    # -- inputs ------------------------------------------------------------------
    def set_light_curve(self, lc: LoadedCurve, source_id: str) -> None:
        """Set the active light curve (clears the previous spectrum/selection)."""
        self.state.current_lc = lc
        self.state.source_id = source_id
        self.state.current_pg = None
        self.state.selected_peak = None
        self.state.selected_period = None
        self.lc_loaded.emit(lc)

    # -- batch mode --------------------------------------------------------------
    def load_sources(self, items: list[SourceItem]) -> None:
        """Enter batch mode with ``items`` (labels via ``sources_changed``)."""
        self.state.sources = list(items)
        self.state.mode = "batch"
        self.sources_changed.emit([item.label for item in items])

    def select_source(self, index: int) -> None:
        """Load and activate source ``index`` (the light curve, then auto-compute)."""
        sources = self.state.sources
        if not sources:
            return
        index = max(0, min(index, len(sources) - 1))
        self.state.current_source_index = index
        item = sources[index]
        try:
            lc = item.loader()
        except Exception as exc:  # noqa: BLE001 - surface load failures to the UI
            self.compute_failed.emit(
                f"load {item.key}: {type(exc).__name__}: {exc}"
            )
            return
        self.source_selected.emit(index)
        self.set_light_curve(lc, item.key)

    def run(
        self,
        method: str,
        backend: str,
        settings: BaseSettings,
        n_peaks: int,
        band: str = "",
    ) -> None:
        """Compute (or cache-hit) the periodogram for the active light curve.

        ``band`` selects a multiband input: a band name for a single band, ``"stacked"``
        to merge bands, or ``""``/``"combined"`` to use the whole multiband curve (for
        multiband-capable methods; otherwise it falls back to a stacked single curve).
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
        source_id = self.state.source_id + (f"::{band}" if band else "")
        key = ResultKey(source_id, method, settings_hash(settings), backend)
        self._pending_key = key
        cached = self._cache.get(key)
        if cached is not None:
            self._apply_result(key, cached, 0.0, from_cache=True)
            return
        self.busy_changed.emit(True)
        self.compute_started.emit()
        self._compute.submit(key, lc_input, method, settings, backend)

    def _resolve_input(self, method: str, band: str) -> LoadedCurve | None:
        """Pick the curve to analyse from the (possibly multiband) active curve."""
        lc = self.state.current_lc
        if not isinstance(lc, MultiBandLightCurve):
            return lc
        if band in lc.bands:
            return lc.bands[band]
        if band == "stacked" or not supports_multiband(method):
            return self._stack(lc)
        return lc

    @staticmethod
    def _stack(mblc: MultiBandLightCurve) -> LightCurve:
        """Merge all bands into one time-sorted single-band curve."""
        time, value, error, _ = mblc.finite().stacked()
        domain = next(iter(mblc.bands.values())).domain
        return LightCurve.from_arrays(time, value, error, domain=domain)

    def _tune_auto_grid(
        self, lc: LoadedCurve, settings: BaseSettings
    ) -> BaseSettings:
        """Improve the default frequency grid for frequency-grid methods.

        (a) Raises an auto max-frequency so sub-day periods aren't missed on sparse data
        (see :func:`cuperiod.gui.meta.auto_max_frequency`) and (b) densifies the default
        ``samples_per_peak`` for smoother, better-resolved periodograms (notably at long
        periods). Explicitly-changed values are left as-is.
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
            updates["maximum_frequency"] = auto_max_frequency(time, nyquist_factor)
        if "samples_per_peak" in fields:
            current = getattr(settings, "samples_per_peak", None)
            if current is not None and current == fields["samples_per_peak"].default:
                updates["samples_per_peak"] = max(
                    int(current), _DEFAULT_SAMPLES_PER_PEAK
                )
        return settings.model_copy(update=updates) if updates else settings

    @staticmethod
    def _time_of(lc: LoadedCurve | None) -> FloatArray:
        """Concatenated time array for a single- or multi-band curve."""
        if isinstance(lc, MultiBandLightCurve):
            times = [band.time for band in lc.bands.values() if band.n]
            return np.concatenate(times) if times else np.empty(0, dtype=np.float64)
        if isinstance(lc, LightCurve):
            return lc.time
        return np.empty(0, dtype=np.float64)

    # -- compute results ---------------------------------------------------------
    def _on_finished(
        self, key: ResultKey, pg: Periodogram, elapsed_ms: float
    ) -> None:
        # Cache even superseded results so revisiting is instant, but only apply the
        # one the user is currently waiting for.
        self._cache.put(key, pg)
        if key == self._pending_key:
            self._apply_result(key, pg, elapsed_ms, from_cache=False)

    def _on_failed(self, key: ResultKey, message: str) -> None:
        if key == self._pending_key:
            self.busy_changed.emit(False)
            self.compute_failed.emit(message)

    def _apply_result(
        self, key: ResultKey, pg: Periodogram, elapsed_ms: float, *, from_cache: bool
    ) -> None:
        self.state.current_pg = pg
        self.state.last_compute_ms = elapsed_ms
        self.state.last_from_cache = from_cache
        self.busy_changed.emit(False)
        self.periodogram_ready.emit(pg)
        peaks = pg.best_periods(
            self.state.n_peaks, alias_diverse=self.state.alias_diverse
        )
        self.peaks_ready.emit(peaks)
        if peaks:
            self.select_peak(peaks[0])
        else:
            self.state.selected_peak = None
            self.state.selected_period = None

    # -- selection ---------------------------------------------------------------
    def select_peak(self, peak: Peak) -> None:
        """Select ``peak`` as the active period (drives the line and phased view)."""
        self.state.selected_peak = peak
        self.state.selected_period = peak.period
        epoch = epoch_for_peak(peak, self._lc_time())
        self.period_changed.emit(peak.period)
        self.fold_changed.emit(peak.period, epoch)

    def select_period(self, period: float) -> None:
        """Select an arbitrary ``period`` (e.g. from dragging the spectrum line)."""
        self.state.selected_peak = None
        self.state.selected_period = period
        epoch = epoch_for_peak(None, self._lc_time())
        self.period_changed.emit(period)
        self.fold_changed.emit(period, epoch)

    def current_time(self) -> FloatArray:
        """Time array of the active curve (bands concatenated); empty if none loaded."""
        return self._time_of(self.state.current_lc)

    def _lc_time(self) -> FloatArray:
        """Time array for the active curve (bands concatenated), for the epoch."""
        return self._time_of(self.state.current_lc)


__all__ = ["AppController", "AppState", "Mode"]
