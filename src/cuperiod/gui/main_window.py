"""The top-level application window: toolbar, controls, info bar, and work area.

Assembles the controller and all panels and wires them together through the controller's
signals: controls → run, results → spectrum/phased/peaks, source browser → batch scroll.
The chosen theme is remembered across sessions via ``QSettings``.

Two analyses share the shell. Switching the controls panel's **Analysis** picker swaps
which docks are on show — *Peaks* for a periodogram, *Frequencies* and *Period spacing*
for pre-whitening — while the spectrum, phased and raw views, the source browser, and
every load path stay exactly as they were. The docks are tabbed together, so the swap
never changes the window layout the user has arranged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.appinfo import version_label
from cuperiod.gui.loaders import (
    demo_sources,
    enumerate_sources,
    load_path,
    preview_file,
)
from cuperiod.gui.models import SourceItem
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets
from cuperiod.gui.state import AppController
from cuperiod.gui.theme import ThemeName, app_icon, apply_theme, palette
from cuperiod.gui.widgets.controls_panel import ControlsPanel
from cuperiod.gui.widgets.info_bar import RunInfoBar
from cuperiod.gui.widgets.peaks_panel import PeaksTable
from cuperiod.gui.widgets.phased_view import PhasedView
from cuperiod.gui.widgets.preview_dialog import PreviewDialog
from cuperiod.gui.widgets.rawlc_view import RawLightCurveView
from cuperiod.gui.widgets.solution_panel import SolutionPanel
from cuperiod.gui.widgets.source_panel import SourceBrowser
from cuperiod.gui.widgets.spacing_panel import SpacingPanel
from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD, SpectrumView
from cuperiod.prewhiten.result import PreWhitenResult

_FILE_FILTER = (
    "Light curves (*.csv *.ecsv *.fits *.fit *.fz *.parquet *.pq "
    "*.tsv *.tab *.dat *.txt);;All files (*)"
)

#: Extensions recognized as light-curve files, mirroring ``_FILE_FILTER`` above; used to
#: validate drag-and-drop drops (which bypass the file dialog's own filtering).
_FILE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".ecsv",
        ".fits",
        ".fit",
        ".fz",
        ".parquet",
        ".pq",
        ".tsv",
        ".tab",
        ".dat",
        ".txt",
    }
)


def _spectrum_heights(
    grid: np.ndarray, values: np.ndarray, at: Sequence[float]
) -> np.ndarray:
    """``values`` sampled at the grid points nearest each frequency in ``at``.

    Vectorized rather than a per-frequency ``argmin``: the pre-whitening grid runs to
    over a million samples and this is on the UI thread.
    """
    wanted = np.asarray(at, dtype=np.float64)
    if grid.size == 0 or wanted.size == 0:
        return np.zeros(wanted.size, dtype=np.float64)
    right = np.searchsorted(grid, wanted).clip(0, grid.size - 1)
    left = (right - 1).clip(0, grid.size - 1)
    nearer_left = np.abs(grid[left] - wanted) <= np.abs(grid[right] - wanted)
    return np.asarray(values[np.where(nearer_left, left, right)], dtype=np.float64)


class MainWindow(QtWidgets.QMainWindow):
    """Main window shell: toolbar, left controls, central plots area, and status bar."""

    def __init__(
        self, app: QtWidgets.QApplication, *, theme: ThemeName = "dark"
    ) -> None:
        super().__init__()
        self._app = app
        saved = str(QtCore.QSettings().value("theme", theme))
        self._theme: ThemeName = "light" if saved == "light" else "dark"
        apply_theme(app, self._theme)

        self.setWindowIcon(app_icon())
        self.setWindowTitle("cuPeriod — periodogram & pre-whitening explorer")
        self.resize(1360, 860)  # default size; overridden below if a geometry was saved
        self.setMinimumSize(1024, 680)
        self.setAcceptDrops(True)

        self._controller = AppController(self)
        self._controls = ControlsPanel()
        self._info = RunInfoBar()
        self._spectrum = SpectrumView(self._theme)
        self._phased = PhasedView(self._theme)
        self._raw = RawLightCurveView(self._theme)
        self._peaks_table = PeaksTable()
        self._solution_panel = SolutionPanel()
        self._spacing_panel = SpacingPanel(self._theme)
        self._source_browser = SourceBrowser()

        self._banner_timer = QtCore.QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._hide_banner)

        self._build_toolbar()
        self._build_central()
        self._build_result_docks()
        self._build_sources_dock()
        self._build_shortcuts()
        self._connect()
        self._restore_window_state()
        self._apply_analysis("periodogram")
        self._show_status("Open a light curve or load a demo to begin.")

    # -- construction ------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Main")
        toolbar.setObjectName("main_toolbar")  # required for saveState()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        title = QtWidgets.QLabel("  cuPeriod  ")
        title.setObjectName("heading")
        toolbar.addWidget(title)
        version = QtWidgets.QLabel(version_label() + "   ")
        version.setObjectName("muted")
        version.setToolTip("Installed version and git revision")
        toolbar.addWidget(version)

        open_action = QtGui.QAction("Open…", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        toolbar.addAction(open_action)

        folder_action = QtGui.QAction("Open folder…", self)
        folder_action.setToolTip("Open a folder of light curves for batch browsing")
        folder_action.triggered.connect(self._open_folder)
        toolbar.addAction(folder_action)

        demo_button = QtWidgets.QToolButton()
        demo_button.setText("Load demo")
        demo_button.setToolTip(
            "Load a bundled example light curve, or browse a demo batch"
        )
        demo_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        demo_button.setMenu(self._build_demo_menu())
        toolbar.addWidget(demo_button)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        toolbar.addWidget(spacer)

        self._theme_action = QtGui.QAction("Theme", self)
        self._theme_action.setToolTip("Toggle dark / light theme")
        self._theme_action.triggered.connect(self._toggle_theme)
        toolbar.addAction(self._theme_action)

    def _build_demo_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        for item in demo_sources():
            action = menu.addAction(item.label)
            action.triggered.connect(lambda _=False, it=item: self._load_demo(it))
        menu.addSeparator()
        batch = menu.addAction("Batch: browse all demo sources")
        batch.triggered.connect(self._load_demo_batch)
        return menu

    def _build_central(self) -> None:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self._controls.setMinimumWidth(320)
        self._controls.setMaximumWidth(460)
        splitter.addWidget(self._controls)

        center = QtWidgets.QWidget()
        self._center_layout = QtWidgets.QVBoxLayout(center)
        self._center_layout.setContentsMargins(8, 8, 8, 8)
        self._center_layout.setSpacing(8)
        self._center_layout.addWidget(self._info)

        self._banner_container = QtWidgets.QWidget()
        banner_layout = QtWidgets.QHBoxLayout(self._banner_container)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.setSpacing(4)
        self._banner = QtWidgets.QLabel("")
        self._banner.setObjectName("error")
        self._banner.setWordWrap(True)
        banner_layout.addWidget(self._banner, 1)
        self._banner_dismiss = QtWidgets.QToolButton()
        self._banner_dismiss.setText("✕")
        self._banner_dismiss.setAutoRaise(True)
        self._banner_dismiss.setToolTip("Dismiss")
        self._banner_dismiss.clicked.connect(self._hide_banner)
        banner_layout.addWidget(self._banner_dismiss, 0, Qt.AlignmentFlag.AlignTop)
        self._banner_container.setVisible(False)
        self._center_layout.addWidget(self._banner_container)

        self._stack = QtWidgets.QStackedWidget()
        self._placeholder = QtWidgets.QLabel(
            "Open a light curve or load a demo to begin."
        )
        self._placeholder.setObjectName("muted")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._placeholder)  # index 0
        self._stack.addWidget(self._build_plot_area())  # index 1
        self._center_layout.addWidget(self._stack, 1)

        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        self.setCentralWidget(splitter)

    def _build_plot_area(self) -> QtWidgets.QWidget:
        plot_split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        plot_split.addWidget(self._spectrum)
        lc_split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        lc_split.addWidget(self._phased)
        lc_split.addWidget(self._raw)
        self._raw.setVisible(False)
        lc_split.setSizes([700, 500])
        plot_split.addWidget(lc_split)
        plot_split.setStretchFactor(0, 3)
        plot_split.setStretchFactor(1, 2)
        plot_split.setSizes([520, 340])
        return plot_split

    def _build_result_docks(self) -> None:
        """The three result docks, tabbed together on the right.

        Only the docks belonging to the active analysis are shown, but all three live in
        the same tab group, so switching analysis never rearranges the window.
        """
        self._peaks_dock = self._make_dock("Peaks", "peaks_dock", self._peaks_table)
        self._solution_dock = self._make_dock(
            "Frequencies", "solution_dock", self._solution_panel
        )
        self._spacing_dock = self._make_dock(
            "Period spacing", "spacing_dock", self._spacing_panel
        )
        self.tabifyDockWidget(self._peaks_dock, self._solution_dock)
        self.tabifyDockWidget(self._solution_dock, self._spacing_dock)

    def _make_dock(
        self, title: str, object_name: str, widget: QtWidgets.QWidget
    ) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)  # required for saveState()
        dock.setWidget(widget)
        feature = QtWidgets.QDockWidget.DockWidgetFeature
        dock.setFeatures(
            feature.DockWidgetMovable
            | feature.DockWidgetFloatable
            | feature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def _build_sources_dock(self) -> None:
        self._sources_dock = QtWidgets.QDockWidget("Sources", self)
        self._sources_dock.setObjectName("sources_dock")  # required for saveState()
        self._sources_dock.setWidget(self._source_browser)
        feature = QtWidgets.QDockWidget.DockWidgetFeature
        self._sources_dock.setFeatures(
            feature.DockWidgetMovable
            | feature.DockWidgetFloatable
            | feature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._sources_dock)
        self._sources_dock.setVisible(False)

    def _build_shortcuts(self) -> None:
        quit_action = QtGui.QAction("Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        if quit_action.shortcut().isEmpty():  # StandardKey.Quit is empty on Windows
            quit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        quit_action.setToolTip("Quit cuPeriod  (Ctrl+Q)")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

        # Ctrl+Return and Ctrl+Enter (main keyboard vs. numpad) both trigger Compute.
        for sequence in ("Ctrl+Return", "Ctrl+Enter"):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.activated.connect(self._compute_via_shortcut)

    def _restore_window_state(self) -> None:
        """Restore the previous session's window geometry/layout, if any was saved."""
        settings = QtCore.QSettings()
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        window_state = settings.value("windowState")
        if window_state is not None:
            self.restoreState(window_state)

    def _connect(self) -> None:
        self._controls.run_requested.connect(self._on_run_requested)
        self._controls.prewhiten_requested.connect(self._on_prewhiten_requested)
        self._controls.analysis_changed.connect(self._on_analysis_changed)
        ctl = self._controller
        ctl.solution_ready.connect(self._on_solution_ready)
        self._solution_panel.component_selected.connect(ctl.select_component)
        ctl.lc_loaded.connect(self._on_lc_loaded)
        ctl.compute_started.connect(self._on_compute_started)
        ctl.busy_changed.connect(self._info.set_busy)
        ctl.busy_changed.connect(self._controls.set_busy)
        ctl.periodogram_ready.connect(self._on_periodogram_ready)
        ctl.periodogram_ready.connect(self._spectrum.set_periodogram)
        ctl.peaks_ready.connect(self._spectrum.set_peaks)
        ctl.period_changed.connect(self._spectrum.set_selected_period)
        self._spectrum.period_selected.connect(ctl.select_period)
        ctl.lc_loaded.connect(self._phased.set_light_curve)
        ctl.lc_loaded.connect(self._raw.set_light_curve)
        self._phased.show_raw_toggled.connect(self._raw.setVisible)
        ctl.fold_changed.connect(self._phased.set_fold)
        ctl.peaks_ready.connect(self._peaks_table.set_peaks)
        self._peaks_table.peak_selected.connect(ctl.select_peak)
        self._source_browser.source_selected.connect(ctl.select_source)
        ctl.sources_changed.connect(self._on_sources_changed)
        ctl.source_selected.connect(self._source_browser.set_current)
        ctl.compute_failed.connect(self._on_compute_failed)
        ctl.selection_cleared.connect(self._spectrum.clear_selection)
        ctl.selection_cleared.connect(self._phased.clear_fold)

    # -- actions -----------------------------------------------------------------
    def _open_file(self) -> None:
        settings = QtCore.QSettings()
        start_dir = str(settings.value("last_dir", ""))
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open light curve", start_dir, _FILE_FILTER
        )
        if not path_str:
            return
        path = Path(path_str)
        settings.setValue("last_dir", str(path.parent))
        self._load_file(path)

    def _load_file(self, path: Path) -> None:
        """Preview (if possible) and load a single light-curve file at ``path``."""
        preview = preview_file(path)
        if preview is not None:
            dialog = PreviewDialog(path.name, preview, self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
        try:
            lc = load_path(path)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._warn(f"Could not load {path.name}:\n{type(exc).__name__}: {exc}")
            return
        self._exit_batch_mode()
        self._controller.set_light_curve(lc, source_id=str(path))
        self.setWindowTitle(f"cuPeriod — {path.name}")

    def _load_demo(self, item: SourceItem) -> None:
        try:
            lc = item.loader()
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._warn(f"Could not load demo:\n{type(exc).__name__}: {exc}")
            return
        self._exit_batch_mode()
        self._controller.set_light_curve(lc, source_id=item.key)
        self.setWindowTitle(f"cuPeriod — {item.label}")

    def _open_folder(self) -> None:
        settings = QtCore.QSettings()
        start_dir = str(settings.value("last_dir", ""))
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open a folder of light curves", start_dir
        )
        if not directory:
            return
        settings.setValue("last_dir", directory)
        self._load_folder(directory)

    def _load_folder(self, directory: str | Path) -> None:
        """Enumerate and batch-load every light curve under ``directory``."""
        try:
            items = enumerate_sources(str(directory))
        except Exception as exc:  # noqa: BLE001 - surface any enumeration error
            self._warn(f"Could not read folder:\n{type(exc).__name__}: {exc}")
            return
        if not items:
            self._warn("No light-curve files found in that folder.")
            return
        self._controller.load_sources(items)
        name = Path(directory).name or str(directory)
        self.setWindowTitle(f"cuPeriod — {name} ({len(items)} sources)")

    def _load_demo_batch(self) -> None:
        self._controller.load_sources(demo_sources())
        self.setWindowTitle("cuPeriod — demo batch")

    def _on_sources_changed(self, labels: object) -> None:
        assert isinstance(labels, list)
        self._source_browser.set_sources(labels)
        self._sources_dock.setVisible(True)
        if labels:
            self._controller.select_source(0)

    def _exit_batch_mode(self) -> None:
        self._controller.state.mode = "single"
        self._controller.state.sources = None
        self._sources_dock.setVisible(False)

    def _on_run_requested(
        self, method: str, backend: str, settings: BaseSettings, n_peaks: int
    ) -> None:
        self._controller.run(
            method, backend, settings, n_peaks, band=self._controls.current_band()
        )

    def _on_prewhiten_requested(self, backend: str, settings: object) -> None:
        from cuperiod.core.config import PreWhitenSettings

        assert isinstance(settings, PreWhitenSettings)
        self._controller.run_prewhiten(
            backend, settings, band=self._controls.current_band()
        )

    # -- analysis switching ------------------------------------------------------
    def _on_analysis_changed(self, analysis: str) -> None:
        self._controller.set_analysis(analysis)  # type: ignore[arg-type]
        self._apply_analysis(analysis)
        self._spectrum.clear()
        self._peaks_table.clear()
        self._solution_panel.clear()
        self._spacing_panel.clear()
        # Only the *results* are stale — the light curve is unchanged, so drop the fold
        # rather than the curve itself (clear() would leave the phased panel empty until
        # the next load).
        self._phased.clear_fold()
        self._info.set_idle()
        self._stack.setCurrentIndex(0)
        self._show_status(
            "Pre-whitening selected — press “Run pre-whitening”."
            if analysis == "prewhiten"
            else "Periodogram selected — press “Compute periodogram”."
        )

    def _apply_analysis(self, analysis: str) -> None:
        """Show the docks that belong to ``analysis`` and raise the leading one."""
        prewhiten = analysis == "prewhiten"
        self._peaks_dock.setVisible(not prewhiten)
        self._solution_dock.setVisible(prewhiten)
        self._spacing_dock.setVisible(prewhiten)
        leading = self._solution_dock if prewhiten else self._peaks_dock
        leading.raise_()

    # -- controller signals ------------------------------------------------------
    def _on_lc_loaded(self, lc: object) -> None:
        # Clear the previous source's results until the new spectrum is computed.
        self._spectrum.clear()
        self._phased.clear()
        self._peaks_table.clear()
        self._solution_panel.clear()
        self._spacing_panel.clear()
        self._stack.setCurrentIndex(0)
        self._info.set_idle()

        self._controls.set_enabled(True)
        time = self._controller.current_time()
        baseline = float(time.max() - time.min()) if time.size else 0.0
        self._controls.set_curve_time(time)

        if isinstance(lc, MultiBandLightCurve):
            self._controls.set_bands(list(lc.band_names))
            self._controls.set_multiband(True)
            points = sum(band.n for band in lc.bands.values())
            note = f"Multiband curve loaded ({lc.n_bands} bands) — press Compute."
            detail = (
                f"Loaded: multiband · {lc.n_bands} bands · {points} points · "
                f"baseline {baseline:.1f} d"
            )
        elif isinstance(lc, LightCurve):
            self._controls.set_bands(None)
            self._controls.set_multiband(False)
            note = "Light curve loaded — press Compute periodogram."
            detail = (
                f"Loaded: {lc.n} points · baseline {baseline:.1f} d · {lc.domain.value}"
            )
        else:
            return
        self._placeholder.setText(
            f'{detail}\n\nPress "Compute periodogram" to see the spectrum.'
        )
        self._show_status(note)
        if self._controller.state.mode == "batch":
            self._controls.request_compute()  # compute on demand as sources scroll

    def _on_compute_started(self) -> None:
        self._info.set_busy(True, self._controls_method())
        self._show_status("Computing…")

    def _on_periodogram_ready(self, pg: object) -> None:
        from cuperiod.core.result import Periodogram

        assert isinstance(pg, Periodogram)
        self._hide_banner()
        self._info.set_result(pg)
        timing = self._timing_text()
        self._info.set_timing(timing)
        self._stack.setCurrentIndex(1)
        self._show_status(
            f"{pg.method} via {pg.backend} — best period "
            f"{pg.best_period():.6g} d — {timing}"
        )

    def _on_solution_ready(self, result: object) -> None:
        assert isinstance(result, PreWhitenResult)
        self._hide_banner()
        self._show_solution_spectrum(result)
        self._solution_panel.set_solution(result)
        self._spacing_panel.set_solution(result)
        self._stack.setCurrentIndex(1)
        timing = self._timing_text()
        self._info.set_timing(timing)
        strongest = (
            f"strongest {result.components[0].frequency:.6g} /d"
            if result.components
            else "no significant frequency"
        )
        self._show_status(
            f"Pre-whitening via {result.backend} — {result.n_components} components, "
            f"{strongest} — {timing}"
        )

    def _show_solution_spectrum(self, result: PreWhitenResult) -> None:
        """Draw the amplitude spectrum, its overlays, and the components.

        The pre-whitening result is wrapped in a :class:`Periodogram` so it flows
        through the existing spectrum view unchanged — full-resolution rendering, the
        draggable selection band, log axes, the crosshair readout, and CSV/PNG export
        all come for free, and the components arrive as ordinary peak markers.
        """
        spectrum = result.spectrum
        if spectrum is None:  # pragma: no cover - the GUI always keeps spectra
            self._spectrum.clear()
            return
        wrapped = Periodogram.from_spectrum(
            method=PREWHITEN_METHOD,
            backend=result.backend,
            frequency=spectrum.frequency,
            power=spectrum.amplitude,
            objective_sense="max",
            n_samples=result.n_samples,
            baseline=result.baseline,
        )
        self._spectrum.set_periodogram(wrapped)
        self._info.set_result(wrapped)
        if result.residual_spectrum is not None:
            self._spectrum.set_overlay(
                result.residual_spectrum.frequency, result.residual_spectrum.amplitude
            )
        if result.window is not None:
            # Scaled to the tallest peak, Period04-style: |W| itself tops out at 1.
            scale = (
                float(spectrum.amplitude.max()) if spectrum.amplitude.size else 1.0
            )
            self._spectrum.set_window(
                result.window.frequency, result.window.amplitude, scale=scale
            )
        # Markers annotate the *curve*, so their height is the plotted spectrum at the
        # component's frequency — not its fitted amplitude. The two are equal for a
        # well-separated mode but diverge whenever components are correlated (a HADS
        # harmonic and its yearly alias sidelobes trade amplitude in the joint fit),
        # and a marker floating above the curve claims a peak that is not there. The
        # fitted amplitude stays on the hover readout and in the Frequencies dock.
        heights = _spectrum_heights(
            spectrum.frequency,
            spectrum.amplitude,
            [component.frequency for component in result.components],
        )
        peaks = [
            Peak(
                period=component.period,
                frequency=component.frequency,
                power=float(heights[i]),
                rank=component.rank,
                extra={"snr": component.snr, "amplitude": component.amplitude},
            )
            for i, component in enumerate(result.components)
        ]
        self._spectrum.set_peaks(peaks)
        if peaks:
            self._spectrum.set_selected_period(peaks[0].period)

    def _timing_text(self) -> str:
        state = self._controller.state
        if state.last_from_cache:
            return "cached"
        elapsed = state.last_compute_ms
        return f"{elapsed / 1000:.2f} s" if elapsed >= 1000 else f"{elapsed:.0f} ms"

    def _on_compute_failed(self, message: str) -> None:
        self._info.set_idle()
        self._show_status(f"Compute failed: {message}")
        self._show_banner(message)

    # -- helpers -----------------------------------------------------------------
    def _controls_method(self) -> str:
        if self._controller.state.analysis == "prewhiten":
            return PREWHITEN_METHOD
        return self._controller.state.method

    def _show_status(self, message: str) -> None:
        status = self.statusBar()
        if status is not None:
            status.showMessage(message)

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "cuPeriod", message)

    def _show_banner(self, message: str) -> None:
        """Show a dismissable error banner (also auto-hides after a while)."""
        self._banner.setText(f"⚠  {message}")
        self._banner_container.setVisible(True)
        self._banner_timer.start(12000)

    def _hide_banner(self) -> None:
        self._banner_timer.stop()
        self._banner_container.setVisible(False)

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        apply_theme(self._app, self._theme)
        pal = palette(self._theme)
        self._spectrum.apply_theme(pal)
        self._phased.apply_theme(pal)
        self._raw.apply_theme(pal)
        self._spacing_panel.apply_theme(pal)
        QtCore.QSettings().setValue("theme", self._theme)

    def _compute_via_shortcut(self) -> None:
        """Ctrl+Return/Ctrl+Enter: request a compute if one would currently run."""
        if self._controls.can_compute():
            self._controls.request_compute()

    # -- lifecycle -----------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Cancel any in-flight compute and persist the window geometry/layout."""
        self._controller.shutdown()
        settings = QtCore.QSettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

    # -- drag and drop -------------------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._is_acceptable_drop(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime = event.mimeData()
        if not self._is_acceptable_drop(mime):
            event.ignore()
            return
        event.acceptProposedAction()
        paths = [Path(url.toLocalFile()) for url in mime.urls()]
        self._handle_dropped_paths(paths)

    @staticmethod
    def _is_acceptable_drop(mime: QtCore.QMimeData) -> bool:
        if not mime.hasUrls():
            return False
        urls = mime.urls()
        if not all(url.isLocalFile() for url in urls):
            return False
        paths = [Path(url.toLocalFile()) for url in urls]
        if len(paths) == 1 and paths[0].is_dir():
            return True
        return all(p.suffix.lower() in _FILE_EXTENSIONS for p in paths)

    def _handle_dropped_paths(self, paths: list[Path]) -> None:
        """Load dropped files/folders the same way as the toolbar's Open actions.

        A single directory is browsed like "Open folder…"; a single file is opened
        (with the usual preview) like "Open…"; multiple files are loaded as a batch.
        """
        if not paths:
            return
        if len(paths) == 1 and paths[0].is_dir():
            self._load_folder(paths[0])
            return
        if len(paths) == 1:
            self._load_file(paths[0])
            return
        valid = [p for p in paths if p.suffix.lower() in _FILE_EXTENSIONS]
        if not valid:
            self._warn("None of the dropped files are recognized light-curve files.")
            return
        if len(valid) == 1:
            self._load_file(valid[0])
            return
        try:
            items = enumerate_sources([str(p) for p in valid])
        except Exception as exc:  # noqa: BLE001 - surface any enumeration error
            self._warn(f"Could not load dropped files:\n{type(exc).__name__}: {exc}")
            return
        if not items:
            self._warn("No light-curve files found among the dropped files.")
            return
        self._controller.load_sources(items)
        self.setWindowTitle(f"cuPeriod — dropped batch ({len(items)} sources)")


__all__ = ["MainWindow"]
