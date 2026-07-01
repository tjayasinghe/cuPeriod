"""The top-level application window: toolbar, controls, info bar, and work area.

Assembles the controller and all panels and wires them together through the controller's
signals: controls → run, results → spectrum/phased/peaks, source browser → batch scroll.
The chosen theme is remembered across sessions via ``QSettings``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
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
from cuperiod.gui.widgets.source_panel import SourceBrowser
from cuperiod.gui.widgets.spectrum_view import SpectrumView

_FILE_FILTER = (
    "Light curves (*.csv *.ecsv *.fits *.fit *.fz *.parquet *.pq "
    "*.tsv *.tab *.dat *.txt)"
)


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
        self.setWindowTitle("cuPeriod — periodogram explorer")
        self.resize(1360, 860)
        self.setMinimumSize(1024, 680)

        self._controller = AppController(self)
        self._controls = ControlsPanel()
        self._info = RunInfoBar()
        self._spectrum = SpectrumView(self._theme)
        self._phased = PhasedView(self._theme)
        self._raw = RawLightCurveView(self._theme)
        self._peaks_table = PeaksTable()
        self._source_browser = SourceBrowser()

        self._banner_timer = QtCore.QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._hide_banner)

        self._build_toolbar()
        self._build_central()
        self._build_peaks_dock()
        self._build_sources_dock()
        self._connect()
        self._show_status("Open a light curve or load a demo to begin.")

    # -- construction ------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Main")
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
        demo_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
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

        self._banner = QtWidgets.QLabel("")
        self._banner.setObjectName("error")
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        self._center_layout.addWidget(self._banner)

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

    def _build_peaks_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Peaks", self)
        dock.setWidget(self._peaks_table)
        feature = QtWidgets.QDockWidget.DockWidgetFeature
        dock.setFeatures(
            feature.DockWidgetMovable
            | feature.DockWidgetFloatable
            | feature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_sources_dock(self) -> None:
        self._sources_dock = QtWidgets.QDockWidget("Sources", self)
        self._sources_dock.setWidget(self._source_browser)
        feature = QtWidgets.QDockWidget.DockWidgetFeature
        self._sources_dock.setFeatures(
            feature.DockWidgetMovable
            | feature.DockWidgetFloatable
            | feature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._sources_dock)
        self._sources_dock.setVisible(False)

    def _connect(self) -> None:
        self._controls.run_requested.connect(self._on_run_requested)
        ctl = self._controller
        ctl.lc_loaded.connect(self._on_lc_loaded)
        ctl.compute_started.connect(self._on_compute_started)
        ctl.busy_changed.connect(self._info.set_busy)
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

    # -- actions -----------------------------------------------------------------
    def _open_file(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open light curve", "", _FILE_FILTER
        )
        if not path_str:
            return
        path = Path(path_str)
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
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open a folder of light curves"
        )
        if not directory:
            return
        try:
            items = enumerate_sources(directory)
        except Exception as exc:  # noqa: BLE001 - surface any enumeration error
            self._warn(f"Could not read folder:\n{type(exc).__name__}: {exc}")
            return
        if not items:
            self._warn("No light-curve files found in that folder.")
            return
        self._controller.load_sources(items)
        self.setWindowTitle(f"cuPeriod — {directory} ({len(items)} sources)")

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

    # -- controller signals ------------------------------------------------------
    def _on_lc_loaded(self, lc: object) -> None:
        # Clear the previous source's results until the new spectrum is computed.
        self._spectrum.clear()
        self._phased.clear()
        self._peaks_table.clear()
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
            f"{detail}\n\nPress “Compute periodogram” to see the spectrum."
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
        return self._controller.state.method

    def _show_status(self, message: str) -> None:
        status = self.statusBar()
        if status is not None:
            status.showMessage(message)

    def _warn(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "cuPeriod", message)

    def _show_banner(self, message: str) -> None:
        """Show a non-blocking error banner (auto-hides after a few seconds)."""
        self._banner.setText(f"⚠  {message}")
        self._banner.setVisible(True)
        self._banner_timer.start(7000)

    def _hide_banner(self) -> None:
        self._banner.setVisible(False)

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        apply_theme(self._app, self._theme)
        pal = palette(self._theme)
        self._spectrum.apply_theme(pal)
        self._phased.apply_theme(pal)
        self._raw.apply_theme(pal)
        QtCore.QSettings().setValue("theme", self._theme)


__all__ = ["MainWindow"]
