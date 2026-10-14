"""The top-level application window: menus, toolbar, controls, plots and docks.

Assembles the controller and every panel and wires them together through the
controller's signals: controls → run, results → spectrum/phased/peaks, source browser →
batch scroll, and — in pre-whitening — the live session: the editable Frequencies
table with its session bar, the Subsets, Variations and History docks, picking in the
spectrum and point deletion in the light curve. Widgets never call each other: they
emit signals, the window hands them to :class:`~cuperiod.gui.state.AppController`, and
the controller's signals update the widgets.

Every user command is one :class:`~cuperiod.gui.commands.CommandRegistry` action. The
menu bar (File, Edit, Pre-whitening, Data, View, Help, each with a mnemonic) is built
from the registry plus a few window commands (open folder, recent files, demos, the
dock toggles, theme and zoom, Help); the same actions drive the session bar, the
context letters and the F1 sheet, and their enabled state follows the controller:
mutating commands are off while a long job runs, Stop (Esc) is on only then.

Two analyses share the shell. Switching the controls panel's **Analysis** picker swaps
which docks are on show — *Peaks* for a periodogram; *Frequencies*, *Subsets*,
*Variations*, *History* and *Period spacing* for pre-whitening — while the spectrum,
phased and raw views, the source browser and every load path stay as they were. Each
analysis keeps its last result: switching back shows it again (from the cache, or the
session the bridge kept) instead of a blank plot. The docks are tabbed together, so the
swap never changes the window layout the user has arranged; View lists every dock, so
a closed one comes back, and *Reset layout* restores the default arrangement.

A loaded curve is shown at once, before anything is computed: the plot area starts in
a *preview* that draws the raw light curve full size under a line saying what was
loaded and which button to press. The first result (or, in pre-whitening, the lazily
opened session's data spectrum) swaps in the spectrum and the phased view.

Every prompt goes through the controller's :class:`~cuperiod.gui.dialogs.DialogProvider`
and is shown with ``open()``: the import dialog (columns, scale, delimiter), file and
folder pickers, the Save / Discard / Cancel question for unsaved sessions, the
keyboard-shortcut sheet and About. Closing, *New session*, opening a curve, a batch or a
project ask that question first when a session with unsaved work would be dropped. The
title bar reads ``data — project.cupw.json* — cuPeriod``; the ``*`` marks unsaved work.

Every string that can carry file content — a load error quoting the file's column
headers, a source label, a file name — is shown as plain text, never as Qt rich text,
so a header such as ``<img src=...>`` reads as those characters instead of being
rendered (and fetched).

The window size follows the screen: the first launch fills most of the available
area (never more), the minimum size keeps every pane usable, and the layout — dock
arrangement, splitter positions, the analysis and the options — is restored on the
next launch, except an empty Sources dock or a hidden toolbar.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Peak, Periodogram
from cuperiod.gui.appinfo import version_label
from cuperiod.gui.commands import CommandRegistry
from cuperiod.gui.dialogs import DialogProvider, QtDialogProvider
from cuperiod.gui.loaders import (
    LOAD_NOTES_KEY,
    ImportChoice,
    ReadOptions,
    demo_sources,
    enumerate_sources,
    load_path,
    preview_file,
)
from cuperiod.gui.models import COMBINED_BAND, STACKED_BAND, LoadedCurve, SourceItem
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, pg
from cuperiod.gui.refusals import Refusal, gui_wording, refusal_offer, refusal_text
from cuperiod.gui.state import (
    PROJECT_FILTER,
    PROJECT_PREFIX,
    PROJECT_SUFFIX,
    AppController,
)
from cuperiod.gui.theme import (
    ThemeName,
    ThemePalette,
    app_icon,
    apply_theme,
    font_scale,
    palette,
    step_font_scale,
)
from cuperiod.gui.viewmodels import SessionView, marker_states
from cuperiod.gui.widgets.controls_panel import ControlsPanel, method_description
from cuperiod.gui.widgets.frequency_table import (
    TABLE_COMMANDS,
    FrequencyTable,
    split_epoch,
    split_group,
)
from cuperiod.gui.widgets.history_panel import HistoryPanel
from cuperiod.gui.widgets.info_bar import RunInfoBar
from cuperiod.gui.widgets.peaks_panel import PeaksTable, statistic_name
from cuperiod.gui.widgets.phased_view import PhasedView
from cuperiod.gui.widgets.preview_dialog import UNDETECTED_ADVICE
from cuperiod.gui.widgets.rawlc_view import RawLightCurveView
from cuperiod.gui.widgets.session_bar import SessionBar
from cuperiod.gui.widgets.solution_panel import SolutionPanel
from cuperiod.gui.widgets.source_panel import SourceBrowser, rich_tooltip
from cuperiod.gui.widgets.spacing_panel import SpacingPanel
from cuperiod.gui.widgets.spectrum_view import PREWHITEN_METHOD, SpectrumView
from cuperiod.gui.widgets.subsets_panel import SubsetsPanel
from cuperiod.gui.widgets.variation_panel import VariationPanel
from cuperiod.prewhiten.result import PreWhitenResult

_logger = logging.getLogger(__name__)

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

#: What the empty window says. Drag and drop works everywhere in the window but has
#: no visible control of its own, so the welcome text is where it gets mentioned.
_WELCOME_TEXT = (
    "Open a light curve (Ctrl+O) or a folder of them, or try Load demo —\n"
    "or drop files or a folder onto this window."
)

#: A curve needs at least this many rows with a finite time and brightness to be
#: analysed at all (one point has no baseline).
_MIN_USABLE_POINTS = 2

#: How much of an error message a dialog or the banner shows before the rest moves to
#: "Show Details…" / the tooltip. Some readers put a whole format table in their
#: exception text, which made a warning box taller than a laptop screen.
_SUMMARY_MAX_LINES = 6
_SUMMARY_MAX_CHARS = 500

#: Plain-language explanations, with a next step, for the failures users actually
#: meet. The raw ``Type: message`` text stays available as the details.
_FAILURE_ADVICE: dict[str, str] = {
    "ColumnResolutionError": (
        "cuPeriod could not tell which columns hold the time and the brightness. "
        + UNDETECTED_ADVICE
    ),
    "InsufficientDataError": (
        "There is not enough usable data for this analysis: it needs more rows with "
        "a finite time, brightness and positive error, or a longer time baseline."
    ),
    "GridTooLargeError": (
        "The trial frequency grid would be too large to compute. Set a lower "
        "maximum frequency, or fewer samples per peak, under Options."
    ),
    "BackendUnavailableError": (
        "The selected backend cannot run on this machine. Choose auto or cpu under "
        "Backend."
    ),
    "MemoryError": (
        "The computation ran out of memory. Lower the maximum frequency or the "
        "samples per peak under Options."
    ),
    "FileNotFoundError": "The file could not be found; it may have been moved.",
    "PermissionError": "The file could not be read: permission denied.",
    "UnicodeDecodeError": "The file is not a text table cuPeriod can read.",
}

_FAILURE_TYPE = re.compile(r"\b(" + "|".join(_FAILURE_ADVICE) + r")\b:")

#: Where Help ▸ Documentation goes (the project's Read the Docs site).
DOCS_URL = "https://cuperiod.readthedocs.io"

#: How many files File ▸ Open recent remembers.
MAX_RECENT = 8

#: The window's designed size; the first launch takes it or most of the screen.
_DESIGN_SIZE = (1440, 900)
#: The smallest size at which every pane stays usable (clamped to the screen).
_MIN_SIZE = (900, 600)
#: The Controls column never gets narrower than this (it grows with the zoom).
_CONTROLS_MIN_WIDTH = 300
#: What the result panes' tab bars say to a mouse or screen-reader user.
_DOCK_TABS_TIP = (
    "Result panes: Left/Right switch panes while this bar has the focus; "
    "View ▸ Panes shows a pane or brings a closed one back"
)


def window_minimum(needed: QtCore.QSize, screen: QtCore.QSize) -> tuple[int, int]:
    """The window minimum: the panes' ``needed`` minimum and at least
    :data:`_MIN_SIZE`, but never more than the ``screen`` can show."""
    width = min(max(_MIN_SIZE[0], needed.width()), screen.width())
    height = min(max(_MIN_SIZE[1], needed.height()), screen.height())
    return width, height


#: QSettings keys.
_KEY_RECENT = "recent_files"
_KEY_IMPORT = "last_import"
_KEY_CONTROLS = "controls"
_KEY_SPLITTERS = "splitters"
_KEY_RAW = "show_raw"

#: The docks of each analysis (object names), in tab order.
_PERIODOGRAM_DOCKS = ("peaks_dock",)
_PREWHITEN_DOCKS = (
    "solution_dock",
    "subsets_dock",
    "variations_dock",
    "history_dock",
    "spacing_dock",
)


def _shorten(text: str) -> str:
    """At most ``_SUMMARY_MAX_LINES`` lines / ``_SUMMARY_MAX_CHARS`` characters."""
    lines = text.splitlines() or [""]
    short = "\n".join(lines[:_SUMMARY_MAX_LINES])
    if len(short) > _SUMMARY_MAX_CHARS:
        short = short[: _SUMMARY_MAX_CHARS - 1].rstrip()
    if short != text.strip():
        short = short.rstrip() + " …"
    return short


def explain_failure(message: str) -> tuple[str, str]:
    """Split a failure ``message`` into ``(summary, details)`` for display.

    ``message`` is what the controller and the loaders report — ``"Could not load
    star: ColumnResolutionError: could not resolve a time column; pass ColumnMap(...)
    ..."`` or a worker's ``"InsufficientDataError: ..."``. A known exception type is
    replaced by plain advice with a next step (keeping any ``Could not load ...:``
    context in front of it); anything else is shortened to a few lines. ``details`` is
    always the full original text, or ``""`` when the summary already is all of it.
    """
    text = message.strip()
    match = _FAILURE_TYPE.search(text)
    if match is not None:
        context = text[: match.start()].strip().rstrip(":").strip()
        advice = _FAILURE_ADVICE[match.group(1)]
        summary = f"{context}: {advice}" if context else advice
        return summary, text
    summary = _shorten(text)
    return summary, ("" if summary == text else text)


#: The application's name, last in every window title.
APP_TITLE = "cuPeriod"


def window_title(data: str, project: str = "") -> str:
    """The main window's title: ``star.csv — star.cupw.json[*] — cuPeriod``.

    The document comes first and the application's name once, last; ``[*]`` is
    Qt's placeholder for the unsaved-work ``*`` (see
    :meth:`QWidget.setWindowModified`). A name that contains ``[*]`` itself is
    escaped, so a file called ``a[*].csv`` cannot move the marker. With no data
    the title just names the application.
    """
    names = [name.replace("[*]", "[*][*]") for name in (data, project) if name]
    if not names:
        return f"{APP_TITLE} — periodogram & pre-whitening explorer"
    return " — ".join(names) + f"[*] — {APP_TITLE}"


def _saved_ui_scale() -> float | None:
    """The zoom saved by the last session, or None (keep the current one)."""
    value = QtCore.QSettings().value("ui_scale")
    if not isinstance(value, int | float | str):
        return None
    try:
        scale = float(value)
    except ValueError:
        return None
    return scale if math.isfinite(scale) and scale > 0.0 else None


def _usable_points(lc: LoadedCurve) -> int:
    """Rows with a finite time and brightness, over all bands."""
    bands = list(lc.bands.values()) if isinstance(lc, MultiBandLightCurve) else [lc]
    return sum(
        int(np.count_nonzero(np.isfinite(band.time) & np.isfinite(band.value)))
        for band in bands
    )


def _total_rows(lc: LoadedCurve) -> int:
    if isinstance(lc, MultiBandLightCurve):
        return sum(band.n for band in lc.bands.values())
    return lc.n


def _curve_domain(lc: object) -> Domain | None:
    """The brightness domain of a curve (its first band's for multiband data)."""
    if isinstance(lc, MultiBandLightCurve):
        bands = list(lc.bands.values())
        return bands[0].domain if bands else None
    if isinstance(lc, LightCurve):
        return lc.domain
    return None


def prewhiten_mode(controls: ControlsPanel) -> bool:
    """Whether the controls are on the pre-whitening analysis."""
    return controls.analysis == "prewhiten"


def _string_list(value: object) -> list[str]:
    """A QSettings list: one entry comes back as a bare string on some backends."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(v) for v in value if isinstance(v, str) and v]
    return []


def import_choice_to_json(choice: ImportChoice) -> str:
    """An :class:`ImportChoice` as JSON for QSettings (the next import's presets)."""
    columns = dataclasses.asdict(choice.columns)
    read = None if choice.read is None else dataclasses.asdict(choice.read)
    domain = None if choice.domain is None else choice.domain.value
    return json.dumps({"columns": columns, "domain": domain, "read": read})


def import_choice_from_json(text: object) -> ImportChoice | None:
    """The inverse of :func:`import_choice_to_json`; None for anything malformed."""
    if not isinstance(text, str) or not text:
        return None
    try:
        data = json.loads(text)
        columns_raw = data["columns"]
        columns = ColumnMap(
            **{
                role: columns_raw.get(role)
                for role in ("time", "value", "error", "band")
                if isinstance(columns_raw.get(role), str)
            }
        )
        domain = Domain(data["domain"]) if data.get("domain") else None
        read_raw = data.get("read")
        read = (
            ReadOptions(
                delimiter=read_raw.get("delimiter")
                if isinstance(read_raw.get("delimiter"), str)
                else None,
                header=bool(read_raw.get("header", True)),
            )
            if isinstance(read_raw, dict)
            else None
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return ImportChoice(columns=columns, domain=domain, read=read)


def assign_mnemonics(menu: QtWidgets.QMenu) -> None:
    """Give every item of ``menu`` a unique Alt mnemonic (word initials first).

    Items that already carry one keep it. A first pass gives word initials; the
    items left over then take any free letter of their text, those with the fewest
    free letters first ("Stop" next to "Step" and "Add peak" has only one left), so
    each item of an open menu is one key press away whenever its letters allow.
    """
    actions = [a for a in menu.actions() if not a.isSeparator() and a.text()]
    used = {
        m.group(1).lower()
        for a in actions
        if (m := re.search(r"&([^&\s])", a.text())) is not None
    }
    pending = [a for a in actions if not re.search(r"&[^&\s]", a.text())]

    def give(action: QtGui.QAction, candidates: list[int]) -> bool:
        text = action.text()
        for index in candidates:
            letter = text[index].lower()
            if letter not in used:
                used.add(letter)
                action.setText(text[:index] + "&" + text[index:])
                return True
        return False

    left: list[QtGui.QAction] = []
    for action in pending:
        initials = [m.start() for m in re.finditer(r"\b[A-Za-z0-9]", action.text())]
        if not give(action, initials):
            left.append(action)

    def free(action: QtGui.QAction) -> int:
        return len({c.lower() for c in action.text() if c.isalnum()} - used)

    while left:
        left.sort(key=free)
        action = left.pop(0)
        text = action.text()
        give(action, [i for i, c in enumerate(text) if c.isalnum()])


def _glyph_icon(kind: str, color: str, size: int = 16) -> QtGui.QIcon:
    """A theme-coloured title-bar glyph (``"close"`` or ``"float"``).

    Drawn here rather than taken from the style: the style's title-bar pixmaps are
    dark on dark under the dark theme.
    """
    pixmap = QtGui.QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidthF(1.6)
    painter.setPen(pen)
    m = size * 0.25
    if kind == "close":
        painter.drawLine(QtCore.QPointF(m, m), QtCore.QPointF(size - m, size - m))
        painter.drawLine(QtCore.QPointF(size - m, m), QtCore.QPointF(m, size - m))
    else:
        s = size * 0.42
        painter.drawRect(QtCore.QRectF(m + 2.0, m, s, s))
        painter.drawRect(QtCore.QRectF(m, m + 2.0 + s * 0.35, s, s))
    painter.end()
    return QtGui.QIcon(pixmap)


#: The phased and raw views sit side by side only from this width on: each needs
#: about 450 px for its toolbar to fit on one or two rows and leave room to plot.
#: Narrower, they take turns under the Phased / Raw light curve tabs.
_SIDE_BY_SIDE_WIDTH = 900

#: The tabs of the light-curve views (see _build_lc_tabs).
_PHASED_TAB, _RAW_TAB = 0, 1
_LC_TABS_TIP = (
    "The pane is too narrow for the phased and the raw light curve side by side: "
    "pick one here (Left/Right while the bar has the focus). Widen the window or "
    "narrow the panes on the right to see both."
)

#: How the lower pane shows the light curves: the raw curve alone before a result,
#: the fold alone, the two side by side, or the two as tabs.
LcLayout = Literal["preview", "single", "side", "tabs"]


def _chrome_height(view: QtWidgets.QWidget, width: int) -> int:
    """The height ``view`` needs besides its plot at ``width``.

    Its toolbars and labels (the rows a wrapping toolbar takes at that width
    included) and the spacing between them, from their size hints: the views need
    not be laid out, or even shown, to be measured.
    """
    layout = view.layout()
    if layout is None:
        return 0
    margins = layout.contentsMargins()
    inner = max(1, width - margins.left() - margins.right())
    total = margins.top() + margins.bottom()
    shown = 0
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None or item.isEmpty():
            continue
        shown += 1
        if isinstance(item.widget(), pg.PlotWidget):
            continue
        total += (
            item.heightForWidth(inner)
            if item.hasHeightForWidth()
            else item.sizeHint().height()
        )
    return total + max(0, shown - 1) * max(0, layout.spacing())


#: The narrowest a result dock gets by default; wider panels scroll sideways.
_DOCK_MIN_WIDTH = 300


def _scrolling(widget: QtWidgets.QWidget, name: str) -> QtWidgets.QScrollArea:
    """``widget`` in a scroll area, so a dock can be narrower than its panel.

    Tabbed docks share one width, at least the widest panel's minimum: the Subsets
    dock alone (757 px) used to push the whole tab group over half of a 1366 px
    window and crush the plots. Inside a scroll area a wide panel scrolls sideways
    instead, and the dock keeps the width it is given.
    """
    area = QtWidgets.QScrollArea()
    area.setObjectName(widget.objectName() + "_scroll" if widget.objectName() else "")
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # a container, not a Tab stop
    area.setAccessibleName(name)
    area.setMinimumWidth(min(_DOCK_MIN_WIDTH, widget.minimumSizeHint().width()))
    return area


def _dock_content(dock: QtWidgets.QDockWidget) -> QtWidgets.QWidget | None:
    """The panel of ``dock`` (inside its scroll area)."""
    content = dock.widget()
    if isinstance(content, QtWidgets.QScrollArea):
        return content.widget()
    return content


class _DockTitleBar(QtWidgets.QWidget):
    """A dock title with 24 x 24 px Float and Close buttons (WCAG 2.5.8).

    The style's own dock buttons are 18 x 14 px and cannot be enlarged from the
    stylesheet. Mouse presses on the title itself are left to the dock, so dragging and
    double-click floating work as before. The buttons take no Tab focus: View lists
    every dock (and Float is a mouse convenience).
    """

    def __init__(self, dock: QtWidgets.QDockWidget) -> None:
        super().__init__(dock)
        self.setObjectName("dockTitleBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._dock = dock
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 2, 2, 2)
        row.setSpacing(2)
        self._label = QtWidgets.QLabel(dock.windowTitle())
        self._label.setObjectName("dockTitle")
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        row.addWidget(self._label, 1)
        title = dock.windowTitle()
        self._float = self._button(
            f"Float {title}", f"Detach the {title} pane into its own window, or "
            "dock it back (or double-click the title)"
        )
        self._float.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        row.addWidget(self._float)
        self._close = self._button(
            f"Close {title}", f"Close the {title} pane (View brings it back)"
        )
        self._close.clicked.connect(dock.close)
        row.addWidget(self._close)
        dock.windowTitleChanged.connect(self._label.setText)

    def _button(self, name: str, tip: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setObjectName("dockTitleButton")
        button.setFixedSize(24, 24)
        button.setIconSize(QtCore.QSize(16, 16))
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setAccessibleName(name)
        button.setToolTip(tip)
        button.setStatusTip(tip)
        return button

    def apply_theme(self, pal: ThemePalette) -> None:
        color = pal.plot_fg
        self._float.setIcon(_glyph_icon("float", color))
        self._close.setIcon(_glyph_icon("close", color))

    @property
    def buttons(self) -> tuple[QtWidgets.QToolButton, QtWidgets.QToolButton]:
        return self._float, self._close


class MainWindow(QtWidgets.QMainWindow):
    """Main window shell: menus, toolbar, controls, plots, docks and status bar."""

    def __init__(
        self, app: QtWidgets.QApplication, *, theme: ThemeName = "dark"
    ) -> None:
        super().__init__()
        #: A tidy of the dock tab bars is queued (see _style_dock_tabs).
        self._tabs_pending = False
        #: A fit of the light-curve pane is queued (see _fit_lc_area).
        self._lc_fit_queued = False
        self._app = app
        saved = str(QtCore.QSettings().value("theme", theme))
        self._theme: ThemeName = "light" if saved == "light" else "dark"
        apply_theme(app, self._theme, font_scale=_saved_ui_scale())

        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)

        #: The "Loaded: N points · baseline ..." line of the current curve, or None.
        self._loaded_detail: str | None = None
        #: Whether a result (spectrum + fold) is on screen, as opposed to the preview.
        self._showing_result = False
        #: "Show Raw LC" per analysis (the preview overrides it). Choosing a point
        #: tool (Data menu, X / G) shows it, since that is where points are picked.
        self._raw_pref = {"periodogram": False, "prewhiten": False}
        #: How the lower pane shows the light curves now (see _fit_lc_area).
        self._lc_layout: LcLayout = "preview"
        #: The spectrum / light-curve split the user had before the raw view came
        #: in (it gets more of the height then); put back when it goes again.
        self._split_before_raw: list[int] | None = None
        #: The split could not be shared yet (the pane had no height): on the next fit.
        self._split_pending = False
        #: The user dragged the spectrum / light-curve split since it was last
        #: shared: it stays where they put it until the layout changes.
        self._split_user = False
        #: Bumped on every load, so a batch selection can tell whether it loaded.
        self._load_count = 0
        #: The batch source the controller just reported as unloadable; the failure
        #: message that follows it is a load failure, not a failed run.
        self._failed_source: int | None = None
        self._demo_menu_filled = False
        #: What the title bar calls the data on show (a file, a demo, a batch).
        self._data_name = ""
        #: A long job (or a one-shot run) is in flight.
        self._busy = False
        #: The run requested last in each analysis, with the curve it was for, so a
        #: switch back to that analysis shows its result again (UX-17).
        self._last_runs: dict[str, tuple[tuple[str, str], Callable[[], None]]] = {}
        #: The last pre-whitening result per curve, for the Period spacing dock.
        self._last_solution: tuple[tuple[str, str], PreWhitenResult] | None = None
        #: The controller is switching the analysis (a project opened): do not try to
        #: restore a previous result on top of what it is about to show.
        self._controller_switch = False
        #: The banner's action ("Choose columns…" for a file that would not load).
        self._banner_action_slot: Callable[[], None] | None = None

        self._controller = AppController(self)
        self._dialogs: DialogProvider = QtDialogProvider(self)
        self._controller.set_dialog_provider(self._dialogs)
        self._controls = ControlsPanel()
        self._info = RunInfoBar()
        self._spectrum = SpectrumView(self._theme)
        self._phased = PhasedView(self._theme)
        self._raw = RawLightCurveView(self._theme)
        self._peaks_table = PeaksTable()
        self._solution_panel = SolutionPanel()
        self._session_bar = SessionBar()
        self._history_panel = HistoryPanel()
        self._subsets_panel = SubsetsPanel()
        self._variation_panel = VariationPanel(self._theme)
        self._spacing_panel = SpacingPanel(self._theme)
        self._source_browser = SourceBrowser()
        self._commands = CommandRegistry(self)
        self._title_bars: list[_DockTitleBar] = []

        self._build_toolbar()
        self._build_central()
        self._build_result_docks()
        self._build_sources_dock()
        self._build_shortcuts()
        self._build_menus()
        self._attach_commands()
        self._restore_controls()
        self._connect()
        self._restore_window_state()
        self._apply_analysis(self._controls.analysis)
        self._controller.set_analysis(self._controls.analysis)  # type: ignore[arg-type]
        # The application sheet is already applied (above); only the pieces that
        # paint their own theme colours need the palette. A second apply_theme here
        # would re-polish every widget of the application once more.
        self._theme_pieces(palette(self._theme))
        self._info.set_idle("no light curve loaded")
        self._share_dialogs()
        self._update_title()
        self._sync_commands()
        self._show_status("Open a light curve or load a demo to begin.")

    # -- public ------------------------------------------------------------------------
    @property
    def controller(self) -> AppController:
        """The controller every widget of this window talks to."""
        return self._controller

    @property
    def commands(self) -> CommandRegistry:
        """The command registry behind the menus, buttons and shortcuts."""
        return self._commands

    @property
    def dialogs(self) -> DialogProvider:
        """Where every prompt of the window and its controller goes."""
        return self._dialogs

    def set_dialog_provider(self, provider: DialogProvider) -> None:
        """Route every prompt (files, import, unsaved work, sheets) to ``provider``
        (tests pass a :class:`~cuperiod.gui.dialogs.FakeDialogProvider`)."""
        self._dialogs = provider
        self._controller.set_dialog_provider(provider)
        self._share_dialogs()

    def _share_dialogs(self) -> None:
        """The views' Export… prompts go through the same provider (open(), never a
        nested event loop; scripted in tests)."""
        for view in (
            self._spectrum,
            self._phased,
            self._peaks_table,
            self._solution_panel,
        ):
            view.set_dialog_provider(self._dialogs)

    # =================================================================================
    # construction
    # =================================================================================
    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Main")
        toolbar.setObjectName("main_toolbar")  # required for saveState()
        toolbar.setMovable(False)
        # The only toolbar: hiding it from the dock/toolbar popup used to stick across
        # launches with no visible way back (UX-08).
        toolbar.toggleViewAction().setVisible(False)
        self.addToolBar(toolbar)
        self._toolbar = toolbar

        title = QtWidgets.QLabel("  cuPeriod  ")
        title.setObjectName("heading")
        toolbar.addWidget(title)
        version = QtWidgets.QLabel(version_label() + "   ")
        version.setObjectName("muted")
        version.setTextFormat(Qt.TextFormat.PlainText)
        version.setToolTip("Installed version and git revision")
        toolbar.addWidget(version)

        self._folder_action = QtGui.QAction("Open folder…", self)
        self._folder_action.setObjectName("action_file_open_folder")
        self._folder_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+F"))
        self._folder_action.setToolTip(
            "Open a folder of light curves for batch browsing  (Ctrl+Shift+F)"
        )
        self._folder_action.setStatusTip(
            "Open a folder of light curves for batch browsing"
        )
        self._folder_action.triggered.connect(self._open_folder)
        self.addAction(self._folder_action)

        toolbar.addAction(self._commands.action("file.open_curve"))
        toolbar.addAction(self._folder_action)

        demo_button = QtWidgets.QToolButton()
        demo_button.setText("Load demo")
        demo_button.setToolTip(
            "Load a bundled example light curve, or browse a demo batch"
        )
        demo_button.setAccessibleName("Load demo")
        demo_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        demo_button.setMenu(self._build_demo_menu())
        toolbar.addWidget(demo_button)

        spacer = QtWidgets.QWidget()
        spacer.setObjectName("toolbarSpacer")  # transparent in the theme sheets
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        toolbar.addWidget(spacer)

        self._theme_action = QtGui.QAction("Toggle dark / light theme", self)
        self._theme_action.setObjectName("action_view_theme")
        self._theme_action.setIconText("Theme")
        self._theme_action.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self._theme_action.setToolTip("Toggle dark / light theme  (Ctrl+T)")
        self._theme_action.setStatusTip("Toggle dark / light theme")
        self._theme_action.triggered.connect(self._toggle_theme)
        self.addAction(self._theme_action)
        toolbar.addAction(self._theme_action)

    def _build_demo_menu(self) -> QtWidgets.QMenu:
        """The Load demo menu, filled in the first time it opens.

        Listing the demos reads the bundled ASAS-SN parquet, which imports pandas —
        about 0.8 s that used to sit on the start-up path before the window appeared.
        """
        menu = QtWidgets.QMenu("Load demo", self)
        menu.setObjectName("menu_demo")
        self._demo_menu = menu
        self._demo_separator = menu.addSeparator()
        batch = menu.addAction("Batch: browse all demo sources")
        batch.setStatusTip("Browse every bundled demo light curve as a batch")
        batch.triggered.connect(
            lambda: self._controller.confirm_then(self._load_demo_batch)
        )
        menu.aboutToShow.connect(self._populate_demo_menu)
        return menu

    def _populate_demo_menu(self) -> None:
        if self._demo_menu_filled:
            return
        self._demo_menu_filled = True
        try:
            items = demo_sources()
        except Exception:  # noqa: BLE001 - a bad demo file must not break the menu
            _logger.warning("could not list the demo sources", exc_info=True)
            unavailable = QtGui.QAction("Demo files could not be read", self._demo_menu)
            unavailable.setEnabled(False)
            self._demo_menu.insertAction(self._demo_separator, unavailable)
            return
        for item in items:
            action = QtGui.QAction(item.label, self._demo_menu)
            action.setStatusTip(f"Load the demo light curve {item.label}")
            action.triggered.connect(
                lambda _=False, it=item: self._controller.confirm_then(
                    lambda: self._load_demo(it)
                )
            )
            self._demo_menu.insertAction(self._demo_separator, action)

    def _build_central(self) -> None:
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("splitter_main")
        splitter.setChildrenCollapsible(False)
        self._main_split = splitter
        self._controls.setMinimumWidth(_CONTROLS_MIN_WIDTH)
        self._controls.setMaximumWidth(460)
        splitter.addWidget(self._controls)

        center = QtWidgets.QWidget()
        center.setMinimumWidth(360)
        self._center_layout = QtWidgets.QVBoxLayout(center)
        # Tight above and below: in a 768 px tall window every pixel of chrome is a
        # pixel less of plot (the info bar keeps a margin of its own).
        self._center_layout.setContentsMargins(8, 4, 8, 4)
        self._center_layout.setSpacing(4)
        self._center_layout.addWidget(self._info)

        self._banner_container = QtWidgets.QWidget()
        banner_layout = QtWidgets.QHBoxLayout(self._banner_container)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        banner_layout.setSpacing(4)
        self._banner = QtWidgets.QLabel("")
        self._banner.setObjectName("error")
        self._banner.setWordWrap(True)
        # Error text quotes the user's file (column headers, cell values): plain text
        # only, or a header like '<img src=...>' would be rendered as HTML.
        self._banner.setTextFormat(Qt.TextFormat.PlainText)
        # Selectable (and so reachable with Tab) to read it out or copy it.
        self._banner.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._banner.setAccessibleName("Error")
        banner_layout.addWidget(self._banner, 1)
        # A next step the user can take right here (e.g. choose a file's columns).
        self._banner_action = QtWidgets.QPushButton("")
        self._banner_action.setVisible(False)
        self._banner_action.clicked.connect(self._on_banner_action)
        banner_layout.addWidget(self._banner_action, 0, Qt.AlignmentFlag.AlignTop)
        self._banner_dismiss = QtWidgets.QToolButton()
        self._banner_dismiss.setText("✕")
        self._banner_dismiss.setAutoRaise(True)
        self._banner_dismiss.setToolTip("Dismiss this message")
        self._banner_dismiss.setAccessibleName("Dismiss error")
        self._banner_dismiss.clicked.connect(self._hide_banner)
        banner_layout.addWidget(self._banner_dismiss, 0, Qt.AlignmentFlag.AlignTop)
        self._banner_container.setVisible(False)
        self._center_layout.addWidget(self._banner_container)

        self._stack = QtWidgets.QStackedWidget()
        self._welcome = QtWidgets.QLabel(_WELCOME_TEXT)
        self._welcome.setObjectName("muted")
        self._welcome.setTextFormat(Qt.TextFormat.PlainText)
        self._welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._welcome)  # index 0: nothing loaded
        self._stack.addWidget(self._build_plot_area())  # index 1: curve / results
        self._center_layout.addWidget(self._stack, 1)

        splitter.addWidget(center)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1000])
        self.setCentralWidget(splitter)

    def _build_plot_area(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        # The preview's guidance line: what is loaded and which button to press.
        self._placeholder = QtWidgets.QLabel("")
        self._placeholder.setObjectName("muted")
        self._placeholder.setTextFormat(Qt.TextFormat.PlainText)
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setAccessibleName("Guidance")
        column.addWidget(self._placeholder)

        plot_split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        plot_split.setObjectName("splitter_plots")
        plot_split.setChildrenCollapsible(False)
        plot_split.addWidget(self._spectrum)
        # The phased and raw views sit side by side when there is room for both and
        # take turns under a tab bar when there is not (see _fit_lc_area): side by
        # side in a narrow window the raw view got an unreadable 89 px sliver, and
        # one above the other each plot was 45 px tall in a 1366 x 768 window.
        lc_area = QtWidgets.QWidget()
        lc_area.setObjectName("lc_area")
        lc_column = QtWidgets.QVBoxLayout(lc_area)
        lc_column.setContentsMargins(0, 0, 0, 0)
        lc_column.setSpacing(0)  # the tabs sit on the view they show
        self._lc_column = lc_column
        self._lc_tabs = self._build_lc_tabs()
        lc_column.addWidget(self._lc_tabs)
        lc_split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        lc_split.setObjectName("splitter_lc")
        lc_split.setChildrenCollapsible(False)
        lc_split.addWidget(self._phased)
        lc_split.addWidget(self._raw)
        self._raw.setVisible(False)
        self._lc_split = lc_split
        lc_column.addWidget(lc_split, 1)
        self._lc_area = lc_area
        # The pane's width decides side by side or tabs, and it changes without the
        # window resizing too (a dock dragged wider, the Controls splitter).
        lc_area.installEventFilter(self)
        plot_split.addWidget(lc_area)
        plot_split.setStretchFactor(0, 3)
        plot_split.setStretchFactor(1, 2)
        plot_split.setSizes([520, 340])
        # Only a drag of the handle (setSizes never emits it).
        plot_split.splitterMoved.connect(self._on_plot_split_moved)
        self._plot_split = plot_split
        column.addWidget(plot_split, 1)
        return page

    def _build_lc_tabs(self) -> QtWidgets.QTabBar:
        """The Phased / Raw light curve tabs of a pane too narrow for both at once."""
        tabs = QtWidgets.QTabBar()
        tabs.setObjectName("lc_tabs")
        tabs.addTab("Phased light curve")
        tabs.addTab("Raw light curve")
        tabs.setTabToolTip(_PHASED_TAB, "The light curve folded on the selected period")
        tabs.setTabToolTip(_RAW_TAB, "The unfolded light curve (Ctrl+R hides it)")
        tabs.setExpanding(False)
        tabs.setDrawBase(False)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        tabs.setAccessibleName("Light-curve views")
        tabs.setToolTip(_LC_TABS_TIP)
        tabs.setStatusTip(_LC_TABS_TIP)
        tabs.setWhatsThis(_LC_TABS_TIP)
        tabs.setVisible(False)
        tabs.currentChanged.connect(self._on_lc_tab_changed)
        return tabs

    def _build_result_docks(self) -> None:
        """The result docks, tabbed together on the right.

        Only the docks belonging to the active analysis are shown, but all of them
        live in the same tab group, so switching analysis never rearranges the window.
        The Frequencies dock carries the session bar above the table.
        """
        self._peaks_dock = self._make_dock("Peaks", "peaks_dock", self._peaks_table)
        frequencies = QtWidgets.QWidget()
        frequencies.setObjectName("frequencies_pane")
        column = QtWidgets.QVBoxLayout(frequencies)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._session_bar)
        column.addWidget(self._solution_panel, 1)
        self._session_bar.setVisible(False)
        self._solution_dock = self._make_dock(
            "Frequencies", "solution_dock", frequencies
        )
        self._subsets_dock = self._make_dock(
            "Subsets", "subsets_dock", self._subsets_panel
        )
        self._variations_dock = self._make_dock(
            "Variations", "variations_dock", self._variation_panel
        )
        self._history_dock = self._make_dock(
            "History", "history_dock", self._history_panel
        )
        self._spacing_dock = self._make_dock(
            "Period spacing", "spacing_dock", self._spacing_panel
        )
        self._tab_result_docks()

    def _tab_result_docks(self) -> None:
        docks = self._result_docks()
        for dock in docks:
            dock.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        for first, second in zip(docks, docks[1:], strict=False):
            self.tabifyDockWidget(first, second)
        QtCore.QTimer.singleShot(0, self, self._style_dock_tabs)

    def _style_dock_tabs(self) -> None:
        """Dock tabs keep their whole titles and scroll when they do not fit, and
        only a tab bar the user can see takes keyboard focus.

        Five tabs in a 400 px dock were elided to "Hi…" and "Sub…". QMainWindow also
        parks the tab bars it is not using off-screen while leaving them "visible",
        so Tab landed on an invisible "Period spacing" tab and its scroll arrows.
        The arrows themselves never take focus: Left/Right on the bar scroll it.
        Runs again whenever a tab bar is added, shown, hidden or moved.
        """
        self._tabs_pending = False
        area = self.rect()
        for bar in self.findChildren(
            QtWidgets.QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            if not bar.property("cup_styled"):
                bar.setProperty("cup_styled", True)
                bar.setElideMode(Qt.TextElideMode.ElideNone)
                bar.setUsesScrollButtons(True)
                bar.setExpanding(False)
                bar.setToolTip(_DOCK_TABS_TIP)
                bar.setStatusTip(_DOCK_TABS_TIP)
                bar.installEventFilter(self)
            live = bar.isVisible() and area.intersects(bar.geometry())
            bar.setFocusPolicy(
                Qt.FocusPolicy.TabFocus if live else Qt.FocusPolicy.NoFocus
            )
        # Every tab bar's scroll arrows (the panes' and the Variations results'):
        # mouse aids, two extra Tab stops that do nothing a keyboard user needs.
        for bar in self.findChildren(QtWidgets.QTabBar):
            for arrow in bar.findChildren(QtWidgets.QToolButton):
                arrow.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _queue_tab_tidy(self) -> None:
        if not getattr(self, "_tabs_pending", True):
            self._tabs_pending = True
            QtCore.QTimer.singleShot(0, self, self._style_dock_tabs)

    def childEvent(self, event: QtCore.QChildEvent) -> None:  # noqa: N802
        super().childEvent(event)
        # A QTabBar is still being constructed when it is added: look at it later.
        if event.added():
            self._queue_tab_tidy()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if isinstance(watched, QtWidgets.QTabBar) and event.type() in (
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.Move,
        ):
            self._queue_tab_tidy()
        elif (
            watched is getattr(self, "_lc_area", None)
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._queue_lc_fit()
        return super().eventFilter(watched, event)

    def _result_docks(self) -> list[QtWidgets.QDockWidget]:
        return [
            self._peaks_dock,
            self._solution_dock,
            self._subsets_dock,
            self._variations_dock,
            self._history_dock,
            self._spacing_dock,
        ]

    def _make_dock(
        self, title: str, object_name: str, widget: QtWidgets.QWidget
    ) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)  # required for saveState()
        dock.setWidget(_scrolling(widget, f"{title} pane"))
        feature = QtWidgets.QDockWidget.DockWidgetFeature
        dock.setFeatures(
            feature.DockWidgetMovable
            | feature.DockWidgetFloatable
            | feature.DockWidgetClosable
        )
        bar = _DockTitleBar(dock)
        dock.setTitleBarWidget(bar)
        self._title_bars.append(bar)
        toggle = dock.toggleViewAction()
        toggle.setText(f"{title} pane")
        toggle.setStatusTip(f"Show or hide the {title} pane")
        # A tab behind another one is parked off-screen but stays "visible", so its
        # controls stayed in the Tab chain (A11Y-15): disable the content while its
        # tab is not the current one, and enable it again when it is.
        dock.visibilityChanged.connect(partial(self._on_dock_visibility, dock))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    @staticmethod
    def _on_dock_visibility(dock: QtWidgets.QDockWidget, visible: bool) -> None:
        content = dock.widget()
        if content is not None:
            content.setEnabled(visible)

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
        bar = _DockTitleBar(self._sources_dock)
        self._sources_dock.setTitleBarWidget(bar)
        self._title_bars.append(bar)
        self._sources_dock.toggleViewAction().setText("Sources pane")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._sources_dock)
        self._sources_dock.visibilityChanged.connect(self._queue_fit_minimum)
        self._sources_dock.setVisible(False)
        self._sources_sized = False

    def _build_shortcuts(self) -> None:
        self._quit_action = QtGui.QAction("Quit", self)
        self._quit_action.setObjectName("action_file_quit")
        # Not StandardKey.Quit: it is empty on Windows and platform-specific
        # elsewhere; Ctrl+Q is Cmd+Q on macOS.
        self._quit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        self._quit_action.setToolTip("Quit cuPeriod  (Ctrl+Q)")
        self._quit_action.setStatusTip("Close the window (asks about unsaved work)")
        self._quit_action.triggered.connect(self.close)
        self.addAction(self._quit_action)

        # Zoom text and plot fonts together (WCAG 1.4.4); listed under View.
        self._zoom_actions: list[QtGui.QAction] = []
        for text, steps, tip in (
            ("Zoom in", 1, "Make text and plot labels larger"),
            ("Zoom out", -1, "Make text and plot labels smaller"),
            ("Reset zoom", 0, "Back to the designed text size"),
        ):
            action = QtGui.QAction(text, self)
            action.setObjectName("action_view_" + text.lower().replace(" ", "_"))
            action.setStatusTip(tip)
            action.triggered.connect(partial(self._zoom, steps))
            self.addAction(action)
            self._zoom_actions.append(action)
        zoom_in, zoom_out, reset = self._zoom_actions
        zoom_in.setShortcuts(QtGui.QKeySequence.StandardKey.ZoomIn)
        zoom_in.setShortcuts([*zoom_in.shortcuts(), QtGui.QKeySequence("Ctrl+=")])
        zoom_out.setShortcuts(QtGui.QKeySequence.StandardKey.ZoomOut)
        reset.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        for action in self._zoom_actions:
            keys = ", ".join(
                s.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
                for s in action.shortcuts()
            )
            action.setToolTip(f"{action.text()}  ({keys})")

        # Shift+F1, then click a control: its What's This? help (every option has one).
        self._whats_this_action = QtWidgets.QWhatsThis.createAction(self)
        self._whats_this_action.setStatusTip(
            "Click a control afterwards to read what it does"
        )
        self.addAction(self._whats_this_action)

        self._raw_action = QtGui.QAction("Show raw light curve", self)
        self._raw_action.setObjectName("action_view_raw_light_curve")
        self._raw_action.setCheckable(True)
        self._raw_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self._raw_action.setToolTip(
            "Show the raw (unfolded) light curve next to the fold  (Ctrl+R)"
        )
        self._raw_action.setStatusTip("Show or hide the raw light-curve panel")
        self._raw_action.toggled.connect(self._on_raw_action)
        self.addAction(self._raw_action)

        # The periodogram's Compute shares Ctrl+Return / F5 with Auto: only one of
        # the two is enabled at a time (see _sync_commands), so the keys never clash.
        self._compute_action = QtGui.QAction("Compute periodogram", self)
        self._compute_action.setObjectName("action_view_compute")
        self._compute_action.setShortcuts(
            [QtGui.QKeySequence(k) for k in ("F5", "Ctrl+Return", "Ctrl+Enter")]
        )
        self._compute_action.setToolTip("Compute the periodogram  (F5, Ctrl+Return)")
        self._compute_action.setStatusTip("Compute the periodogram of the curve")
        self._compute_action.triggered.connect(self._compute_via_shortcut)
        self.addAction(self._compute_action)

        self._errors_action = QtGui.QAction("Compute uncertainties", self)
        self._errors_action.setObjectName("action_session_compute_errors")
        self._errors_action.setStatusTip(
            "Full result with the configured error estimator (a bootstrap can take "
            "a while; Stop ends it)"
        )
        self._errors_action.triggered.connect(self._controller.compute_errors)

        self._reset_layout_action = QtGui.QAction("Reset layout", self)
        self._reset_layout_action.setObjectName("action_view_reset_layout")
        self._reset_layout_action.setStatusTip(
            "Put every pane back where it starts out"
        )
        self._reset_layout_action.triggered.connect(self.reset_layout)

        self._analysis_group = QtGui.QActionGroup(self)
        self._analysis_group.setExclusive(True)
        self._analysis_actions: dict[str, QtGui.QAction] = {}
        for analysis, text, tip in (
            ("periodogram", "Periodogram", "One period-search statistic"),
            ("prewhiten", "Pre-whitening", "Iterative sinusoid extraction"),
        ):
            action = QtGui.QAction(text, self)
            action.setObjectName(f"action_view_analysis_{analysis}")
            action.setCheckable(True)
            action.setStatusTip(tip)
            action.triggered.connect(
                lambda _=False, a=analysis: self._controls.set_analysis(a)
            )
            self._analysis_group.addAction(action)
            self._analysis_actions[analysis] = action

        self._docs_action = QtGui.QAction("Documentation", self)
        self._docs_action.setObjectName("action_help_documentation")
        self._docs_action.setStatusTip(f"Open the cuPeriod documentation ({DOCS_URL})")
        self._docs_action.triggered.connect(self._open_documentation)
        self._about_action = QtGui.QAction("About cuPeriod", self)
        self._about_action.setObjectName("action_help_about")
        self._about_action.setStatusTip("Version, licence and where to report issues")
        self._about_action.triggered.connect(self._show_about)

    def _build_menus(self) -> None:
        """The menu bar: the registry's commands plus the window's own."""
        bar = self.menuBar()
        assert bar is not None
        # Titles with mnemonics first; the registry appends to menus of the same title.
        for title in ("&File", "&Edit", "&Pre-whitening", "&Data", "&View", "&Help"):
            menu = bar.addMenu(title)
            assert menu is not None
            menu.setObjectName("menu_" + title.strip("&").lower().replace("-", "_"))
        menus = self._commands.populate_menus(bar)
        self._menus = menus
        action = self._commands.action

        file_menu = menus["File"]
        file_menu.insertAction(action("file.open_project"), self._folder_action)
        self._recent_menu = QtWidgets.QMenu("Open recent", self)
        self._recent_menu.setObjectName("menu_recent")
        self._recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.insertMenu(action("file.open_project"), self._recent_menu)
        file_menu.insertMenu(action("file.open_project"), self._demo_menu)
        file_menu.insertSeparator(action("file.open_project"))
        file_menu.addSeparator()
        self._sessions_menu = QtWidgets.QMenu("Unsaved sessions", self)
        self._sessions_menu.setObjectName("menu_sessions")
        self._sessions_menu.setStatusTip(
            "Sessions with unsaved work, kept while you look at other curves"
        )
        self._sessions_menu.aboutToShow.connect(self._populate_sessions_menu)
        file_menu.addMenu(self._sessions_menu)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        prewhiten_menu = menus["Pre-whitening"]
        prewhiten_menu.insertAction(action("session.finalize"), self._errors_action)

        view = menus["View"]
        view.addSeparator()
        analysis_menu = view.addMenu("Analysis")
        assert analysis_menu is not None
        for item in self._analysis_actions.values():
            analysis_menu.addAction(item)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self._compute_action)
        assign_mnemonics(analysis_menu)
        view.addSeparator()
        panes = view.addMenu("Panes")
        assert panes is not None
        panes.setObjectName("menu_panes")
        for dock in self._result_docks():
            panes.addAction(dock.toggleViewAction())
        panes.addAction(self._sources_dock.toggleViewAction())
        assign_mnemonics(panes)
        view.addAction(self._raw_action)
        view.addSeparator()
        view.addAction(self._theme_action)
        for item in self._zoom_actions:
            view.addAction(item)
        view.addSeparator()
        view.addAction(self._reset_layout_action)

        help_menu = menus["Help"]
        help_menu.addAction(self._whats_this_action)
        help_menu.addSeparator()
        help_menu.addAction(self._docs_action)
        help_menu.addAction(self._about_action)

        for menu in menus.values():
            assign_mnemonics(menu)

    def _attach_commands(self) -> None:
        """Pane letters and the bars that show registry actions."""
        registry = self._commands
        registry.attach_context("spectrum", self._spectrum)
        registry.attach_context("lightcurve", self._raw)
        self._solution_panel.attach_commands(registry)
        self._session_bar.bind_commands(registry)
        self._history_panel.bind_commands(registry)
        registry.triggered.connect(self._on_command)
        # The window keeps the enabled states in step with the controller (4.9).
        self._raw.tool_changed.connect(self._on_tool_changed)

    def _restore_controls(self) -> None:
        saved = QtCore.QSettings().value(_KEY_CONTROLS)
        if isinstance(saved, str) and saved:
            self._controls.restore_state(saved)
        raw = QtCore.QSettings().value(_KEY_RAW)
        if isinstance(raw, str) and raw:
            try:
                prefs = json.loads(raw)
            except ValueError:
                prefs = None
            if isinstance(prefs, dict):
                for key in self._raw_pref:
                    if isinstance(prefs.get(key), bool):
                        self._raw_pref[key] = prefs[key]

    def _restore_window_state(self) -> None:
        """Restore the previous session's window geometry/layout, if any was saved.

        Only byte arrays are handed to Qt: a hand-edited or corrupted settings store
        can hold a string there, and ``restoreGeometry(str)`` raises inside the
        constructor, so the window could not start until the setting was deleted. A
        saved rectangle that no longer meets any screen falls back to the default.
        """
        self._apply_screen_size()
        settings = QtCore.QSettings()
        restored = False
        saved_geometry = settings.value("geometry")
        if isinstance(saved_geometry, QtCore.QByteArray | bytes | bytearray):
            restored = self.restoreGeometry(QtCore.QByteArray(saved_geometry))
            if restored and not self._on_some_screen():
                self._apply_screen_size(center=True)
        saved_state = settings.value("windowState")
        state_ok = False
        if isinstance(saved_state, QtCore.QByteArray | bytes | bytearray):
            state_ok = self.restoreState(QtCore.QByteArray(saved_state))
        splitters = settings.value(_KEY_SPLITTERS)
        self._splitters_restored = False
        if isinstance(splitters, str) and splitters:
            self._splitters_restored = self._restore_splitters(splitters)
        # Batch mode never survives a restart, so a Sources dock saved as visible
        # would come back empty ("No sources") and take width from the plots. Hiding
        # it after the restore also repairs layouts saved by earlier versions; the
        # toolbar likewise always comes back.
        self._sources_dock.setVisible(False)
        self._toolbar.setVisible(True)
        # Without a saved layout the docks get their default share on the first
        # show, once the window has its real size (resizeDocks needs the geometry).
        self._needs_dock_widths = not state_ok

    def _apply_screen_size(self, *, center: bool = False) -> None:
        """Size the window from the screen: the design size or most of the screen.

        The minimum keeps the panes usable but never exceeds the screen either, so
        the window always fits a small laptop or a high-DPI display at 150 %.
        """
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover - a headless platform has one
            self.resize(*_DESIGN_SIZE)
            return
        avail = screen.availableGeometry()
        min_w, min_h = window_minimum(self.minimumSizeHint(), avail.size())
        self.setMinimumSize(min_w, min_h)
        width = max(min_w, min(_DESIGN_SIZE[0], int(avail.width() * 0.92)))
        height = max(min_h, min(_DESIGN_SIZE[1], int(avail.height() * 0.9)))
        self.resize(width, height)
        if center:
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())

    def _fit_minimum(self) -> None:
        """Keep the window minimum at what the panes need now (docks, zoom).

        The Controls column grows with the zoom (its buttons were clipped to
        "n pre-whiteni" at 125 %), and the window may not get narrower than the
        Controls, the plots and the docks together: at a 900 px minimum against
        976 px of panes, the plots were drawn over the Controls.
        """
        self._controls.setMinimumWidth(
            max(_CONTROLS_MIN_WIDTH, self._controls.minimumSizeHint().width())
        )
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover - a headless platform has one
            return
        self.setMinimumSize(
            *window_minimum(self.minimumSizeHint(), screen.availableGeometry().size())
        )

    def _queue_fit_minimum(self) -> None:
        QtCore.QTimer.singleShot(0, self, self._fit_minimum)

    def _on_some_screen(self) -> bool:
        frame = self.frameGeometry()
        return any(
            s.availableGeometry().intersects(frame)
            for s in QtGui.QGuiApplication.screens()
        )

    def _default_dock_widths(self) -> None:
        """Give the right-hand docks about a third of the window (not their sum of
        minimums, which left the plots crushed)."""
        width = int(min(600, max(300, self.width() * 0.3)))
        self.resizeDocks(
            self._result_docks(), [width] * len(self._result_docks()),
            Qt.Orientation.Horizontal,
        )

    def _splitters(self) -> dict[str, QtWidgets.QSplitter]:
        return {
            "main": self._main_split,
            "plots": self._plot_split,
            "lc": self._lc_split,
        }

    def _restore_splitters(self, text: str) -> bool:
        try:
            saved = json.loads(text)
        except ValueError:
            return False
        if not isinstance(saved, dict):
            return False
        restored = False
        for name, splitter in self._splitters().items():
            value = saved.get(name)
            if isinstance(value, str):
                data = QtCore.QByteArray.fromBase64(value.encode("ascii"))
                restored = splitter.restoreState(data) or restored
        # The fold and the raw curve sit side by side or take turns as tabs, never
        # one above the other (older versions saved a vertical split).
        self._lc_split.setOrientation(Qt.Orientation.Horizontal)
        return restored

    def reset_layout(self) -> None:
        """View ▸ Reset layout: docks tabbed on the right, default splitters."""
        for dock in self._result_docks():
            dock.setFloating(False)
        self._tab_result_docks()
        self._sources_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._sources_dock)
        self._sources_dock.setVisible(self._controller.state.mode == "batch")
        self._toolbar.setVisible(True)
        self._apply_analysis(self._controls.analysis)
        self._main_split.setSizes([340, max(400, self.width() - 340)])
        self._plot_split.setSizes([520, 340])
        self._split_before_raw = None
        self._default_dock_widths()
        self._fit_lc_area(force=True)
        self._show_status("Layout reset.")

    # =================================================================================
    # wiring
    # =================================================================================
    def _connect(self) -> None:
        controls = self._controls
        controls.run_requested.connect(self._on_run_requested)
        controls.prewhiten_requested.connect(self._on_prewhiten_requested)
        controls.step_requested.connect(self._on_step_requested)
        controls.stop_requested.connect(self._stop)
        controls.analysis_changed.connect(self._on_analysis_changed)
        controls.band_changed.connect(self._on_band_changed)
        controls.settings_edited.connect(self._on_settings_edited)
        controls.compute_attempted.connect(self._hide_banner)
        ctl = self._controller
        ctl.solution_ready.connect(self._on_solution_ready)
        self._solution_panel.component_selected.connect(ctl.select_component)
        ctl.lc_loaded.connect(self._on_lc_loaded)
        ctl.compute_started.connect(self._on_compute_started)
        ctl.busy_changed.connect(self._on_busy_changed)
        ctl.periodogram_ready.connect(self._on_periodogram_ready)
        ctl.periodogram_ready.connect(self._spectrum.set_periodogram)
        ctl.peaks_ready.connect(self._spectrum.set_peaks)
        ctl.period_changed.connect(self._spectrum.set_selected_period)
        self._spectrum.period_selected.connect(ctl.select_period)
        ctl.lc_loaded.connect(self._phased.set_light_curve)
        ctl.lc_loaded.connect(self._raw.set_light_curve)
        self._phased.show_raw_toggled.connect(self._on_show_raw_toggled)
        self._phased.period_requested.connect(ctl.select_period)
        ctl.fold_changed.connect(self._phased.set_fold)
        ctl.peaks_ready.connect(self._peaks_table.set_peaks)
        self._peaks_table.peak_selected.connect(ctl.select_peak)
        # A click on a spectrum marker selects that whole peak (a BLS/TLS t0
        # included); the tables follow the selection without re-emitting it.
        self._spectrum.peak_selected.connect(ctl.select_peak)
        ctl.period_changed.connect(self._peaks_table.select_period)
        ctl.period_changed.connect(self._solution_panel.select_period)
        # A failed export is reported in the banner, like every other error.
        for view in (
            self._spectrum,
            self._phased,
            self._peaks_table,
            self._solution_panel,
        ):
            view.export_failed.connect(self._show_banner)
        self._source_browser.source_selected.connect(self._select_source)
        ctl.sources_changed.connect(self._on_sources_changed)
        # Queued: when a click or an arrow key lands on a source that fails to load,
        # the controller snaps the selection back while Qt is still handling that
        # click, and Qt's own selection update would then highlight the failed row
        # over the restored one.
        ctl.source_selected.connect(
            self._source_browser.set_current, Qt.ConnectionType.QueuedConnection
        )
        ctl.source_failed.connect(self._on_source_failed)
        ctl.lc_cleared.connect(self._on_lc_cleared)
        ctl.compute_failed.connect(self._on_compute_failed)
        ctl.selection_cleared.connect(self._spectrum.clear_selection)
        ctl.selection_cleared.connect(self._phased.clear_fold)
        ctl.analysis_changed.connect(self._on_controller_analysis)
        ctl.status_message.connect(self._show_status)
        self._connect_session()

    def _connect_session(self) -> None:
        """The interactive pre-whitening signals (design 4.2-4.8)."""
        ctl = self._controller
        ctl.session_changed.connect(self._on_session_changed)
        ctl.session_progress.connect(self._on_session_progress)
        ctl.session_message.connect(self._on_session_message)
        ctl.session_refused.connect(self._on_session_refused)
        ctl.session_busy.connect(self._info.set_working)
        ctl.variations_ready.connect(self._variation_panel.set_result)
        ctl.components_selected.connect(self._on_components_selected)
        ctl.cursor_changed.connect(self._on_cursor_changed)
        ctl.project_changed.connect(self._on_project_changed)
        ctl.preview_failed.connect(self._on_preview_failed)
        ctl.settings_restored.connect(self._on_settings_restored)
        ctl.close_approved.connect(self.close)

        spectrum = self._spectrum
        spectrum.add_requested.connect(ctl.add_frequency)
        spectrum.cursor_changed.connect(ctl.set_cursor)
        spectrum.component_clicked.connect(lambda i: ctl.select_components([i]))

        self._raw.delete_requested.connect(ctl.delete_points)
        self._raw.restore_requested.connect(ctl.restore_points)
        self._raw.selection_changed.connect(self._on_points_selected)

        panel = self._solution_panel
        panel.components_selected.connect(ctl.select_components)
        panel.edit_requested.connect(self._dispatch_edit)
        panel.group_requested.connect(self._dispatch_group)
        panel.status_message.connect(self._show_status)

        bar = self._session_bar
        bar.edit_requested.connect(self._dispatch_edit)
        bar.group_requested.connect(self._dispatch_group)
        bar.status_message.connect(self._show_status)
        bar.command_requested.connect(self._commands.triggered)

        history = self._history_panel
        history.goto_requested.connect(ctl.session_goto)
        history.copy_script_requested.connect(ctl.copy_entry_script)
        history.export_script_requested.connect(self._export_script)

        subsets = self._subsets_panel
        subsets.partition_requested.connect(ctl.define_partition)
        subsets.edit_requested.connect(self._dispatch_edit)
        subsets.group_requested.connect(self._dispatch_group)
        subsets.points_highlighted.connect(self._on_points_highlighted)
        subsets.message.connect(self._show_status)
        self._raw.time_cursor_changed.connect(self._on_time_cursor)

        variations = self._variation_panel
        variations.query_requested.connect(
            lambda q: ctl.query_variations(
                q.ids, q.mode, q.by, q.improve_frequencies
            )
        )
        variations.edit_requested.connect(self._dispatch_edit)
        variations.group_requested.connect(self._dispatch_group)
        variations.message.connect(self._show_status)

    # -- edits from the panels -----------------------------------------------------
    def _dispatch_edit(self, op: str, args: object) -> None:
        """One op from a panel → the controller (``expect_id_epoch`` split off).

        *Improve* goes through ``session_improve`` so a changed Options form is
        applied first (D17), as Improve all does.
        """
        mapping = dict(args) if isinstance(args, Mapping) else {}
        clean, epoch = split_epoch(mapping)
        if op == "improve":
            request = self._controls.prewhiten_request(report=False)
            if request is not None:
                ids = clean.get("ids")
                self._controller.session_improve(
                    request[0],
                    request[1],
                    self._controls.current_band(),
                    ids=None if ids is None else [int(i) for i in ids],
                    expect_id_epoch=epoch,
                )
                return
        self._controller.session_do(op, expect_id_epoch=epoch, **clean)

    def _dispatch_group(self, payload: object) -> None:
        """Several ops that undo as one: a tuple of ``(op, args)`` or a mapping with
        ``actions`` and ``expect_id_epoch`` (the Subsets and Variations docks)."""
        epoch: int | None = None
        actions: Any = payload
        if isinstance(payload, Mapping):
            actions = payload.get("actions", ())
            raw = payload.get("expect_id_epoch")
            epoch = None if raw is None else int(raw)
        if not isinstance(actions, Sequence):
            return
        clean, inner = split_group(
            (str(op), args if isinstance(args, Mapping) else {}) for op, args in actions
        )
        if clean:
            self._controller.session_do_group(
                clean, expect_id_epoch=epoch if epoch is not None else inner
            )

    # -- registry --------------------------------------------------------------------
    def _on_command(self, command_id: str) -> None:
        """Carry out a registry command (menu, button, global key or pane letter)."""
        if command_id in TABLE_COMMANDS and command_id != "edit.delete":
            return  # the frequency table carries these out itself
        handlers: dict[str, Callable[[], object]] = {
            "edit.delete": self._delete,
            "session.auto": self._compute_via_shortcut,
            "session.step": self._controls.request_step,
            "session.improve_all": self._improve_all,
            "session.add_at_cursor": self._add_at_cursor,
            "session.add_frequency": lambda: None,  # the session bar focuses its field
            "session.stop": self._stop,
            "data.restore_selected": self._raw.restore_selected,
            "data.clear_selection": self._raw.clear_selection,
            "data.tool_pan": lambda: self._use_tool("pan"),
            "data.tool_rect": lambda: self._use_tool("rect"),
            "data.tool_lasso": lambda: self._use_tool("lasso"),
            "file.open_curve": lambda: self._controller.confirm_then(self._open_file),
            "file.open_project": self._open_project,
            "file.export_script": self._export_script,
            "file.export_frequencies": self._export_frequencies,
            "file.verify_project": self._verify_project,
            "file.new_session": self._new_session,
            "view.focus_spectrum": self._spectrum.focus_plot,
            "view.focus_frequencies": self._focus_frequencies,
            "view.focus_light_curve": self._focus_light_curve,
            "view.focus_subsets": lambda: self._focus_dock(self._subsets_dock),
            "view.focus_history": self._focus_history,
            "view.shortcuts": self._show_shortcuts,
        }
        handler = handlers.get(command_id)
        if handler is not None:
            handler()
            return
        self._controller.run_command(command_id)

    def _sync_commands(self) -> None:
        """The registry's enabled state from the controller (design 4.9).

        A "session" here is pre-whitening with a curve on show (the session itself
        opens lazily on the first action). In periodogram mode the same keys (F5,
        Ctrl+Return) run the window's Compute instead of Auto; Undo and Redo follow
        the displayed history.
        """
        state = self._controller.state
        has_curve = state.current_lc is not None
        session = self._controls.analysis == "prewhiten" and has_curve
        # "Preview failed" (4.9): the session could not open, so its commands are
        # off. Auto and Step stay, like the Controls buttons they mirror: they open
        # the session afresh with the form's backend and say in the banner why not.
        failed = session and bool(state.preview_error)
        usable = session and not failed
        registry = self._commands
        registry.set_state(session=usable, busy=self._busy)
        if failed:
            for command_id in ("session.auto", "session.step"):
                registry.action(command_id).setEnabled(not self._busy)
        view = self._session_view()
        self._compute_action.setEnabled(
            has_curve and not self._busy and not prewhiten_mode(self._controls)
        )
        registry.action("edit.undo").setEnabled(
            usable and not self._busy and view is not None and view.can_undo
        )
        registry.action("edit.redo").setEnabled(
            usable and not self._busy and view is not None and view.can_redo
        )
        registry.action("file.new_session").setEnabled(usable and not self._busy)
        self._errors_action.setEnabled(usable and not self._busy)

    def _session_view(self) -> SessionView | None:
        view = self._controller.state.session_view
        return view if isinstance(view, SessionView) else None

    def _on_tool_changed(self, tool: str) -> None:
        action_id = {
            "pan": "data.tool_pan",
            "rect": "data.tool_rect",
            "lasso": "data.tool_lasso",
        }.get(tool)
        if action_id is not None:
            self._commands.action(action_id).setChecked(True)

    # =================================================================================
    # actions
    # =================================================================================
    def _open_file(self) -> None:
        start_dir = str(QtCore.QSettings().value("last_dir", ""))
        self._dialogs.open_file(
            "Open light curve", _FILE_FILTER, self._on_file_chosen, directory=start_dir
        )

    def _on_file_chosen(self, path_str: str | None) -> None:
        if not path_str:
            return
        path = Path(path_str)
        QtCore.QSettings().setValue("last_dir", str(path.parent))
        self._load_file(path)

    def _load_file(self, path: Path) -> None:
        """Open a single light-curve file at ``path`` (through the import dialog).

        The import dialog shows how the file reads and lets the user choose the
        time, brightness, error and band columns, the brightness scale and (for a
        text table) the delimiter and header row; the previous choice is offered
        again. Files that cannot be previewed load as they are.
        """
        preview = preview_file(path)
        if preview is None:
            self._load_with(path, None)
            return
        self._dialogs.import_file(
            str(path),
            preview,
            lambda choice: self._on_import_choice(path, choice),
            last=self._last_import(),
        )

    def _on_import_choice(self, path: Path, choice: ImportChoice | None) -> None:
        if choice is None:
            self._show_status(f"Opening {path.name} was cancelled.")
            return
        QtCore.QSettings().setValue(_KEY_IMPORT, import_choice_to_json(choice))
        self._load_with(path, choice)

    @staticmethod
    def _last_import() -> ImportChoice | None:
        return import_choice_from_json(QtCore.QSettings().value(_KEY_IMPORT))

    def _load_with(self, path: Path, choice: ImportChoice | None) -> None:
        try:
            lc = choice.load(path) if choice is not None else load_path(path)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._warn(f"Could not load {path.name}:\n{type(exc).__name__}: {exc}")
            return
        if not self._check_usable(lc, path.name):
            return
        self._exit_batch_mode()
        self._data_name = path.name
        self._controller.set_light_curve(lc, source_id=str(path))
        self._add_recent(path)
        self._update_title()

    def _load_demo(self, item: SourceItem) -> None:
        try:
            lc = item.loader()
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._warn(f"Could not load demo:\n{type(exc).__name__}: {exc}")
            return
        if not self._check_usable(lc, item.label):
            return
        # A demo can name the method it is meant for (the Kepler transit is a BLS
        # demo: GLS reports half its period). Apply it in periodogram mode, where the
        # method picker is the user's to change back.
        hint = item.meta.get("hint_method")
        applied = (
            isinstance(hint, str)
            and self._controls.analysis == "periodogram"
            and self._controls.current_method() != hint
            and self._controls.set_method(hint)
        )
        self._exit_batch_mode()
        self._data_name = item.label
        self._controller.set_light_curve(lc, source_id=item.key, label=item.label)
        self._update_title()
        if applied and isinstance(hint, str):
            what = method_description(hint).split(" — ")[0] or hint
            self._show_status(
                f"Method set to {hint} ({what}) for this demo — press "
                f"{self._controls.action_label}."
            )

    def _check_usable(self, lc: LoadedCurve, name: str) -> bool:
        """Refuse a curve with nothing to analyse, saying so; True if it is usable.

        A header-only file loads as a zero-point curve; accepting it enabled Compute
        and announced "Light curve loaded" over an empty plot.
        """
        usable = _usable_points(lc)
        if usable >= _MIN_USABLE_POINTS:
            return True
        rows = _total_rows(lc)
        if rows == 0:
            self._warn(f"{name} contains no data rows.")
        else:
            self._warn(
                f"{name} has no usable data: {usable} of its {rows} rows have a "
                f"finite time and brightness, and at least {_MIN_USABLE_POINTS} are "
                "needed."
            )
        return False

    def _open_folder(self) -> None:
        def pick() -> None:
            start_dir = str(QtCore.QSettings().value("last_dir", ""))
            self._dialogs.open_folder(
                "Open a folder of light curves", self._on_folder_chosen,
                directory=start_dir,
            )

        self._controller.confirm_then(pick)

    def _on_folder_chosen(self, directory: str | None) -> None:
        if not directory:
            return
        QtCore.QSettings().setValue("last_dir", directory)
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
        name = Path(directory).name or str(directory)
        self._data_name = f"{name} ({len(items)} sources)"
        self._controller.load_sources(items)
        self._update_title()

    def _load_demo_batch(self) -> None:
        try:
            items = demo_sources()
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            self._warn(f"Could not list the demos:\n{type(exc).__name__}: {exc}")
            return
        self._data_name = "demo batch"
        self._controller.load_sources(items)
        self._update_title()

    def _on_sources_changed(self, labels: object) -> None:
        assert isinstance(labels, list)
        self._source_browser.set_sources(labels)
        self._sources_dock.setVisible(True)
        if not self._sources_sized:
            # Wide enough for a label's useful tail (type and period), not the
            # 150 px the dock opened at (UX-23).
            self._sources_sized = True
            self.resizeDocks([self._sources_dock], [260], Qt.Orientation.Horizontal)
        # Start on the first source that loads, so one unreadable file at the top of
        # a folder does not leave the previous curve on screen.
        for index in range(min(len(labels), 20)):
            if self._select_source(index):
                break

    def _select_source(self, index: int) -> bool:
        """Activate batch source ``index``; return whether it loaded.

        A failed load leaves the controller on the previous source; it reports the
        row through ``source_failed`` (marked in the browser) and the reason through
        ``compute_failed``.
        """
        loads = self._load_count
        self._controller.select_source(index)
        return self._load_count != loads

    def _on_source_failed(self, index: int) -> None:
        """Grey out a batch source that could not be loaded, so Next steps past it."""
        self._failed_source = index
        self._source_browser.mark_failed(index, "Could not be loaded.")
        # Re-sync once Qt has finished handling the click or arrow key that got here:
        # its own selection update would otherwise leave the failed row highlighted.
        QtCore.QTimer.singleShot(0, self, self._sync_source_highlight)

    def _sync_source_highlight(self) -> None:
        if self._controller.state.mode == "batch":
            self._source_browser.set_current(
                self._controller.state.current_source_index
            )

    def _choose_columns_for_source(self, index: int) -> None:
        """A batch file that would not load: open it in the import dialog and, with
        the columns chosen there, load it as that batch source (UX-03)."""
        sources = self._controller.state.sources
        if sources is None or not 0 <= index < len(sources):
            return
        item = sources[index]
        path = Path(item.key)
        preview = preview_file(path) if path.is_file() else None
        if preview is None:
            self._show_status(f"{item.label} cannot be previewed.")
            return

        def chosen(choice: ImportChoice | None) -> None:
            current = self._controller.state.sources
            if choice is None or current is None or index >= len(current):
                return
            QtCore.QSettings().setValue(_KEY_IMPORT, import_choice_to_json(choice))
            current[index] = dataclasses.replace(
                current[index], loader=partial(choice.load, path)
            )
            self._source_browser.clear_failed(index)
            self._hide_banner()
            self._select_source(index)

        self._dialogs.import_file(str(path), preview, chosen, last=self._last_import())

    def _on_lc_cleared(self) -> None:
        """The controller dropped the active curve (a new batch): show nothing of it.

        If the new batch's first sources fail to load, the previous curve and its
        results must not stay on screen under the new batch's name, nor be
        computable.
        """
        self._loaded_detail = None
        self._showing_result = False
        self._clear_results()
        self._controls.set_enabled(False)
        self._info.set_idle("no light curve loaded")
        self._stack.setCurrentIndex(0)
        self._sync_commands()

    def _exit_batch_mode(self) -> None:
        self._controller.exit_batch_mode()
        self._sources_dock.setVisible(False)

    def _curve_tag(self) -> tuple[str, str]:
        state = self._controller.state
        return state.source_id, state.data_id

    def _on_run_requested(
        self, method: str, backend: str, settings: BaseSettings, n_peaks: int
    ) -> None:
        band = self._controls.current_band()
        self._last_runs["periodogram"] = (
            self._curve_tag(),
            partial(self._controller.run, method, backend, settings, n_peaks, band),
        )
        self._controller.run(method, backend, settings, n_peaks, band=band)

    def _on_prewhiten_requested(self, backend: str, settings: object) -> None:
        assert isinstance(settings, PreWhitenSettings)
        band = self._controls.current_band()
        self._last_runs["prewhiten"] = (
            self._curve_tag(),
            partial(self._controller.run_prewhiten, backend, settings, band),
        )
        self._controller.run_prewhiten(backend, settings, band=band)

    def _on_step_requested(self, backend: str, settings: object) -> None:
        assert isinstance(settings, PreWhitenSettings)
        self._controller.session_step(
            backend, settings, band=self._controls.current_band()
        )

    def _use_tool(self, tool: str) -> None:
        """A point tool: show the light curve it works in, then pick the tool."""
        self._bring_raw_forward()
        self._raw.set_tool(tool)  # type: ignore[arg-type]

    def _improve_all(self) -> None:
        request = self._controls.prewhiten_request()
        if request is None:
            return
        self._controller.session_improve(
            request[0], request[1], self._controls.current_band()
        )

    def _add_at_cursor(self) -> None:
        if not self._spectrum.request_add_at_cursor():
            self._controller.run_command("session.add_at_cursor")

    def _delete(self) -> None:
        """Delete: the table's rows when it has the focus (it acts itself), else
        the light curve's selected points."""
        if self._solution_panel.table.has_focus_within():
            return
        if self._raw.selection_count() > 0:
            self._raw.delete_selected()
        else:
            self._show_status(
                "Select frequencies in the table, or points in the light curve, "
                "to delete."
            )

    def _stop(self) -> None:
        """Stop (Esc): the session's long job after its step, or the one-shot run."""
        self._controller.cancel_compute()
        self._poll_finishing()

    def _poll_finishing(self) -> None:
        """Say "finishing previous run…" while a stopped run still holds the worker."""
        finishing = self._controller.finishing
        self._info.set_finishing(finishing)
        if finishing:
            QtCore.QTimer.singleShot(500, self, self._poll_finishing)

    def _new_session(self) -> None:
        request = self._controls.prewhiten_request()
        if request is None:
            return
        self._controller.new_session(
            request[0], request[1], self._controls.current_band()
        )

    def _open_project(self) -> None:
        self._dialogs.open_file(
            "Open project",
            PROJECT_FILTER,
            self._on_project_chosen,
            directory=str(QtCore.QSettings().value("last_dir", "")),
        )

    def _on_project_chosen(self, path: str | None) -> None:
        if path:
            QtCore.QSettings().setValue("last_dir", str(Path(path).parent))
            self._controller.open_project(path)

    def _verify_project(self) -> None:
        def chosen(path: str | None) -> None:
            if path:
                self._show_status(f"Verifying {Path(path).name}…")
                self._controller.open_project(path, verify=True)

        self._dialogs.open_file("Verify project", PROJECT_FILTER, chosen)

    def _export_script(self) -> None:
        stem = Path(self._data_name or "session").stem or "session"

        def chosen(path: str | None) -> None:
            if path:
                self._controller.export_script(path)

        self._dialogs.save_file(
            "Export script",
            "Python scripts (*.py);;All files (*)",
            f"{stem}.py",
            chosen,
        )

    def _export_frequencies(self) -> None:
        stem = Path(self._data_name or "frequencies").stem or "frequencies"

        def chosen(path: str | None) -> None:
            if not path:
                return
            try:
                self._solution_panel.export_csv(path)
            except OSError as exc:
                self._show_banner(f"Could not write {Path(path).name}: {exc}")
                return
            self._show_status(f"Wrote the frequency table to {path}")

        self._dialogs.save_file(
            "Export frequencies",
            "CSV files (*.csv);;All files (*)",
            f"{stem}_frequencies.csv",
            chosen,
        )

    def _show_shortcuts(self) -> None:
        self._dialogs.show_html("Keyboard shortcuts", self._commands.sheet_html())

    def _open_documentation(self) -> None:
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl(DOCS_URL)):
            self._show_status(f"Could not open a browser; the documentation is at "
                              f"{DOCS_URL}")

    def _show_about(self) -> None:
        import html

        text = (
            f"<h3>cuPeriod</h3><p>{html.escape(version_label())}</p>"
            "<p>GPU/CPU periodograms, pre-whitening and period-change tools for "
            "astronomical light curves.</p>"
            f'<p>Documentation: <a href="{DOCS_URL}">{DOCS_URL}</a></p>'
            "<p>Press F1 for the keyboard shortcuts.</p>"
        )
        self._dialogs.show_html("About cuPeriod", text)

    # -- focus -------------------------------------------------------------------------
    def _focus_dock(self, dock: QtWidgets.QDockWidget) -> QtWidgets.QWidget | None:
        if not dock.isVisible():
            dock.setVisible(True)
        dock.raise_()
        area = dock.widget()
        if area is not None:
            area.setEnabled(True)
        content = _dock_content(dock)
        if content is not None:
            content.setFocus(Qt.FocusReason.ShortcutFocusReason)
        return content

    def _focus_frequencies(self) -> None:
        if self._controls.analysis == "prewhiten":
            self._focus_dock(self._solution_dock)
            self._solution_panel.table.setFocus(Qt.FocusReason.ShortcutFocusReason)
        else:
            self._focus_dock(self._peaks_dock)

    def _focus_history(self) -> None:
        self._focus_dock(self._history_dock)
        self._history_panel.tree.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _focus_light_curve(self) -> None:
        self._bring_raw_forward()
        self._raw.focus_plot()

    # -- recent files ----------------------------------------------------------------
    def _recent(self) -> list[str]:
        return _string_list(QtCore.QSettings().value(_KEY_RECENT))[:MAX_RECENT]

    def _add_recent(self, path: str | Path) -> None:
        text = str(Path(path))
        recent = [p for p in self._recent() if p != text]
        QtCore.QSettings().setValue(_KEY_RECENT, [text, *recent][:MAX_RECENT])

    def _populate_recent_menu(self) -> None:
        menu = self._recent_menu
        menu.clear()
        recent = self._recent()
        if not recent:
            empty = menu.addAction("No recent files")
            assert empty is not None
            empty.setEnabled(False)
            return
        for number, path in enumerate(recent, start=1):
            name = Path(path).name.replace("&", "&&")
            label = f"&{number} {name}" if number < 10 else name
            action = menu.addAction(label)
            assert action is not None
            action.setToolTip(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda _=False, p=path: self._open_recent(p))
        menu.addSeparator()
        clear = menu.addAction("&Clear the list")
        assert clear is not None
        clear.triggered.connect(lambda: QtCore.QSettings().setValue(_KEY_RECENT, []))

    def _open_recent(self, path: str) -> None:
        target = Path(path)
        if not target.exists():
            QtCore.QSettings().setValue(
                _KEY_RECENT, [p for p in self._recent() if p != path]
            )
            self._show_banner(f"{target.name} is no longer at {target.parent}.")
            return
        if path.endswith(PROJECT_SUFFIX):
            self._controller.open_project(path)
        else:
            self._controller.confirm_then(lambda: self._load_file(target))

    def _populate_sessions_menu(self) -> None:
        menu = self._sessions_menu
        menu.clear()
        dirty = self._controller.dirty_sessions()
        if not dirty:
            empty = menu.addAction("No unsaved sessions")
            assert empty is not None
            empty.setEnabled(False)
            return
        for key, name in dirty:
            action = menu.addAction(name.replace("&", "&&"))
            assert action is not None
            action.setStatusTip(f"Show the session of {name} again")
            action.triggered.connect(
                lambda _=False, k=key: self._controller.activate_session(k)
            )

    # =================================================================================
    # analysis switching
    # =================================================================================
    def _on_analysis_changed(self, analysis: str) -> None:
        """The Analysis picker moved: swap docks and show that analysis's result."""
        self._apply_analysis(analysis)
        self._leave_session_views()
        self._spectrum.clear()
        # Only the *results* are stale — the light curve is unchanged, so drop the fold
        # rather than the curve itself (clear() would leave the phased panel empty until
        # the next load).
        self._phased.clear_fold()
        has_curve = self._controller.state.current_lc is not None
        if has_curve:
            self._info.set_idle("not computed yet")
            self._show_preview()
        else:
            self._info.set_idle("no light curve loaded")
            self._stack.setCurrentIndex(0)
        self._controller.set_analysis(analysis)  # type: ignore[arg-type]
        self._sync_commands()
        if self._controller_switch:
            return
        label = "Pre-whitening" if analysis == "prewhiten" else "Periodogram"
        self._show_status(f"{label} selected — press “{self._controls.action_label}”.")
        if has_curve:
            self._restore_last_result(analysis)
            if analysis == "prewhiten":
                self._schedule_preview()

    def _on_controller_analysis(self, analysis: str) -> None:
        """The controller switched the analysis (a project, File ▸ Sessions)."""
        if analysis == self._controls.analysis:
            return
        self._controller_switch = True
        try:
            self._controls.set_analysis(analysis)
        finally:
            self._controller_switch = False

    def _restore_last_result(self, analysis: str) -> None:
        """Show this analysis's last result again (UX-17), if it was for this curve.

        A pre-whitening session the bridge kept is displayed by the controller
        itself; otherwise the last request is repeated, which the cache answers at
        once (in pre-whitening with the exact session that produced it).
        """
        if analysis == "prewhiten" and self._session_view() is not None:
            last = self._last_solution
            if last is not None and last[0] == self._curve_tag():
                self._spacing_panel.set_solution(last[1])
            return
        entry = self._last_runs.get(analysis)
        if entry is not None and entry[0] == self._curve_tag():
            entry[1]()

    def _apply_analysis(self, analysis: str) -> None:
        """Show the docks that belong to ``analysis`` and raise the leading one."""
        prewhiten = analysis == "prewhiten"
        self._peaks_dock.setVisible(not prewhiten)
        for dock in (
            self._solution_dock,
            self._subsets_dock,
            self._variations_dock,
            self._history_dock,
            self._spacing_dock,
        ):
            dock.setVisible(prewhiten)
        leading = self._solution_dock if prewhiten else self._peaks_dock
        leading.raise_()
        self._analysis_actions[analysis].setChecked(True)
        QtCore.QTimer.singleShot(0, self, self._style_dock_tabs)
        self._queue_fit_minimum()
        want = self._raw_pref.get(analysis, False)
        if self._phased._show_raw.isChecked() != want:
            self._phased._show_raw.setChecked(want)

    def _leave_session_views(self) -> None:
        """No session is on show any more: views back to their plain modes."""
        self._spectrum.set_pick_enabled(False)
        self._raw.clear_session()
        self._session_bar.clear()
        self._session_bar.setVisible(False)
        self._history_panel.clear()
        self._subsets_panel.clear()
        self._variation_panel.clear()

    # =================================================================================
    # preview vs results
    # =================================================================================
    def _show_preview(self) -> None:
        """Show the loaded curve itself, full size, until a result exists."""
        self._showing_result = False
        self._stack.setCurrentIndex(1)
        self._placeholder.setVisible(True)
        self._spectrum.setVisible(False)
        self._fit_lc_area()
        self._refresh_guidance()

    def _show_results(self) -> None:
        """Swap the preview for the spectrum and the fold (raw view as chosen)."""
        self._showing_result = True
        self._stack.setCurrentIndex(1)
        self._placeholder.setVisible(False)
        self._spectrum.setVisible(True)
        self._fit_lc_area()

    def _on_show_raw_toggled(self, checked: bool) -> None:
        self._raw_pref[self._controls.analysis] = checked
        if self._raw_action.isChecked() != checked:
            self._raw_action.setChecked(checked)
        if checked:
            # Asked for: in front, also when it shares the pane with the fold.
            self._lc_tabs.setCurrentIndex(_RAW_TAB)
        self._fit_lc_area()

    def _on_raw_action(self, checked: bool) -> None:
        if self._phased._show_raw.isChecked() != checked:
            self._phased._show_raw.setChecked(checked)

    def _bring_raw_forward(self) -> None:
        """Show the raw light curve, in front of the fold when they share the pane
        (a point tool, Focus light curve)."""
        if not self._raw.isVisible() and not self._phased._show_raw.isChecked():
            self._raw_action.setChecked(True)
        self._lc_tabs.setCurrentIndex(_RAW_TAB)

    def _on_lc_tab_changed(self, _index: int) -> None:
        if self._lc_layout == "tabs":
            self._fit_lc_area()

    @property
    def lc_layout(self) -> LcLayout:
        """How the light curves are laid out now: ``"preview"`` (the raw curve
        alone, before a result), ``"single"`` (the fold alone), ``"side"`` (fold
        and raw curve side by side) or ``"tabs"`` (the two as tabs)."""
        return self._lc_layout

    def _lc_mode(self) -> LcLayout:
        if not self._showing_result:
            return "preview"
        if not self._phased._show_raw.isChecked():
            return "single"
        return "side" if self._lc_area.width() >= _SIDE_BY_SIDE_WIDTH else "tabs"

    def _queue_lc_fit(self) -> None:
        if not self._lc_fit_queued:
            self._lc_fit_queued = True
            QtCore.QTimer.singleShot(0, self, self._fit_lc_area)

    def _fit_lc_area(self, *, force: bool = False) -> None:
        """Lay the light curves out for the room there is, and share the height.

        Before a result the raw curve fills the pane alone. After one the fold
        shows, with the raw curve (when it is on) beside it where the pane is wide
        enough for both toolbars and as a second tab where it is not: one above the
        other, each plot got 45 px in a 1366 x 768 window (UX-05). While the raw
        curve is on, the spectrum and the light curve get plots of the same height,
        also as the window or the panes change size, until the user drags the split
        (it then stays where they put it until the layout changes); turning the raw
        curve off gives the spectrum back the share it had.
        """
        self._lc_fit_queued = False
        previous, mode = self._lc_layout, self._lc_mode()
        self._lc_layout = mode
        raw_tab = self._lc_tabs.currentIndex() == _RAW_TAB
        self._lc_tabs.setVisible(mode == "tabs")
        self._phased.setVisible(
            mode in ("single", "side") or (mode == "tabs" and not raw_tab)
        )
        self._raw.setVisible(
            mode in ("preview", "side") or (mode == "tabs" and raw_tab)
        )
        if mode == "side" and (previous != "side" or force):
            self._lc_split.setSizes([1000, 1000])  # half each
        with_raw = ("side", "tabs")
        if mode in with_raw:
            if previous == "single" and self._split_before_raw is None:
                self._split_before_raw = self._plot_split.sizes()
            if previous != mode or force:
                self._split_user = False
            if not self._split_user or self._split_pending:
                self._split_pending = not self._share_plot_height()
        elif mode == "single" and previous in with_raw:
            saved, self._split_before_raw = self._split_before_raw, None
            if saved is not None and len(saved) == 2 and min(saved) > 0:
                self._plot_split.setSizes(saved)

    def _on_plot_split_moved(self, _pos: int, _index: int) -> None:
        self._split_user = True

    def _share_plot_height(self) -> bool:
        """Give the spectrum and the light curve plots of the same height.

        Each pane keeps what its toolbars need at its width (rows they wrap onto
        included) and the plots split the rest evenly; in the tabs the taller of
        the two views counts, so switching tabs never moves the split. False when
        the pane has no height yet.
        """
        total = sum(self._plot_split.sizes())
        width = self._plot_split.width()
        if total < 100 or width < 100:
            return False
        top = _chrome_height(self._spectrum, width)
        if self._lc_layout == "side":
            handle = self._lc_split.handleWidth()
            half = max(1, (self._lc_area.width() - handle) // 2)
            bottom = max(
                _chrome_height(self._phased, half), _chrome_height(self._raw, half)
            )
        else:
            bottom = (
                self._lc_tabs.sizeHint().height()
                + max(0, self._lc_column.spacing())
                + max(
                    _chrome_height(self._phased, width),
                    _chrome_height(self._raw, width),
                )
            )
        plot = max(0, (total - top - bottom) // 2)
        first = min(total - 1, top + plot)
        self._plot_split.setSizes([first, total - first])
        return True

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._queue_fit_minimum()
        if self._needs_dock_widths:
            self._needs_dock_widths = False
            QtCore.QTimer.singleShot(0, self, self._default_layout)

    def _default_layout(self) -> None:
        """First launch: a narrower Controls column in a small window, and the
        docks' default share."""
        if not self._splitters_restored:
            controls = 300 if self.width() < 1200 else 340
            rest = max(1, self._main_split.width() - controls)
            self._main_split.setSizes([controls, rest])
        self._default_dock_widths()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._queue_lc_fit()

    def _refresh_guidance(self, *, busy: bool = False) -> None:
        """Rewrite the preview line for the current analysis (and run state).

        Built from the Compute button's own label, so it names the button that is
        actually there: pre-whitening users used to be told to press "Compute
        periodogram", a button their analysis does not have.
        """
        if self._loaded_detail is None:
            return
        if busy:
            tail = f"Computing {self._controls_method()}…"
        elif self._controls.analysis == "prewhiten":
            tail = (
                f'Press "{self._controls.action_label}" (Auto) to extract the '
                "frequency solution, or Step (F7) to take one peak at a time."
            )
        else:
            tail = f'Press "{self._controls.action_label}" to see the spectrum.'
        self._placeholder.setText(f"{self._loaded_detail}\n{tail}")

    def _analysis_time(self) -> FloatArray:
        """Finite times of the input the next run will analyse.

        Mirrors the controller's input selection (one band, the joint curve, or the
        stack), so the grid hint and the greyed 'auto' values describe the grid that
        will actually run rather than every band concatenated.
        """
        lc = self._controller.state.current_lc
        band = self._controls.current_band()
        if isinstance(lc, MultiBandLightCurve):
            # The all-band tokens win over a band of the same name, as in the
            # controller.
            if band not in (COMBINED_BAND, STACKED_BAND) and band in lc.bands:
                time = np.asarray(lc.bands[band].time, dtype=np.float64)
            else:
                parts = [b.time for b in lc.bands.values() if b.n]
                time = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        elif isinstance(lc, LightCurve):
            time = np.asarray(lc.time, dtype=np.float64)
        else:
            time = np.empty(0, dtype=np.float64)
        return time[np.isfinite(time)]

    def _refresh_curve_time(self, *_: object) -> None:
        self._controls.set_curve_time(self._analysis_time())

    def _on_band_changed(self, band: str) -> None:
        self._refresh_curve_time()
        if self._controls.analysis == "prewhiten":
            self._schedule_preview()

    def _schedule_preview(self) -> None:
        """Open the displayed curve's session lazily (the data spectrum shows
        without a click; single mode only, on a CPU backend — D22)."""
        state = self._controller.state
        if state.current_lc is None or state.mode != "single":
            return
        request = self._controls.prewhiten_request(report=False)
        if request is not None:
            self._controller.schedule_preview(
                request[0], request[1], self._controls.current_band()
            )

    def _on_settings_edited(self) -> None:
        """Flag a displayed result as out of date once the controls no longer match."""
        if self._controls.analysis == "prewhiten":
            self._schedule_preview()
        state = self._controller.state
        if (
            state.mode == "batch"
            and self._controls.analysis == "periodogram"
            and self._showing_result
            and self._controls.current_method() != state.method
            and self._controls.can_compute()
        ):
            # Batch browsing computes as it goes; a method picked mid-batch would
            # otherwise leave the old method's spectrum under the new name (UX-23).
            self._controls.request_compute()
            return
        if not self._showing_result or self._info.is_stale():
            return
        self._info.set_stale(True)
        self._show_status(
            f"Settings changed — press “{self._controls.action_label}” to update "
            "the result."
        )

    # =================================================================================
    # controller signals
    # =================================================================================
    def _on_lc_loaded(self, lc: object) -> None:
        self._load_count += 1
        self._hide_banner()  # a successful load supersedes the last error
        # Clear the previous source's results until the new spectrum is computed.
        self._clear_results()
        self._info.set_idle("not computed yet")

        self._controls.set_enabled(True)
        if isinstance(lc, MultiBandLightCurve):
            # set_bands announces the band, and the controls then pick up its times.
            self._controls.set_bands(list(lc.band_names))
            self._controls.set_multiband(True)
        elif isinstance(lc, LightCurve):
            self._controls.set_bands(None)
            self._controls.set_multiband(False)
        else:
            return
        domain = _curve_domain(lc)
        self._spectrum.set_amplitude_unit(domain)
        unit = None
        if domain is not None:
            unit = "mag" if domain == Domain.MAGNITUDE else "flux"
        self._solution_panel.set_amplitude_unit(unit)
        source = self._controller.state.source_id
        if source.startswith(PROJECT_PREFIX):
            self._data_name = self._project_data_name(lc, source)
        time = self._controller.current_time()
        time = time[np.isfinite(time)]
        baseline = float(time.max() - time.min()) if time.size else 0.0
        action = self._controls.action_label
        if isinstance(lc, MultiBandLightCurve):
            points = sum(band.n for band in lc.bands.values())
            note = f"Multiband curve loaded ({lc.n_bands} bands) — press {action}."
            detail = (
                f"Loaded: multiband · {lc.n_bands} bands · {points} points · "
                f"baseline {baseline:.1f} d"
            )
        else:
            note = f"Light curve loaded — press {action}."
            detail = (
                f"Loaded: {lc.n} points · baseline {baseline:.1f} d · {lc.domain.value}"
            )
        # What the reader had to do to the file (rows dropped for want of a band,
        # a band column ignored) is said where the curve is described.
        notes = [str(n) for n in lc.meta.get(LOAD_NOTES_KEY, ()) if str(n)]
        if notes:
            detail += "\n" + "\n".join(notes)
            note += "  Note: " + "; ".join(notes)
        self._loaded_detail = detail
        self._show_preview()
        self._show_status(note)
        self._update_title()
        self._sync_commands()
        if self._controller.state.mode == "batch":
            self._controls.request_compute()  # compute on demand as sources scroll
        elif self._controls.analysis == "prewhiten":
            self._schedule_preview()

    @staticmethod
    def _project_data_name(lc: object, source: str) -> str:
        meta = getattr(lc, "meta", {}) or {}
        for key in ("name", "source", "path", "file"):
            value = meta.get(key) if isinstance(meta, Mapping) else None
            if isinstance(value, str) and value:
                return Path(value).name
        return Path(source[len(PROJECT_PREFIX) :]).name.removesuffix(PROJECT_SUFFIX)

    def _clear_results(self) -> None:
        self._spectrum.clear()
        self._phased.clear()
        self._peaks_table.clear()
        self._solution_panel.clear()
        self._spacing_panel.clear()
        self._leave_session_views()

    def _on_compute_started(self) -> None:
        self._hide_banner()
        method = self._controls_method()
        self._info.set_busy(True, method)
        self._show_status(f"Computing {method}…  (Esc stops)")
        if not self._showing_result:
            self._refresh_guidance(busy=True)

    def _on_busy_changed(self, busy: bool) -> None:
        self._busy = busy
        self._info.set_busy(busy)
        self._controls.set_busy(busy)
        self._solution_panel.set_busy(busy)
        self._session_bar.set_busy(busy)
        self._subsets_panel.set_busy(busy)
        self._variation_panel.set_busy(busy)
        self._sync_commands()
        if not busy:
            self._poll_finishing()

    def _on_periodogram_ready(self, pg: object) -> None:
        assert isinstance(pg, Periodogram)
        self._hide_banner()
        self._peaks_table.set_statistic(statistic_name(pg))
        self._info.set_result(pg)
        timing = self._timing_text() or "<1 ms"
        self._info.set_timing(timing)
        self._show_results()
        self._show_status(
            f"{pg.method} via {pg.backend} — best period "
            f"{pg.best_period():.6g} d — {timing}"
        )

    def _on_solution_ready(self, result: object) -> None:
        """A pre-whitening result: Auto's completion, a finalize, a cached one, or
        the light result of an edit that moved the active frequencies.

        A light result arrives with the session view that carries it (the
        Frequencies table, spectrum and light curve are drawn from the view), so only
        the Period spacing dock and the badges take it. A full result is drawn
        read-only at once and the session's next view makes the table editable.
        """
        assert isinstance(result, PreWhitenResult)
        self._hide_banner()
        self._last_solution = (self._curve_tag(), result)
        self._spacing_panel.set_solution(result)
        view = self._session_view()
        light = view is not None and view.result is result
        if not light:
            self._show_solution_spectrum(result)
            self._solution_panel.set_solution(result)
        self._show_results()
        timing = self._timing_text()
        self._info.set_timing(timing or "—")
        strongest = (
            f"strongest {result.components[0].frequency:.6g} c/d"
            if result.components
            else "no significant frequency"
        )
        if not light:
            self._show_status(
                f"Pre-whitening via {result.backend} — {result.n_components} "
                f"components, {strongest}" + (f" — {timing}" if timing else "")
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
        # fitted amplitude stays on the hover readout and in the Frequencies dock,
        # where their disagreement is reported as the blend ratio.
        heights = spectrum.amplitude_at(
            np.asarray([c.frequency for c in result.components], dtype=np.float64)
        )
        peaks = [
            Peak(
                period=component.period,
                frequency=component.frequency,
                power=float(heights[i]),
                rank=component.rank,
                extra={
                    "snr": component.snr,
                    "amplitude": component.amplitude,
                    "blended": float(component.blended),
                },
            )
            for i, component in enumerate(result.components)
        ]
        self._spectrum.set_peaks(peaks)
        if peaks:
            self._spectrum.set_selected_period(peaks[0].period)

    # -- the session ---------------------------------------------------------------
    def _on_session_changed(self, view: object) -> None:
        """Paint the displayed session (every committed state, and Auto's steps)."""
        if not isinstance(view, SessionView):
            return
        if self._controls.analysis != "prewhiten":
            return
        selected = self._controller.state.selected_components
        self._spectrum.set_pick_enabled(True)
        if not self._showing_result:
            self._show_results()
        self._spectrum.set_session_view(view, selected=selected)
        self._raw.set_session_view(
            view, domain=_curve_domain(self._controller.state.current_lc)
        )
        self._solution_panel.set_view(view)
        self._session_bar.setVisible(True)
        self._session_bar.set_view(view)
        self._history_panel.set_view(view)
        self._subsets_panel.set_view(view)
        self._variation_panel.set_view(view)
        label_columns = getattr(view, "label_columns", ())
        if label_columns:
            self._subsets_panel.set_label_columns(label_columns)
        if self._info.is_idle():
            self._info.set_result(self._wrap_view(view))
        self._sync_commands()

    @staticmethod
    def _wrap_view(view: SessionView) -> Periodogram:
        """The session's data spectrum as a periodogram, for the badges."""
        spectrum = view.data_spectrum
        time = np.asarray(view.time, dtype=np.float64)
        used = time[~np.asarray(view.deleted, dtype=np.bool_)] if time.size else time
        used = used[np.isfinite(used)]
        baseline = float(used.max() - used.min()) if used.size >= 2 else 0.0
        return Periodogram.from_spectrum(
            method=PREWHITEN_METHOD,
            backend=view.backend or "?",
            frequency=np.asarray(spectrum.frequency, dtype=np.float64),
            power=np.asarray(spectrum.amplitude, dtype=np.float64),
            objective_sense="max",
            n_samples=int(used.size),
            baseline=baseline,
        )

    def _on_session_progress(self, report: object) -> None:
        """Narrate Auto in the info bar ("Auto — F7 accepted, S/N 5.3")."""
        kind = str(getattr(report, "kind", ""))
        component = getattr(report, "component_id", None)
        stats = getattr(report, "statistics", {}) or {}
        snr = stats.get("snr") if isinstance(stats, Mapping) else None
        parts = [f"F{component} {kind}" if component is not None else kind]
        if isinstance(snr, float) and math.isfinite(snr):
            parts.append(f"S/N {snr:.1f}")
        text = "Auto — " + ", ".join(p for p in parts if p)
        self._info.set_activity(text)

    def _on_session_message(self, severity: str, text: str) -> None:
        """Warnings, refusals and notices go to the status bar; errors also to the
        banner.

        A refused edit leaves the table showing the user's change (a check box keeps
        its click), so the table is put back to the session's values.
        """
        text = gui_wording(text)
        self._show_status(text)
        if severity in ("warning", "error"):
            self._solution_panel.resync()
        if severity == "error":
            self._show_banner(text)

    def _on_session_refused(self, refusal: object) -> None:
        """A refused edit: why, in the window's terms (from the library's remedy
        code, else by rewording its message), and — when the remedy is one the
        window can carry out — the offer to do it, asked through the dialogs."""
        if not isinstance(refusal, Refusal):
            return
        self._on_session_message("warning", refusal_text(refusal))
        offer = refusal_offer(refusal, self._offer_rows(refusal))
        if offer is None:
            return

        def answered(accepted: bool) -> None:
            if accepted:
                self._controller.retry_refused(refusal)
            else:
                self._show_status("Nothing was changed.")

        self._dialogs.confirm(offer.title, offer.text, offer.accept, answered)

    def _offer_rows(self, refusal: Refusal) -> Sequence[Any]:
        """The components of the refused session (a tie parent's children are named
        from them), or none when another session is on show."""
        view = self._session_view()
        return view.rows if view is not None and view.key == refusal.key else ()

    def _on_components_selected(self, ids: object) -> None:
        chosen = tuple(int(i) for i in ids) if isinstance(ids, Sequence) else ()
        self._solution_panel.select_components(chosen)
        self._variation_panel.set_selected_components(chosen)
        view = self._session_view()
        if view is not None and self._controls.analysis == "prewhiten":
            self._spectrum.set_marker_states(marker_states(view.rows, chosen))

    def _on_cursor_changed(self, frequency: float) -> None:
        self._spectrum.set_cursor(frequency if math.isfinite(frequency) else None)

    def _on_points_selected(self, count: int) -> None:
        if count:
            self._show_status(
                f"{count} point{'s' if count != 1 else ''} selected — Del deletes, "
                "U restores, Esc clears."
            )

    def _on_time_cursor(self, time: float) -> None:
        """The light curve's time cursor moved (a click, or Left/Right): Subsets ▸
        Split at cuts there. NaN (another session) clears it there too."""
        self._subsets_panel.set_time_cursor(time)
        if math.isfinite(time):
            self._show_status(
                f"Time cursor at t = {time:.5f} — Subsets ▸ Split at cuts there."
            )

    def _on_points_highlighted(self, mask: object) -> None:
        """A subset row selected in the Subsets dock: select its points (so Delete
        and Restore act on that night)."""
        if mask is None:
            self._raw.clear_selection()
            return
        if self._session_view() is not None:
            self._raw.set_selection(mask)

    def _on_project_changed(self, path: str, dirty: bool) -> None:
        if path:
            self._add_recent(path)
        if self._controller.state.mode == "single":
            self._sources_dock.setVisible(False)
        self._update_title()

    def _on_preview_failed(self, text: str) -> None:
        """The lazy spectrum preview failed: say so where the spectrum would be.

        The session commands go off (state 4.9, from the controller's
        ``preview_error``). The spectrum pane comes up above the light curve with
        the text inside it, and the guidance line repeats it. A result still on show
        is taken down first: it belongs to another session (the band picked before,
        say), as the displayed one is the session that could not open, and the
        failure used to reach only the status bar under it.
        """
        self._show_status(text)
        self._sync_commands()
        if self._showing_result:
            self._leave_session_views()
            self._solution_panel.clear()
            self._spacing_panel.clear()
            self._spectrum.clear()
            self._phased.clear_fold()  # the curve stays; only the fold was a result
            self._info.set_idle("not computed yet")
            self._show_preview()
        detail = f"{self._loaded_detail}\n" if self._loaded_detail else ""
        self._placeholder.setVisible(True)
        self._placeholder.setText(f"{detail}{text}")
        self._spectrum.set_placeholder(text)
        self._spectrum.setVisible(True)

    def _on_settings_restored(self, settings: object) -> None:
        if isinstance(settings, PreWhitenSettings):
            self._controls.set_prewhiten_settings(settings)

    def _timing_text(self) -> str:
        state = self._controller.state
        if state.last_from_cache:
            return "cached"
        elapsed = state.last_compute_ms
        if elapsed <= 0.0:
            return ""  # not timed (a Polish and prune said "0 ms")
        return f"{elapsed / 1000:.2f} s" if elapsed >= 1000 else f"{elapsed:.0f} ms"

    def _on_compute_failed(self, message: str) -> None:
        summary, _ = explain_failure(message)
        first_line = summary.splitlines()[0] if summary else ""
        failed_source, self._failed_source = self._failed_source, None
        if failed_source is not None:
            # A batch source failed to *load*. The controller stayed on the previous
            # source, so whatever is displayed still matches the badges: leave them.
            self._source_browser.mark_failed(failed_source, summary)
            self._show_status(f"Load failed: {first_line}")
            action: tuple[str, Callable[[], None]] | None = None
            if "ColumnResolutionError" in message:
                action = (
                    "Choose columns…",
                    partial(self._choose_columns_for_source, failed_source),
                )
            self._show_banner(message, action=action)
            return
        self._info.set_failed(self._controls_method())
        self._show_status(f"{self._controls.action_label} failed: {first_line}")
        self._show_banner(message)
        if not self._showing_result:
            self._refresh_guidance()

    # =================================================================================
    # helpers
    # =================================================================================
    def _controls_method(self) -> str:
        if self._controller.state.analysis == "prewhiten":
            return PREWHITEN_METHOD
        return self._controller.state.method

    def _show_status(self, message: str) -> None:
        status = self.statusBar()
        if status is not None:
            status.showMessage(message)

    def _update_title(self) -> None:
        """``data — project.cupw.json* — cuPeriod`` (``*``: unsaved session work).

        The ``*`` is Qt's modified marker (``[*]``), which macOS shows as the dot
        in the close button instead.
        """
        path = self._controller.state.project_path
        project = Path(path).name if path else ""
        self.setWindowTitle(window_title(self._data_name, project))
        # Only a title naming data has the marker (Qt warns about one without).
        named = bool(self._data_name or project)
        self.setWindowModified(named and self._controller.is_session_dirty())

    def _warning_box(self, message: str) -> QtWidgets.QMessageBox:
        """The warning dialog for ``message``: plain text, a short summary, details.

        Built separately from :meth:`_warn` so it can be inspected without showing.
        """
        summary, details = explain_failure(message)
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Warning,
            "cuPeriod",
            summary,
            QtWidgets.QMessageBox.StandardButton.Ok,
            self,
        )
        # The message quotes file names and column headers: never rich text.
        box.setTextFormat(Qt.TextFormat.PlainText)
        if details:
            box.setDetailedText(details)
        return box

    def _warn(self, message: str) -> None:
        """Show a warning without a nested event loop (``open()``, not ``exec()``)."""
        box = self._warning_box(message)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.open()

    def _show_banner(
        self,
        message: str,
        *,
        action: tuple[str, Callable[[], None]] | None = None,
    ) -> None:
        """Show a dismissable error banner, with an optional next-step button.

        It stays until dismissed or superseded (a new attempt, a new result or a new
        curve): an error that vanishes on a timer is missed by anyone who looked away
        or reads slowly. The summary is plain language; the full text is in the
        tooltip, and the label can be selected and copied.
        """
        summary, details = explain_failure(message)
        self._banner.setText(f"⚠  {summary}")
        self._banner.setToolTip(rich_tooltip(details) if details else "")
        self._banner.setAccessibleDescription(details or summary)
        if action is None:
            self._banner_action_slot = None
            self._banner_action.setVisible(False)
        else:
            text, slot = action
            self._banner_action_slot = slot
            self._banner_action.setText(text)
            self._banner_action.setToolTip(text.rstrip("…"))
            self._banner_action.setVisible(True)
        self._banner_container.setVisible(True)
        # Screen readers announce an alert without the user having to find it.
        QtGui.QAccessible.updateAccessibility(
            QtGui.QAccessibleEvent(self._banner, QtGui.QAccessible.Event.Alert)
        )

    def _on_banner_action(self) -> None:
        slot = self._banner_action_slot
        if slot is not None:
            slot()

    def _hide_banner(self) -> None:
        self._banner_container.setVisible(False)

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._restyle()
        QtCore.QSettings().setValue("theme", self._theme)

    def _zoom(self, steps: int) -> None:
        """Zoom the whole UI ``steps`` notches (0 = back to the designed size)."""
        scale = 1.0 if steps == 0 else step_font_scale(font_scale(), steps)
        self._restyle(scale)
        QtCore.QSettings().setValue("ui_scale", scale)
        self._show_status(f"Zoom {font_scale():.0%}")
        # After the new sheet has reached every widget: fixed column widths would
        # clip the larger text, and the panes need more room.
        QtCore.QTimer.singleShot(0, self, self._after_zoom)

    def _after_zoom(self) -> None:
        for table in self.findChildren(QtWidgets.QTableView):
            if isinstance(table, FrequencyTable):
                table.fit_columns()  # keeps room for the ID check box
            else:
                table.resizeColumnsToContents()
        self._fit_minimum()

    def _restyle(self, scale: float | None = None) -> None:
        """Re-apply the theme (and zoom ``scale``; None keeps it) to every view."""
        apply_theme(self._app, self._theme, font_scale=scale)
        pal = palette(self._theme)
        self._spectrum.apply_theme(pal)
        self._phased.apply_theme(pal)
        self._raw.apply_theme(pal)
        self._spacing_panel.apply_theme(pal)
        self._variation_panel.apply_theme(pal)
        self._theme_pieces(pal)

    def _theme_pieces(self, pal: ThemePalette) -> None:
        """Theme what the application sheet cannot reach (drawn icons, the table's
        check boxes)."""
        self._solution_panel.apply_theme(pal)
        for bar in self._title_bars:
            bar.apply_theme(pal)

    def _compute_via_shortcut(self) -> None:
        """Ctrl+Return/Ctrl+Enter/F5: request a compute if one would currently run."""
        if self._controls.can_compute():
            self._controls.request_compute()

    # =================================================================================
    # lifecycle
    # =================================================================================
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Ask about unsaved sessions, then stop compute and persist the layout.

        With unsaved work the close is refused while the Save / Discard / Cancel
        question is open; the controller's ``close_approved`` closes the window
        again once the work is saved or discarded.
        """
        if not self._controller.confirm_close():
            event.ignore()
            return
        self._controller.shutdown()
        self._save_window_state()
        super().closeEvent(event)

    def _save_window_state(self) -> None:
        settings = QtCore.QSettings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        splitters = {
            name: bytes(splitter.saveState().toBase64().data()).decode("ascii")
            for name, splitter in self._splitters().items()
        }
        settings.setValue(_KEY_SPLITTERS, json.dumps(splitters))
        settings.setValue(_KEY_CONTROLS, self._controls.saved_state())
        settings.setValue(_KEY_RAW, json.dumps(self._raw_pref))

    # -- drag and drop -------------------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if self._is_acceptable_drop(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        mime = event.mimeData()
        if not self._is_acceptable_drop(mime):
            event.ignore()
            return
        event.acceptProposedAction()
        paths = [Path(url.toLocalFile()) for url in mime.urls()]
        # Load after the drop has returned: on Windows the drag source (Explorer)
        # waits inside DoDragDrop until the drop handler returns, so opening the
        # import dialog from here would freeze it until the dialog was closed.
        QtCore.QTimer.singleShot(0, self, lambda: self._load_dropped(paths))

    def _load_dropped(self, paths: list[Path]) -> None:
        # The drop leaves the drag source's window in front; bring this one forward
        # so an import or warning dialog does not open behind it.
        self.raise_()
        self.activateWindow()
        self._controller.confirm_then(lambda: self._handle_dropped_paths(paths))

    @staticmethod
    def _is_acceptable_drop(mime: QtCore.QMimeData) -> bool:
        """Local items of which at least one is a folder or a light-curve file.

        Anything else in a mixed selection (a README next to the CSVs) is skipped
        and reported by :meth:`_handle_dropped_paths` rather than refusing the drop.
        """
        if not mime.hasUrls():
            return False
        urls = mime.urls()
        if not urls or not all(url.isLocalFile() for url in urls):
            return False
        paths = [Path(url.toLocalFile()) for url in urls]
        return any(
            p.is_dir()
            or p.suffix.lower() in _FILE_EXTENSIONS
            or p.name.endswith(PROJECT_SUFFIX)
            for p in paths
        )

    def _handle_dropped_paths(self, paths: list[Path]) -> None:
        """Load dropped files/folders the same way as the File menu's Open actions.

        A single directory is browsed like "Open folder…"; a single file is opened
        (through the import dialog) like "Open…", a project like "Open project…";
        anything more is loaded as one batch, folders expanded into their light
        curves. Items that are not light-curve files are skipped and named in the
        status bar.
        """
        if not paths:
            return
        if len(paths) == 1 and paths[0].is_dir():
            self._load_folder(paths[0])
            return
        if len(paths) == 1 and paths[0].name.endswith(PROJECT_SUFFIX):
            self._controller.open_project(str(paths[0]))
            return
        folders = [p for p in paths if p.is_dir()]
        files = [p for p in paths if p.suffix.lower() in _FILE_EXTENSIONS]
        files = [p for p in files if not p.is_dir()]
        skipped = [p.name for p in paths if p not in folders and p not in files]
        if not folders and not files:
            self._warn("None of the dropped files are recognized light-curve files.")
            return
        if not folders and len(files) == 1:
            self._load_file(files[0])
            self._report_skipped(skipped)
            return
        items: list[SourceItem] = []
        try:
            if files:
                items += enumerate_sources([str(p) for p in files])
            for folder in folders:
                # Label each source with its folder too: two folders often hold
                # files of the same name, and identical rows cannot be told apart.
                # (The key is the file's path already; results are cached by data.)
                items += [
                    dataclasses.replace(item, label=f"{folder.name}/{item.label}")
                    for item in enumerate_sources(str(folder))
                ]
        except Exception as exc:  # noqa: BLE001 - surface any enumeration error
            self._warn(f"Could not load dropped files:\n{type(exc).__name__}: {exc}")
            return
        if not items:
            self._warn("No light-curve files found among the dropped items.")
            return
        self._data_name = f"dropped batch ({len(items)} sources)"
        self._controller.load_sources(items)
        self._update_title()
        self._report_skipped(skipped)

    def _report_skipped(self, names: list[str]) -> None:
        if not names:
            return
        shown = ", ".join(names[:5]) + (", …" if len(names) > 5 else "")
        self._show_status(
            f"Skipped {len(names)} dropped item(s) that are not light-curve files: "
            f"{shown}"
        )


__all__ = [
    "APP_TITLE",
    "DOCS_URL",
    "MAX_RECENT",
    "LcLayout",
    "MainWindow",
    "assign_mnemonics",
    "explain_failure",
    "gui_wording",
    "import_choice_from_json",
    "import_choice_to_json",
    "window_title",
]
