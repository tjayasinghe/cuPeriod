"""Theme accessibility: WCAG 2.1 contrast of both stylesheets and the plot palette,
visible arrows / ticks / keyboard-focus rings (rendered offscreen), a colour-blind-safe
band cycle, UI zoom, and the plot axes' SI-prefix scaling.

The colour maths below is written out independently of :mod:`cuperiod.gui.theme` so
the tests do not trust the code they check. Contrast is the WCAG 2.1 ratio; the
colour-vision-deficiency (CVD) simulation is Machado, Oliveira & Fernandes (2009) at
full severity, and colour difference is CIEDE2000.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.gui import theme as th
from cuperiod.gui.qt import Qt, QtCore, QtGui, QtWidgets, pg

THEMES = ("dark", "light")
RESOURCES = Path(th.__file__).parent / "resources"

# -- colour maths ------------------------------------------------------------------

RGB = tuple[float, float, float]


def _rgb(hex_str: str) -> RGB:
    h = hex_str.strip().lstrip("#")
    assert len(h) == 6, hex_str
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(c: RGB) -> float:
    return 0.2126 * _linear(c[0]) + 0.7152 * _linear(c[1]) + 0.0722 * _linear(c[2])


def _contrast(a: str | RGB, b: str | RGB) -> float:
    la = _luminance(_rgb(a) if isinstance(a, str) else a)
    lb = _luminance(_rgb(b) if isinstance(b, str) else b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _over(fg: str, bg: str, alpha: int) -> RGB:
    """``fg`` at ``alpha`` (0-255) composited over ``bg``, in sRGB as QPainter does."""
    f, b = _rgb(fg), _rgb(bg)
    a = alpha / 255.0
    return (
        f[0] * a + b[0] * (1 - a),
        f[1] * a + b[1] * (1 - a),
        f[2] * a + b[2] * (1 - a),
    )


_CVD = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _simulate(c: RGB, kind: str) -> RGB:
    if kind == "normal":
        return c
    lin = [_linear(v) for v in c]
    out = []
    for row in _CVD[kind]:
        v = min(1.0, max(0.0, sum(m * x for m, x in zip(row, lin, strict=True))))
        out.append(
            255.0 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)
        )
    return (out[0], out[1], out[2])


def _lab(c: RGB) -> RGB:
    r, g, b = (_linear(v) for v in c)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _ciede2000(c1: RGB, c2: RGB) -> float:
    (L1, a1, b1), (L2, a2, b2) = _lab(c1), _lab(c2)
    cbar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cbar**7 / (cbar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0
    dh = h2p - h1p
    if c1p * c2p == 0:
        dh = 0.0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    d_l, d_c = L2 - L1, c2p - c1p
    d_h = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    lbar, cbarp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbar = (h1p + h2p) / 2
    else:
        hbar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hbar - 30))
        + 0.24 * math.cos(math.radians(2 * hbar))
        + 0.32 * math.cos(math.radians(3 * hbar + 6))
        - 0.20 * math.cos(math.radians(4 * hbar - 63))
    )
    rc = 2 * math.sqrt(cbarp**7 / (cbarp**7 + 25**7))
    rt = -math.sin(math.radians(60 * math.exp(-(((hbar - 275) / 25) ** 2)))) * rc
    sl = 1 + 0.015 * (lbar - 50) ** 2 / math.sqrt(20 + (lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * cbarp, 1 + 0.015 * cbarp * t
    return math.sqrt(
        (d_l / sl) ** 2
        + (d_c / sc) ** 2
        + (d_h / sh) ** 2
        + rt * (d_c / sc) * (d_h / sh)
    )


def _cvd_distance(a: str, b: str, kinds: tuple[str, ...]) -> float:
    """Smallest CIEDE2000 between ``a`` and ``b`` over the simulated ``kinds``."""
    ca, cb = _rgb(a), _rgb(b)
    return min(_ciede2000(_simulate(ca, k), _simulate(cb, k)) for k in kinds)


def test_colour_maths_matches_known_values() -> None:
    assert _contrast("#ffffff", "#000000") == pytest.approx(21.0)
    assert _contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)
    assert _ciede2000(_rgb("#ffffff"), _rgb("#000000")) == pytest.approx(100.0, abs=0.1)
    assert _ciede2000(_rgb("#56b4e9"), _rgb("#56b4e9")) == 0.0
    # the public helper agrees with the reference maths
    assert th.contrast_ratio("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)
    assert th.contrast_ratio(QtGui.QColor("#000000"), "#ffffff") == pytest.approx(21.0)


# -- stylesheet parsing --------------------------------------------------------------

Rules = dict[str, dict[str, str]]
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def _parse_qss(text: str) -> Rules:
    """``{selector: {property: value}}``; later declarations win, as in Qt."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules: Rules = {}
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        decls: dict[str, str] = {}
        for decl in match.group(2).split(";"):
            if ":" in decl:
                key, value = decl.split(":", 1)
                decls[key.strip().lower()] = value.strip()
        for selector in match.group(1).split(","):
            rules.setdefault(" ".join(selector.split()), {}).update(decls)
    return rules


@pytest.fixture(params=THEMES)
def theme_rules(request: pytest.FixtureRequest) -> tuple[str, Rules]:
    return request.param, _parse_qss(th.load_qss(request.param))


def _colour(rules: Rules, selector: str, prop: str) -> str:
    """The hex colour ``selector`` gives ``prop`` (shorthands resolved)."""
    decls = rules.get(selector)
    assert decls is not None, f"no rule for {selector!r}"
    candidates = [prop]
    if prop == "background-color":
        candidates.append("background")
    if prop.startswith("border") and prop.endswith("-color"):
        side = prop[len("border-") : -len("-color")]
        candidates += (
            [f"border-{side}", "border-color", "border"] if side else ["border"]
        )
    for key in candidates:
        if key in decls:
            found = _HEX.search(decls[key])
            if found:
                return found.group(0).lower()
    raise AssertionError(f"{selector!r} sets no colour for {prop!r}: {decls}")


def _border_width(rules: Rules, selector: str) -> int:
    """Border width of ``selector``, inherited from its rule without ``:focus``."""
    for sel in (selector, selector.removesuffix(":focus")):
        decls = rules.get(sel, {})
        for key in ("border-width", "border"):
            found = re.search(r"(\d+)px", decls.get(key, ""))
            if found:
                return int(found.group(1))
    raise AssertionError(f"{selector!r} sets no border width")


# (label, (selector, property) of the text, (selector, property) of its background)
_TEXT_PAIRS: list[tuple[str, tuple[str, str], tuple[str, str]]] = [
    ("body text", ("QWidget", "color"), ("QWidget", "background-color")),
    (
        "body text on the main window",
        ("QWidget", "color"),
        ("QMainWindow", "background-color"),
    ),
    ("body text on a card", ("QWidget", "color"), ("QFrame#card", "background-color")),
    ("muted text", ("QLabel#muted", "color"), ("QWidget", "background-color")),
    (
        "muted text on the main window",
        ("QLabel#muted", "color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "muted text on a group box",
        ("QLabel#muted", "color"),
        ("QGroupBox", "background-color"),
    ),
    ("heading", ("QLabel#heading", "color"), ("QMainWindow", "background-color")),
    ("badge", ("QLabel#badge", "color"), ("QLabel#badge", "background-color")),
    (
        "accent badge",
        ("QLabel#badgeAccent", "color"),
        ("QLabel#badgeAccent", "background-color"),
    ),
    ("error text", ("QLabel#error", "color"), ("QWidget", "background-color")),
    (
        "error text on the main window",
        ("QLabel#error", "color"),
        ("QMainWindow", "background-color"),
    ),
    ("tooltip", ("QToolTip", "color"), ("QToolTip", "background-color")),
    ("button", ("QPushButton", "color"), ("QPushButton", "background-color")),
    (
        "button :hover",
        ("QPushButton", "color"),
        ("QPushButton:hover", "background-color"),
    ),
    (
        "button :pressed",
        ("QPushButton", "color"),
        ("QPushButton:pressed", "background-color"),
    ),
    (
        "checked tool button",
        ("QToolButton:checked", "color"),
        ("QToolButton:checked", "background-color"),
    ),
    (
        "primary button",
        ('QPushButton[primary="true"]', "color"),
        ('QPushButton[primary="true"]', "background-color"),
    ),
    (
        "primary button :hover",
        ('QPushButton[primary="true"]', "color"),
        ('QPushButton[primary="true"]:hover', "background-color"),
    ),
    (
        "primary button :pressed",
        ('QPushButton[primary="true"]', "color"),
        ('QPushButton[primary="true"]:pressed', "background-color"),
    ),
    ("input text", ("QLineEdit", "color"), ("QLineEdit", "background-color")),
    (
        "selected text in an input",
        ("QLineEdit", "selection-color"),
        ("QLineEdit", "selection-background-color"),
    ),
    (
        "selected text elsewhere",
        ("QWidget", "selection-color"),
        ("QWidget", "selection-background-color"),
    ),
    (
        "value shown for an 'auto' setting",
        ('QAbstractSpinBox[auto="true"]', "color"),
        ('QAbstractSpinBox[auto="true"]', "background-color"),
    ),
    (
        "combo popup item",
        ("QWidget", "color"),
        ("QComboBox QAbstractItemView", "background-color"),
    ),
    (
        "combo popup current item",
        ("QComboBox QAbstractItemView", "selection-color"),
        ("QComboBox QAbstractItemView", "selection-background-color"),
    ),
    ("menu item", ("QMenu", "color"), ("QMenu", "background-color")),
    (
        "menu current item",
        ("QMenu::item:selected", "color"),
        ("QMenu::item:selected", "background-color"),
    ),
    (
        "group box title",
        ("QGroupBox::title", "color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "group box title on the panel",
        ("QGroupBox::title", "color"),
        ("QWidget", "background-color"),
    ),
    ("status bar", ("QStatusBar", "color"), ("QStatusBar", "background-color")),
    ("table item", ("QWidget", "color"), ("QTableView", "background-color")),
    (
        "table item on an alternate row",
        ("QWidget", "color"),
        ("QTableView", "alternate-background-color"),
    ),
    (
        "selected table item",
        ("QTableView::item:selected", "color"),
        ("QTableView::item:selected", "background-color"),
    ),
    (
        "hovered table item",
        ("QWidget", "color"),
        ("QTableView::item:hover", "background-color"),
    ),
    (
        "table header",
        ("QHeaderView::section", "color"),
        ("QHeaderView::section", "background-color"),
    ),
    ("progress text", ("QProgressBar", "color"), ("QProgressBar", "background-color")),
    ("tab", ("QTabBar::tab", "color"), ("QTabBar::tab", "background-color")),
    (
        "selected tab",
        ("QTabBar::tab:selected", "color"),
        ("QTabBar::tab:selected", "background-color"),
    ),
    (
        "focused tab",
        ("QTabBar::tab:selected", "color"),
        ("QTabBar::tab:focus", "background-color"),
    ),
    (
        "dock title",
        ("QDockWidget", "color"),
        ("QDockWidget::title", "background-color"),
    ),
]


@pytest.mark.parametrize(
    ("label", "fg", "bg"), _TEXT_PAIRS, ids=[p[0] for p in _TEXT_PAIRS]
)
def test_stylesheet_text_meets_wcag_aa(
    theme_rules: tuple[str, Rules], label: str, fg: tuple[str, str], bg: tuple[str, str]
) -> None:
    """WCAG 2.1 SC 1.4.3: 4.5:1 for every text/background pair (13 px is small)."""
    theme, rules = theme_rules
    text, back = _colour(rules, *fg), _colour(rules, *bg)
    ratio = _contrast(text, back)
    assert ratio >= 4.5, f"{theme}: {label} {text} on {back} is {ratio:.2f}:1"


# (label, colour (selector, property), neighbour (selector, property)) - SC 1.4.11
_BOUNDARY_PAIRS: list[tuple[str, tuple[str, str], tuple[str, str]]] = [
    (
        "input border vs its field",
        ("QLineEdit", "border-color"),
        ("QLineEdit", "background-color"),
    ),
    (
        "input border vs the panel",
        ("QLineEdit", "border-color"),
        ("QWidget", "background-color"),
    ),
    (
        "input border vs the main window",
        ("QLineEdit", "border-color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "input border vs a group box",
        ("QLineEdit", "border-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "button border vs the panel",
        ("QPushButton", "border-color"),
        ("QWidget", "background-color"),
    ),
    (
        "button border vs the main window",
        ("QPushButton", "border-color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "checkbox border vs the panel",
        ("QCheckBox::indicator", "border-color"),
        ("QWidget", "background-color"),
    ),
    (
        "checkbox border vs its box",
        ("QCheckBox::indicator", "border-color"),
        ("QCheckBox::indicator", "background-color"),
    ),
    (
        "checked box vs unchecked box",
        ("QCheckBox::indicator:checked", "background-color"),
        ("QCheckBox::indicator", "background-color"),
    ),
    (
        "checked tool button vs the panel",
        ("QToolButton:checked", "border-color"),
        ("QWidget", "background-color"),
    ),
    (
        "selected row vs a row",
        ("QTableView::item:selected", "background-color"),
        ("QTableView", "background-color"),
    ),
    (
        "selected row vs an alternate row",
        ("QTableView::item:selected", "background-color"),
        ("QTableView", "alternate-background-color"),
    ),
    (
        "current combo item vs the popup",
        ("QComboBox QAbstractItemView", "selection-background-color"),
        ("QComboBox QAbstractItemView", "background-color"),
    ),
    (
        "current menu item vs the menu",
        ("QMenu::item:selected", "background-color"),
        ("QMenu", "background-color"),
    ),
    (
        "selected tab marker vs the tab",
        ("QTabBar::tab:selected", "border-top-color"),
        ("QTabBar::tab:selected", "background-color"),
    ),
    (
        "selected tab marker vs an unselected tab",
        ("QTabBar::tab:selected", "border-top-color"),
        ("QTabBar::tab", "background-color"),
    ),
    (
        "scrollbar handle vs a table",
        ("QScrollBar::handle", "background-color"),
        ("QTableView", "background-color"),
    ),
    (
        "scrollbar handle vs the main window",
        ("QScrollBar::handle", "background-color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "progress chunk vs its trough",
        ("QProgressBar::chunk", "background-color"),
        ("QProgressBar", "background-color"),
    ),
]


@pytest.mark.parametrize(
    ("label", "fg", "bg"), _BOUNDARY_PAIRS, ids=[p[0] for p in _BOUNDARY_PAIRS]
)
def test_stylesheet_control_boundaries_meet_3_to_1(
    theme_rules: tuple[str, Rules], label: str, fg: tuple[str, str], bg: tuple[str, str]
) -> None:
    """WCAG 2.1 SC 1.4.11: what identifies a control or its state is >= 3:1."""
    theme, rules = theme_rules
    a, b = _colour(rules, *fg), _colour(rules, *bg)
    ratio = _contrast(a, b)
    assert ratio >= 3.0, f"{theme}: {label} {a} vs {b} is {ratio:.2f}:1"


# focus selector -> (fill it is drawn against, surround it is drawn against). The
# primary button's ring grows into the margin outside its teal fill, so it is judged
# against the surround alone (no colour is 3:1 against both the fill and the panel).
_FOCUS_RULES: list[tuple[str, tuple[str, str] | None, tuple[str, str]]] = [
    (
        "QPushButton:focus",
        ("QPushButton", "background-color"),
        ("QWidget", "background-color"),
    ),
    (
        "QToolButton:focus",
        ("QPushButton", "background-color"),
        ("QMainWindow", "background-color"),
    ),
    ('QPushButton[primary="true"]:focus', None, ("QWidget", "background-color")),
    (
        "QLineEdit:focus",
        ("QLineEdit", "background-color"),
        ("QWidget", "background-color"),
    ),
    (
        "QComboBox:focus",
        ("QLineEdit", "background-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "QSpinBox:focus",
        ("QLineEdit", "background-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "QDoubleSpinBox:focus",
        ("QLineEdit", "background-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "QCheckBox:focus",
        ("QWidget", "background-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "QTableView:focus",
        ("QTableView", "background-color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "QListWidget:focus",
        ("QTableView", "background-color"),
        ("QMainWindow", "background-color"),
    ),
    (
        "QScrollArea:focus",
        ("QScrollArea", "background-color"),
        ("QGroupBox", "background-color"),
    ),
    (
        "QGraphicsView:focus",
        ("QGraphicsView", "border-color"),
        ("QWidget", "background-color"),
    ),
    (
        "QTabBar::tab:focus",
        ("QTabBar::tab", "background-color"),
        ("QMainWindow", "background-color"),
    ),
]


@pytest.mark.parametrize(
    ("selector", "inner", "outer"), _FOCUS_RULES, ids=[r[0] for r in _FOCUS_RULES]
)
def test_every_focusable_control_has_a_focus_ring(
    theme_rules: tuple[str, Rules],
    selector: str,
    inner: tuple[str, str] | None,
    outer: tuple[str, str],
) -> None:
    """WCAG 2.4.7: a >= 2 px ring that stands out 3:1 from what it is drawn next to."""
    theme, rules = theme_rules
    assert selector in rules, f"{theme}: no focus style for {selector}"
    ring = _colour(rules, selector, "border-color")
    assert _border_width(rules, selector) >= 2, f"{theme}: {selector} ring under 2 px"
    for neighbour in (outer,) if inner is None else (inner, outer):
        other = _colour(rules, *neighbour)
        ratio = _contrast(ring, other)
        assert ratio >= 3.0, (
            f"{theme}: {selector} ring {ring} vs {neighbour} {other} is {ratio:.2f}:1"
        )


def test_stylesheet_images_exist_and_contrast(theme_rules: tuple[str, Rules]) -> None:
    """Arrows, ticks and grips are theme-coloured SVGs at >= 3:1 on what they sit on."""
    theme, rules = theme_rules
    # (sub-control drawing the image, selector + property of the colour under it)
    expected = {
        "QComboBox::down-arrow": ("QLineEdit", "background-color"),
        "QSpinBox::up-arrow": ("QSpinBox::up-button", "background-color"),
        "QSpinBox::down-arrow": ("QSpinBox::down-button", "background-color"),
        "QDoubleSpinBox::up-arrow": ("QSpinBox::up-button", "background-color"),
        "QDoubleSpinBox::down-arrow": ("QSpinBox::down-button", "background-color"),
        "QCheckBox::indicator:checked": (
            "QCheckBox::indicator:checked",
            "background-color",
        ),
        "QSplitter::handle:horizontal": ("QSplitter::handle", "background-color"),
        "QSplitter::handle:vertical": ("QSplitter::handle", "background-color"),
        "QMainWindow::separator:horizontal": (
            "QMainWindow::separator",
            "background-color",
        ),
        "QMainWindow::separator:vertical": (
            "QMainWindow::separator",
            "background-color",
        ),
    }
    for selector, under in expected.items():
        decls = rules.get(selector, {})
        found = re.search(r"url\(cuperiodtheme:([\w.-]+)\)", decls.get("image", ""))
        assert found, f"{theme}: {selector} draws no image"
        svg = RESOURCES / found.group(1)
        assert svg.is_file(), f"{theme}: {selector} -> missing {svg.name}"
        colours = {c.lower() for c in _HEX.findall(svg.read_text(encoding="utf-8"))}
        assert colours, svg.name
        back = _colour(rules, *under)
        for colour in colours:
            ratio = _contrast(colour, back)
            assert ratio >= 3.0, (
                f"{theme}: {svg.name} {colour} on {back} is {ratio:.2f}:1"
            )
    # every url() in the sheet resolves (a typo would silently draw nothing)
    for name in re.findall(r"url\(cuperiodtheme:([\w.-]+)\)", th.load_qss(theme)):  # type: ignore[arg-type]
        assert (RESOURCES / name).is_file(), name


# -- palette (QPalette + pyqtgraph colours) --------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
def test_qpalette_text_pairs(theme: th.ThemeName, qapp: QtWidgets.QApplication) -> None:
    del qapp
    qp = th.qpalette(th.palette(theme))
    role = QtGui.QPalette.ColorRole

    def hx(r: QtGui.QPalette.ColorRole) -> str:
        return str(qp.color(r).name())

    for fg, bg in (
        (role.Text, role.Base),
        (role.WindowText, role.Window),
        (role.ButtonText, role.Button),
        (role.ToolTipText, role.ToolTipBase),
        (role.HighlightedText, role.Highlight),
        (role.PlaceholderText, role.Base),
    ):
        ratio = _contrast(hx(fg), hx(bg))
        assert ratio >= 4.5, f"{theme}: {fg.name} on {bg.name} is {ratio:.2f}:1"


@pytest.mark.parametrize("theme", THEMES)
def test_plot_palette_contrast(theme: th.ThemeName) -> None:
    pal = th.palette(theme)
    bg = pal.plot_bg
    text = {"axis/legend text": pal.plot_fg, "muted text": pal.muted}
    for label, colour in text.items():
        assert _contrast(colour, bg) >= 4.5, f"{theme}: {label} {colour}"
    # Non-text marks, including the alpha the views draw them at (spectrum_view /
    # spacing_panel): selection-band edges at 170, the comb line at 200.
    marks: dict[str, str | RGB] = {
        "spectrum curve": pal.curve,
        "residual spectrum": pal.residual,
        "peak marker": pal.peak,
        "best-peak marker": pal.best_peak,
        "selected-period ring": pal.accent,
        "selection band edge (alpha 170)": _over(pal.accent, bg, 170),
        "comb best-spacing line (alpha 200)": _over(pal.accent, bg, 200),
        "spectral window (alpha 200)": _over(pal.muted, bg, 200),
    }
    for i in range(len(pal.band_cycle)):
        marks[f"band {i}"] = str(pal.band_color(i).name())
    for label, colour in marks.items():
        ratio = _contrast(colour, bg)
        assert ratio >= 3.0, f"{theme}: {label} {colour} on {bg} is {ratio:.2f}:1"


@pytest.mark.parametrize("theme", THEMES)
def test_band_cycle_is_colour_blind_safe(theme: th.ThemeName) -> None:
    """No two bands collapse under protan/deutan/tritan vision (Rubin's six bands
    stay well apart); band symbols back the colours up beyond that."""
    cycle = [c.lower() for c in th.palette(theme).band_cycle]
    assert len(cycle) == 8 and len(set(cycle)) == 8
    red_green = ("normal", "protanopia", "deuteranopia")
    worst_rg = min(
        _cvd_distance(a, b, red_green) for a, b in itertools.combinations(cycle, 2)
    )
    worst_tri = min(
        _cvd_distance(a, b, ("tritanopia",))
        for a, b in itertools.combinations(cycle, 2)
    )
    worst_six = min(
        _cvd_distance(a, b, (*red_green, "tritanopia"))
        for a, b in itertools.combinations(cycle[:6], 2)
    )
    # The cycle at HEAD had a reddish-purple / grey pair at dE00 6.2 under deuteranopia.
    assert worst_rg >= 8.0, f"{theme}: worst red-green pair dE00 {worst_rg:.1f}"
    assert worst_tri >= 7.0, f"{theme}: worst tritan pair dE00 {worst_tri:.1f}"
    assert worst_six >= 10.0, f"{theme}: worst pair of the first six {worst_six:.1f}"


@pytest.mark.parametrize("theme", THEMES)
def test_residual_trace_is_distinct_from_the_data_trace(theme: th.ThemeName) -> None:
    pal = th.palette(theme)
    kinds = ("normal", *_CVD)
    # at HEAD the residual was the accent: dE00 7.8 from the data under deuteranopia
    assert _cvd_distance(pal.residual, pal.curve, kinds) >= 20.0
    assert (
        _cvd_distance(pal.residual, _hex(_over(pal.muted, pal.plot_bg, 200)), kinds)
        >= 10.0
    )


def _hex(c: RGB) -> str:
    r, g, b = (round(v) for v in c)
    return f"#{r:02x}{g:02x}{b:02x}"


def test_band_symbols_back_up_the_colours() -> None:
    from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols

    for theme in THEMES:
        pal = th.palette(theme)  # type: ignore[arg-type]
        symbols = [pal.band_symbol(i) for i in range(len(pal.band_cycle))]
        assert len(set(symbols)) == len(symbols) == 8
        assert all(s in Symbols for s in symbols)
        assert pal.band_symbol(0) == "o"  # a single-band curve keeps its circles
        assert pal.band_symbol(8) == pal.band_symbol(0)  # cycles like the colours


# -- rendered checks (offscreen Qt) ----------------------------------------------------


@pytest.fixture
def apply(qapp: QtWidgets.QApplication) -> Iterator[Callable[..., th.ThemePalette]]:
    """``apply(theme, **kw)`` themes the app; the previous look is restored after."""
    sheet, palette_, font = qapp.styleSheet(), qapp.palette(), qapp.font()

    def _apply(theme: th.ThemeName, **kw: float) -> th.ThemePalette:
        return th.apply_theme(qapp, theme, **kw)

    yield _apply
    th.apply_theme(qapp, "dark", font_scale=1.0)
    qapp.setFont(font)
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette_)


def _image(widget: QtWidgets.QWidget) -> np.ndarray:
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    return buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def _lum_array(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def _ratio_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = _lum_array(a), _lum_array(b)
    return (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)


def _settle(qtbot: QtBot) -> None:
    QtWidgets.QApplication.processEvents()
    qtbot.wait(10)
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("theme", THEMES)
def test_combo_and_spin_boxes_draw_arrows_and_checkboxes_a_tick(
    theme: th.ThemeName, apply: Callable[..., th.ThemePalette], qtbot: QtBot
) -> None:
    """UX-01 / A11Y-09: the arrow areas and a checked box carry a glyph at >= 3:1."""
    apply(theme)
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    combo = QtWidgets.QComboBox()
    combo.addItems(["GLS", "BLS"])
    spin = QtWidgets.QSpinBox()
    spin.setValue(50)  # mid-range: at a limit that arrow is deliberately dimmed (:off)
    checked = QtWidgets.QCheckBox()
    checked.setChecked(True)
    for w in (combo, spin, checked):
        lay.addWidget(w)
    qtbot.addWidget(host)
    host.resize(240, 140)
    host.show()
    qtbot.waitExposed(host)
    _settle(qtbot)

    # Pixels at >= 3:1 from the face colour inside each arrow / indicator rect. The
    # rect also catches a few border pixels: without the images the counts are
    # combo 6-8, spin 0-1, tick 0; with them 20, 15 and 38.
    def glyph_pixels(widget: QtWidgets.QWidget, rect: QtCore.QRect) -> int:
        img = _image(widget)[
            rect.top() : rect.bottom() + 1, rect.left() : rect.right() + 1
        ]
        flat = img.reshape(-1, 3)
        colours, counts = np.unique(flat, axis=0, return_counts=True)
        back = colours[np.argmax(counts)]  # the dominant colour is the button face
        return int((_ratio_array(img, np.broadcast_to(back, img.shape)) >= 3.0).sum())

    style = combo.style()
    opt = QtWidgets.QStyleOptionComboBox()
    combo.initStyleOption(opt)
    arrow = style.subControlRect(
        QtWidgets.QStyle.ComplexControl.CC_ComboBox,
        opt,
        QtWidgets.QStyle.SubControl.SC_ComboBoxArrow,
        combo,
    )
    assert glyph_pixels(combo, arrow) >= 14, f"{theme}: no combo arrow"

    sopt = QtWidgets.QStyleOptionSpinBox()
    spin.initStyleOption(sopt)  # type: ignore[attr-defined]
    for sub in (
        QtWidgets.QStyle.SubControl.SC_SpinBoxUp,
        QtWidgets.QStyle.SubControl.SC_SpinBoxDown,
    ):
        rect = spin.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_SpinBox, sopt, sub, spin
        )
        assert glyph_pixels(spin, rect) >= 10, f"{theme}: no spin arrow ({sub})"

    copt = QtWidgets.QStyleOptionButton()
    checked.initStyleOption(copt)  # type: ignore[attr-defined]
    box = checked.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, copt, checked
    )
    assert glyph_pixels(checked, box.adjusted(2, 2, -2, -2)) >= 20, f"{theme}: no tick"


def _focus_probe(
    host: QtWidgets.QWidget,
    target: QtWidgets.QWidget,
    qtbot: QtBot,
    area: QtCore.QRect | None = None,
) -> tuple[int, int, int]:
    """Pixels around ``target`` (or ``area`` of it) that change when it takes keyboard
    focus: (changed, changed by >= 3:1, perimeter of the area)."""
    focused = QtWidgets.QApplication.focusWidget()
    if focused is not None:
        focused.clearFocus()
    _settle(qtbot)
    before = _image(host)
    host.setAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange, True)
    target.setFocus(Qt.FocusReason.TabFocusReason)
    _settle(qtbot)
    assert target.hasFocus()
    after = _image(host)
    rect = area if area is not None else target.rect()
    top_left = target.mapTo(host, rect.topLeft())
    x0, y0 = max(0, top_left.x() - 3), max(0, top_left.y() - 3)
    x1, y1 = top_left.x() + rect.width() + 3, top_left.y() + rect.height() + 3
    a, b = before[y0:y1, x0:x1], after[y0:y1, x0:x1]
    changed = np.any(a != b, axis=-1)
    strong = changed & (_ratio_array(a, b) >= 3.0)
    return int(changed.sum()), int(strong.sum()), 2 * (rect.width() + rect.height())


@pytest.mark.parametrize("theme", THEMES)
def test_keyboard_focus_is_visible_on_every_control(
    theme: th.ThemeName, apply: Callable[..., th.ThemePalette], qtbot: QtBot
) -> None:
    """A11Y-02: focus draws a ring of >= 2 px around the whole control, at least one
    pixel of it changing by >= 3:1 (at HEAD buttons, checkboxes, tables, the scroll
    area and plots changed no pixel at all)."""
    pal = apply(theme)
    host = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(host)
    button = QtWidgets.QPushButton("Reset")
    primary = QtWidgets.QPushButton("Compute periodogram")
    primary.setProperty("primary", True)
    tool = QtWidgets.QToolButton()
    tool.setText("Export")
    tool.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    check = QtWidgets.QCheckBox("log x")
    bare_check = QtWidgets.QCheckBox()
    combo = QtWidgets.QComboBox()
    combo.addItems(["GLS"])
    spin = QtWidgets.QSpinBox()
    edit = QtWidgets.QLineEdit("0.5")
    table = QtWidgets.QTableWidget(3, 2)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(QtWidgets.QLabel("\n".join("option" for _ in range(30))))
    scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    plot = pg.PlotWidget()
    th.apply_plot_theme(plot, pal)
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(QtWidgets.QLabel("peaks"), "Peaks")
    tabs.addTab(QtWidgets.QLabel("frequencies"), "Frequencies")
    controls: list[QtWidgets.QWidget] = [
        button,
        primary,
        tool,
        check,
        bare_check,
        combo,
        spin,
        edit,
        table,
        scroll,
        plot,
    ]
    for i, w in enumerate([*controls, tabs]):
        grid.addWidget(w, i // 3, i % 3)
    qtbot.addWidget(host)
    host.resize(720, 520)
    host.show()
    qtbot.waitExposed(host)
    host.activateWindow()
    _settle(qtbot)
    failures = []
    for w in controls:
        changed, strong, perimeter = _focus_probe(host, w, qtbot)
        name = type(w).__name__ + (" (primary)" if w is primary else "")
        if changed < 1.6 * perimeter or strong < 0.8 * perimeter:
            failures.append(
                f"{name}: changed {changed}, >=3:1 {strong}, perim {perimeter}"
            )
    # The current tab: its top edge already wears the accent selection marker, so
    # focus tints the face and adds the accent side rails (the >= 3:1 change).
    bar = tabs.tabBar()
    tab = bar.tabRect(0)
    changed, strong, perimeter = _focus_probe(host, bar, qtbot, tab)
    if changed < 1.6 * perimeter or strong < 3 * tab.height():
        failures.append(f"QTabBar: changed {changed}, >=3:1 {strong}, tab {tab}")
    assert not failures, f"{theme}: " + "; ".join(failures)


@pytest.mark.parametrize("theme", THEMES)
def test_current_table_cell_is_framed(
    theme: th.ThemeName, apply: Callable[..., th.ThemePalette], qtbot: QtBot
) -> None:
    """With 'outline: none' the current cell had no frame, so arrowing through a
    selected row showed nothing; it now carries a >= 3:1 frame."""
    apply(theme)
    table = QtWidgets.QTableWidget(3, 3)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    qtbot.addWidget(table)
    table.resize(360, 200)
    table.show()
    qtbot.waitExposed(table)
    table.activateWindow()
    table.setFocus(Qt.FocusReason.TabFocusReason)
    table.setCurrentCell(1, 0)
    _settle(qtbot)
    before = _image(table.viewport())
    table.setCurrentCell(1, 1)  # same row stays selected, only the current cell moves
    _settle(qtbot)
    after = _image(table.viewport())
    changed = np.any(before != after, axis=-1) & (_ratio_array(before, after) >= 3.0)
    assert changed.sum() >= 40, f"{theme}: current cell unframed ({changed.sum()} px)"


@pytest.mark.parametrize("theme", THEMES)
def test_pointer_targets_are_large_enough(
    theme: th.ThemeName, apply: Callable[..., th.ThemePalette], qtbot: QtBot
) -> None:
    """A11Y-16 (WCAG 2.5.8): check boxes 24 px tall, spin buttons 24 px wide,
    splitter handles 8 px and scroll bars 16 px (HEAD: 18, 16, 4 and 12), and the dock
    float / close buttons never shrink below the base style's 18x14 with their icons
    drawn (a sheet rule on them once collapsed them to 18x8 and clipped the icons)."""
    apply(theme)
    win = QtWidgets.QMainWindow()
    dock = QtWidgets.QDockWidget("Peaks", win)
    table = QtWidgets.QTableWidget(80, 3)
    dock.setWidget(table)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    central = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(central)
    spin, check = QtWidgets.QSpinBox(), QtWidgets.QCheckBox()
    split = QtWidgets.QSplitter()
    split.addWidget(QtWidgets.QLabel("a"))
    split.addWidget(QtWidgets.QLabel("b"))
    for w in (spin, check, split):
        lay.addWidget(w)
    win.setCentralWidget(central)
    qtbot.addWidget(win)
    win.resize(700, 400)
    win.show()
    qtbot.waitExposed(win)
    _settle(qtbot)

    sopt = QtWidgets.QStyleOptionSpinBox()
    spin.initStyleOption(sopt)  # type: ignore[attr-defined]
    up = spin.style().subControlRect(
        QtWidgets.QStyle.ComplexControl.CC_SpinBox,
        sopt,
        QtWidgets.QStyle.SubControl.SC_SpinBoxUp,
        spin,
    )
    assert up.width() >= 24, f"{theme}: spin button {up.width()} px wide"
    assert check.height() >= 24, f"{theme}: check box {check.height()} px tall"
    assert split.handleWidth() >= 8, f"{theme}: splitter handle {split.handleWidth()}"
    assert table.verticalScrollBar().width() >= 16, f"{theme}: scroll bar"
    buttons = [
        b
        for b in dock.findChildren(QtWidgets.QAbstractButton)
        if b.objectName() in ("qt_dockwidget_closebutton", "qt_dockwidget_floatbutton")
    ]
    assert len(buttons) == 2
    for b in buttons:
        assert b.width() >= 18 and b.height() >= 14, f"{theme}: {b.objectName()}"
        img = _image(b)
        flat = img.reshape(-1, 3)
        colours, counts = np.unique(flat, axis=0, return_counts=True)
        back = np.broadcast_to(colours[np.argmax(counts)], img.shape)
        assert (_ratio_array(img, back) >= 1.5).sum() >= 10, f"{theme}: no icon"


# -- UI zoom (A11Y-11) -----------------------------------------------------------------


def test_stylesheet_scales_every_length() -> None:
    raw = th.load_qss("dark")
    assert th.stylesheet("dark") == raw  # 1.0 is the designed sheet, verbatim
    big = th.stylesheet("dark", 1.5)
    assert "font-size: 20px" in big  # 13 px body text
    assert "font-size: 22px" in big or "font-size: 23px" in big  # 15 px heading
    lengths = [int(v) for v in re.findall(r"(\d+)px", raw)]
    scaled = [int(v) for v in re.findall(r"(\d+)px", big)]
    assert len(lengths) == len(scaled)
    assert all(
        (s == 0) if v == 0 else s >= max(1, math.floor(v * 1.5))
        for v, s in zip(lengths, scaled, strict=True)
    )
    assert "top: -2px" in big  # offsets scale too, keeping their sign
    assert "#0b0f14" in big  # colours untouched


def test_step_font_scale_walks_the_zoom_steps() -> None:
    assert th.step_font_scale(1.0, 1) > 1.0
    assert th.step_font_scale(1.0, -1) < 1.0
    assert th.step_font_scale(th.FONT_SCALE_STEPS[-1], 1) == th.FONT_SCALE_STEPS[-1]
    assert th.step_font_scale(th.FONT_SCALE_STEPS[0], -1) == th.FONT_SCALE_STEPS[0]
    assert (
        th.step_font_scale(1.13, 1)
        == th.FONT_SCALE_STEPS[th.FONT_SCALE_STEPS.index(1.1) + 1]
    )
    assert th.step_font_scale(1.25, 0) == 1.25


def _label_pt(style: dict[str, str]) -> float:
    return float(style["font-size"].removesuffix("pt"))


def test_zoom_enlarges_widget_and_plot_text(
    apply: Callable[..., th.ThemePalette], qtbot: QtBot
) -> None:
    """A11Y-11: one zoom factor scales widget text (headings keep their ratio) and the
    plot tick and axis-label fonts. Sizes are the requested ones (``font()``), which
    the sheet sets; offscreen runs may have no font files to render them with."""
    pal = apply("dark", font_scale=1.0)
    label = QtWidgets.QLabel("text")
    heading = QtWidgets.QLabel("Controls")
    heading.setObjectName("heading")
    plot = pg.PlotWidget()
    plot.setLabel("left", "Power", **th.plot_label_style(pal))
    for w in (label, heading, plot):
        qtbot.addWidget(w)
        w.ensurePolished()
    th.apply_plot_theme(plot, pal)
    base, base_heading = label.font().pixelSize(), heading.font().pixelSize()
    base_tick = plot.getPlotItem().getAxis("left").style["tickFont"].pointSizeF()
    base_label = _label_pt(th.plot_label_style(pal))
    assert base >= th.BASE_FONT_PX and base_heading > base

    pal = apply("dark", font_scale=1.5)
    assert th.font_scale() == 1.5
    for w in (label, heading):
        w.setStyleSheet("")  # re-polish under the new application sheet
        w.ensurePolished()
    th.apply_plot_theme(plot, pal)
    assert label.font().pixelSize() == pytest.approx(base * 1.5, abs=1)
    assert heading.font().pixelSize() == pytest.approx(base_heading * 1.5, abs=1)
    tick = plot.getPlotItem().getAxis("left").style["tickFont"].pointSizeF()
    assert tick == pytest.approx(base_tick * 1.5, rel=0.02)
    assert _label_pt(th.plot_label_style(pal)) == pytest.approx(base_label * 1.5)
    assert (
        th.plot_label_style(pal)["font-size"]
        in plot.getPlotItem().getAxis("left").labelString()
    )  # apply_plot_theme re-applied the label in the zoomed style

    apply("dark", font_scale=1.0)  # and back
    label.setStyleSheet("")
    label.ensurePolished()
    assert label.font().pixelSize() == base


def test_a_larger_application_font_is_honoured(
    apply: Callable[..., th.ThemePalette], qapp: QtWidgets.QApplication, qtbot: QtBot
) -> None:
    """A11Y-11: a larger application font used to change nothing (QSS forced 13 px)."""
    big = QtGui.QFont(qapp.font())
    big.setPixelSize(19)
    qapp.setFont(big)
    apply("light", font_scale=1.0)
    label = QtWidgets.QLabel("text")
    heading = QtWidgets.QLabel("Controls")
    heading.setObjectName("heading")
    for w in (label, heading):
        qtbot.addWidget(w)
        w.ensurePolished()
    assert label.font().pixelSize() == 19
    assert heading.font().pixelSize() > 19
    assert th.plot_label_style(th.palette("light"))["font-size"] != "12pt"


def test_font_scale_rejects_nonsense(apply: Callable[..., th.ThemePalette]) -> None:
    with pytest.raises(ValueError, match="font_scale"):
        apply("dark", font_scale=0.0)
    with pytest.raises(ValueError, match="font_scale"):
        apply("dark", font_scale=float("nan"))
    apply("dark", font_scale=50.0)  # clamped, not refused
    assert th.font_scale() == th.FONT_SCALE_STEPS[-1]


# -- plot axes (UX-16) -----------------------------------------------------------------


def test_axes_never_rescale_by_an_si_prefix(qtbot: QtBot) -> None:
    """A GLS power of 0.73 was drawn as '731' under 'GLS power (x0.001)'."""
    pal = th.palette("dark")
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.setLabel("left", "GLS power", **th.plot_label_style(pal))
    plot.setLabel("bottom", "Frequency (cycles/day)", **th.plot_label_style(pal))
    th.apply_plot_theme(plot, pal)
    plot.plot([0.01, 0.02, 0.03], [0.1, 0.731, 0.2])
    plot.getPlotItem().vb.autoRange()
    QtWidgets.QApplication.processEvents()
    for side in ("left", "bottom"):
        axis = plot.getPlotItem().getAxis(side)
        assert axis.autoSIPrefixScale == 1.0, side
        assert "(x" not in axis.labelString(), side
    # a theme switch with data on screen must not pick a stale x1000 scale up, and a
    # view relabelling afterwards must not turn it back on
    th.apply_plot_theme(plot, th.palette("light"))
    plot.setLabel("left", "Amplitude", **th.plot_label_style(pal))
    plot.getPlotItem().vb.setYRange(0.0, 0.0004)
    QtWidgets.QApplication.processEvents()
    axis = plot.getPlotItem().getAxis("left")
    assert axis.autoSIPrefixScale == 1.0
    assert "(x" not in axis.labelString()
    assert axis.tickStrings([0.0002], axis.autoSIPrefixScale * axis.scale, 0.0001) == [
        "0.0002"
    ]


def test_si_prefix_is_reset_without_the_enable_ranges_hook(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pyqtgraph < 0.13.4 (allowed by the >=0.13 pin) has no enable-ranges hook to pin
    the scale with, so a scale computed earlier must still be dropped by hand. The hook
    is stubbed out here (pyqtgraph 0.14 calls it itself, so it cannot be deleted)."""
    from pyqtgraph.graphicsItems.AxisItem import AxisItem

    pal = th.palette("dark")
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.setLabel("left", "Amplitude", **th.plot_label_style(pal))
    plot.getPlotItem().vb.setYRange(0.0, 0.0004, padding=0)
    QtWidgets.QApplication.processEvents()
    axis = plot.getPlotItem().getAxis("left")
    assert axis.autoSIPrefixScale != 1.0  # pyqtgraph's default: a x1000 scale
    monkeypatch.setattr(AxisItem, "setSIPrefixEnableRanges", lambda *_a, **_k: None)
    th.apply_plot_theme(plot, pal)
    assert axis.autoSIPrefixScale == 1.0
    assert "(x" not in axis.labelString()
