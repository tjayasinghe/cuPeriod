"""Tests for the controls panel: sticky pickers, band tokens, hints, focus order."""

from __future__ import annotations

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.gui.meta import backend_options, multiband_method_names
from cuperiod.gui.models import COMBINED_BAND, STACKED_BAND
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.widgets.controls_panel import (
    COMPUTE_LABEL,
    PREWHITEN_RUN_LABEL,
    ControlsPanel,
    method_description,
)


def _panel(qtbot: QtBot) -> ControlsPanel:
    panel = ControlsPanel()
    qtbot.addWidget(panel)
    return panel


def _pick(combo: QtWidgets.QComboBox, text: str) -> None:
    """Select ``text`` the way a user does (signals live)."""
    index = combo.findText(text, Qt.MatchFlag.MatchExactly)
    assert index >= 0, f"{text!r} is not offered"
    combo.setCurrentIndex(index)


def _pick_band(panel: ControlsPanel, data: str) -> None:
    index = panel._band_combo.findData(data)
    assert index >= 0
    panel._band_combo.setCurrentIndex(index)


# -- GUI-B-3 / GUI-A-10 / UX-10: the pickers keep the user's choice ------------------


def test_backend_choice_survives_a_reload(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    _pick(panel._backend_combo, "cpu")
    # What MainWindow does on every curve load and batch step.
    panel.set_bands(None)
    panel.set_multiband(False)
    assert panel._backend_combo.currentText() == "cpu"


def test_backend_falls_back_to_auto_only_while_it_is_not_offered(
    qtbot: QtBot,
) -> None:
    gls_only = [b for b in backend_options("GLS") if b not in backend_options("PDM")]
    if not gls_only:
        pytest.skip("no backend here that GLS offers and PDM does not")
    panel = _panel(qtbot)
    _pick(panel._backend_combo, gls_only[0])
    _pick(panel._method_combo, "PDM")
    assert panel._backend_combo.currentText() == "auto"
    _pick(panel._method_combo, "GLS")  # offered again: the choice comes back
    assert panel._backend_combo.currentText() == gls_only[0]


def test_backend_choice_survives_a_method_switch(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    _pick(panel._backend_combo, "cpu")
    _pick(panel._method_combo, "PDM")
    assert panel._backend_combo.currentText() == "cpu"


def test_band_choice_survives_a_reload_with_the_same_bands(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    panel.set_bands(["g", "r"])
    panel.set_multiband(True)
    _pick_band(panel, "r")
    panel.set_bands(["g", "r"])
    panel.set_multiband(True)
    assert panel.current_band() == "r"
    # A source without that band falls back to the default for the list...
    panel.set_bands(["g", "i"])
    assert panel.current_band() == COMBINED_BAND
    # ...and a later source that has it again gets it back.
    panel.set_bands(["g", "r"])
    assert panel.current_band() == "r"


def test_a_prewhitening_band_does_not_leak_into_the_joint_default(
    qtbot: QtBot,
) -> None:
    panel = _panel(qtbot)
    panel.set_bands(["g", "r"])
    panel.set_multiband(True)
    assert panel.current_band() == COMBINED_BAND
    panel.set_analysis("prewhiten")
    assert panel.current_band() == "g"
    panel.set_analysis("periodogram")
    assert panel.current_band() == COMBINED_BAND


def test_an_unchanged_method_list_does_not_rebuild_the_form(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    form = panel._form
    panel.set_bands(None)
    panel.set_multiband(False)
    assert panel._form is form  # scroll position and edits in progress survive


# -- GUI-B-10: the all-band entries cannot collide with a real band ------------------


def test_synthetic_band_entries_differ_from_real_bands_of_the_same_name(
    qtbot: QtBot,
) -> None:
    panel = _panel(qtbot)
    panel.set_multiband(True)
    _pick(panel._method_combo, multiband_method_names()[0])
    panel.set_bands(["combined", "stacked", "g"])
    panel._band_combo.setCurrentIndex(0)
    assert panel.current_band() == COMBINED_BAND
    _pick(panel._band_combo, "combined")
    assert panel.current_band() == "combined"
    assert COMBINED_BAND != "combined"

    panel.set_analysis("prewhiten")
    _pick_band(panel, STACKED_BAND)
    assert panel.current_band() == STACKED_BAND
    _pick(panel._band_combo, "stacked")
    assert panel.current_band() == "stacked"


# -- GUI-B-9: the grid hint ignores non-finite times ----------------------------------


def test_grid_hint_ignores_nan_times(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0.0, 30.0, 300))
    time[::10] = np.nan
    panel.set_curve_time(time)
    hint = panel._grid_hint.text()
    assert hint and "nan" not in hint and "inf" not in hint
    finite = time[np.isfinite(time)]
    expected = 1.0 / float(finite.max() - finite.min())
    assert f"f: {expected:.3g}" in hint  # the auto minimum is 1 / finite baseline


def test_band_change_announces_the_new_band(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    panel.set_bands(["g", "r"])
    panel.set_multiband(True)
    with qtbot.waitSignal(panel.band_changed, timeout=1000) as blocker:
        _pick_band(panel, "r")
    assert blocker.args == ["r"]


# -- A11Y-03: the options sit between the pickers and Compute in Tab order ------------


def _tab_stops(panel: ControlsPanel) -> list[QtWidgets.QWidget]:
    stops: list[QtWidgets.QWidget] = []
    widget = panel._peaks_spin.nextInFocusChain()
    while widget is not panel._peaks_spin and len(stops) < 500:
        if (
            widget.focusPolicy() & Qt.FocusPolicy.TabFocus
            and widget.focusProxy() is None
            and widget.isVisibleTo(panel)
            and widget.isEnabled()
            and widget.window() is panel.window()
        ):
            stops.append(widget)
        widget = widget.nextInFocusChain()
    return stops


@pytest.mark.parametrize("analysis", ["periodogram", "prewhiten"])
def test_options_follow_the_pickers_in_tab_order(qtbot: QtBot, analysis: str) -> None:
    panel = _panel(qtbot)
    panel.set_enabled(True)
    panel.set_analysis(analysis)
    if analysis == "periodogram":
        _pick(panel._method_combo, "PDM")  # a rebuilt form, as after any switch
    stops = _tab_stops(panel)
    form = panel._form
    assert form is not None
    in_form = [form.isAncestorOf(w) for w in stops]
    assert any(in_form)
    first = in_form.index(True)
    last = len(in_form) - 1 - in_form[::-1].index(True)
    assert all(in_form[first : last + 1]), "options are not one contiguous block"
    assert first == 0, "the options do not come right after the pickers"
    assert stops[last + 1] is panel._compute_btn
    assert panel._scroll.focusPolicy() == Qt.FocusPolicy.NoFocus


# -- UX-13: the options form fits its panel -----------------------------------------


@pytest.mark.parametrize("analysis", ["periodogram", "prewhiten"])
def test_options_form_fits_a_narrow_panel(qtbot: QtBot, analysis: str) -> None:
    from cuperiod.gui.theme import apply_theme

    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    apply_theme(app, "dark")  # the QSS padding is what pushed it over
    panel = _panel(qtbot)
    panel.set_analysis(analysis)
    panel.resize(320, 900)  # the window's minimum width for the panel
    panel.show()
    qtbot.wait(10)
    assert panel._form is not None
    viewport = panel._scroll.viewport().width()
    assert panel._form.minimumSizeHint().width() <= viewport
    assert not panel._scroll.horizontalScrollBar().isVisible()


# -- UX-14: validation errors use the form's words ------------------------------------


def test_validation_error_names_fields_like_the_form(qtbot: QtBot) -> None:
    from pydantic import ValidationError

    from cuperiod.core.config import GLSSettings
    from cuperiod.gui.widgets.controls_panel import _first_error_message

    fields = list(GLSSettings.model_fields)
    with pytest.raises(ValidationError) as model_level:
        GLSSettings(minimum_frequency=5.0, maximum_frequency=1.0)
    text = _first_error_message(model_level.value, fields)
    assert text == "minimum frequency (5.0) must be < maximum frequency (1.0)"
    with pytest.raises(ValidationError) as field_level:
        GLSSettings(minimum_frequency=-1.0)
    text = _first_error_message(field_level.value, fields)
    assert text.startswith("minimum frequency: ")

    panel = _panel(qtbot)
    panel._show_error("<b>not markup</b>")
    assert panel._error.textFormat() == Qt.TextFormat.PlainText


# -- UX-22 / UX-25: labels say what things are -----------------------------------------


def test_disabled_compute_says_why(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    assert not panel.can_compute()
    assert "load a light curve" in panel._compute_btn.toolTip().lower()
    panel.set_enabled(True)
    assert "Ctrl+Enter" in panel._compute_btn.toolTip()
    panel.set_busy(True)
    assert "in progress" in panel._compute_btn.toolTip()


def test_methods_carry_a_description(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    combo = panel._method_combo
    for i in range(combo.count()):
        tip = combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
        assert tip and tip == method_description(combo.itemText(i))
    _pick(combo, "BLS")
    assert "Box Least Squares" in combo.toolTip()
    assert panel._note.text().startswith("Box Least Squares")


def test_grid_hint_uses_cycles_per_day(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    panel.set_curve_time(np.linspace(0.0, 30.0, 200))
    hint = panel._grid_hint.text()
    assert "c/d" in hint and " /d" not in hint


def test_action_label_follows_the_analysis(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    assert panel.action_label == COMPUTE_LABEL == panel._compute_btn.text()
    panel.set_analysis("prewhiten")
    assert panel.action_label == PREWHITEN_RUN_LABEL == panel._compute_btn.text()


def test_set_method_applies_an_offered_method_only(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    with qtbot.assertNotEmitted(panel.settings_edited):
        assert panel.set_method("BLS")
    assert panel.current_method() == "BLS"
    assert not panel.set_method("NOPE")
    assert panel.current_method() == "BLS"


def test_user_edits_are_announced_and_rebuilds_are_not(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    with qtbot.waitSignal(panel.settings_edited, timeout=1000):
        _pick(panel._method_combo, "PDM")
    with qtbot.waitSignal(panel.settings_edited, timeout=1000):
        _pick(panel._backend_combo, "cpu")
    with qtbot.assertNotEmitted(panel.settings_edited):
        panel.set_bands(["g", "r"])
        panel.set_multiband(True)
        panel.set_curve_time(np.linspace(0.0, 30.0, 200))


def test_hint_labels_are_plain_text(qtbot: QtBot) -> None:
    panel = _panel(qtbot)
    for label in (panel._note, panel._backend_hint, panel._grid_hint, panel._error):
        assert label.textFormat() == Qt.TextFormat.PlainText


# -- requests from B2-GUI-CORE (UX-11, GUI-A-18) -------------------------------------


def test_multiband_options_show_only_for_the_joint_fit(qtbot: QtBot) -> None:
    from cuperiod.gui.settingsform import MULTIBAND

    def section_hidden() -> bool:
        assert panel._form is not None
        section = panel._form._sections.get(MULTIBAND)
        assert section is not None, "GLS has no multiband section"
        return section.isHidden()

    panel = _panel(qtbot)
    panel.set_bands(["g", "r"])
    panel.set_multiband(True)
    assert not section_hidden()  # GLS on "combined (all bands)": the joint fit
    _pick_band(panel, "r")
    assert section_hidden()  # one band: the mb_* options do nothing
    _pick_band(panel, COMBINED_BAND)
    assert not section_hidden()
    panel.set_bands(None)
    panel.set_multiband(False)
    assert section_hidden()


def test_backend_hint_resolves_with_the_forms_settings(qtbot: QtBot) -> None:
    from cuperiod.core.config import PDMSettings

    if "torch" not in backend_options("PDM"):
        pytest.skip("torch is not installed here")
    panel = _panel(qtbot)
    panel._settings_cache["PDM"] = PDMSettings(device="cpu")  # as if typed in
    _pick(panel._method_combo, "PDM")
    _pick(panel._backend_combo, "torch")
    # Bare torch runs on the form's device, whatever GPU the machine has.
    assert panel._backend_hint.text().endswith("(CPU)")
