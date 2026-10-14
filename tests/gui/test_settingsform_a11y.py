"""Options-form accessibility leftovers (A11Y-04 names, A11Y-18 keyboard help)."""

from __future__ import annotations

from pytestqt.qtbot import QtBot

from cuperiod.core.config import GLSSettings, PreWhitenSettings
from cuperiod.gui.qt import QtWidgets
from cuperiod.gui.settingsform import (
    HELP_PLACEHOLDER,
    PydanticSettingsForm,
    _MultiChoice,
    _OptionalSpin,
)


def test_section_toggles_carry_keyboard_help(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    toggles = form.findChildren(QtWidgets.QToolButton, "section_toggle")
    assert toggles
    for toggle in toggles:
        assert toggle.accessibleName().endswith(" options")
        assert toggle.whatsThis() and toggle.statusTip() and toggle.toolTip()


def test_groups_are_named_and_criterion_notes_reach_both_halves(
    qtbot: QtBot,
) -> None:
    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    amplitude = form.field_widget("min_amplitude")
    assert isinstance(amplitude, _OptionalSpin)
    assert amplitude.accessibleName()
    note = "stop criterion 'amplitude'"
    assert note in amplitude.whatsThis()
    assert note in amplitude._auto.whatsThis() and note in amplitude._spin.statusTip()
    criteria = form.field_widget("stop_criteria")
    assert isinstance(criteria, _MultiChoice)
    assert criteria.accessibleName() and criteria.whatsThis()


def test_the_help_line_follows_the_focus(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    form.show()
    qtbot.waitExposed(form)
    assert form.help_text == HELP_PLACEHOLDER
    help_label = form.findChild(QtWidgets.QLabel, "field_help")
    assert help_label is not None and help_label.accessibleName()
    spin = form.field_widget("samples_per_peak")
    assert spin is not None
    form.activateWindow()
    spin.setFocus()
    qtbot.waitUntil(lambda: form.help_text != HELP_PLACEHOLDER, timeout=2_000)
    assert form.help_text == spin.whatsThis()
    # The line edit inside a spin box has no help of its own: its parent's is shown.
    form._on_focus_changed(None, spin.findChild(QtWidgets.QLineEdit))
    assert form.help_text == spin.whatsThis()
    # Focus outside the form leaves the last description in place.
    outside = QtWidgets.QLineEdit()
    qtbot.addWidget(outside)
    form._on_focus_changed(spin, outside)
    assert form.help_text == spin.whatsThis()
