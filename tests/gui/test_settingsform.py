"""Tests for the pure ``field_to_spec`` descriptor over every settings model."""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from pytestqt.qtbot import QtBot

from cuperiod.core.config import (
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    PreWhitenSettings,
    StringLengthSettings,
    SuperSmootherSettings,
    TLSSettings,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.gui.qt import Qt, QtGui, QtWidgets
from cuperiod.gui.settingsform import (
    GUI_OWNED_FIELDS,
    FieldSpec,
    PydanticSettingsForm,
    _configure_int_spin,
    _GSpinBox,
    _OptionalSpin,
    _snap_step,
    field_to_spec,
)

ALL_SETTINGS: list[type[BaseSettings]] = [
    BLSSettings,
    CESettings,
    GLSSettings,
    MHAOVSettings,
    PDMSettings,
    StringLengthSettings,
    TLSSettings,
]

#: Every model the GUI builds an options form for.
FORM_MODELS: list[type[BaseSettings]] = [
    *ALL_SETTINGS,
    SuperSmootherSettings,
    PreWhitenSettings,
]

_KINDS = {
    "combo",
    "bool",
    "int",
    "float",
    "optional_int",
    "optional_float",
    "multi_choice",
    "float_list",
}


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_every_field_maps_or_skips(model_cls: type[BaseSettings]) -> None:
    for name, info in model_cls.model_fields.items():
        spec = field_to_spec(name, info)
        assert spec is None or spec.kind in _KINDS


def test_gls_specific_fields() -> None:
    fields = GLSSettings.model_fields
    backend = field_to_spec("backend", fields["backend"])
    assert backend is not None and backend.kind == "combo"
    assert "auto" in backend.choices

    fit_mean = field_to_spec("fit_mean", fields["fit_mean"])
    assert fit_mean is not None and fit_mean.kind == "bool"

    spp = field_to_spec("samples_per_peak", fields["samples_per_peak"])
    assert spp is not None and spp.kind == "int"
    assert spp.lo == 1.0

    min_freq = field_to_spec("minimum_frequency", fields["minimum_frequency"])
    assert min_freq is not None and min_freq.kind == "optional_float"
    assert min_freq.default is None


def test_literal_choices_match_get_args() -> None:
    field = GLSSettings.model_fields["fap_method"]
    spec = field_to_spec("fap_method", field)
    assert spec is not None
    assert set(spec.choices) == {str(a) for a in get_args(field.annotation)}


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_defaults_build_valid(model_cls: type[BaseSettings]) -> None:
    assert model_cls() is not None


def test_bad_bounds_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        GLSSettings(minimum_frequency=10.0, maximum_frequency=1.0)


# -- float spinbox display formatting (%g instead of fixed decimals) ---------------


def test_gspinbox_formats_compactly(qtbot: QtBot) -> None:
    spin = _GSpinBox()
    qtbot.addWidget(spin)
    spin.setDecimals(12)
    spin.setRange(-1.0e12, 1.0e12)
    spin.setValue(3.0)
    assert spin.textFromValue(spin.value()) == "3"
    spin.setValue(1e-9)
    assert spin.textFromValue(spin.value()) == "1e-09"
    assert isinstance(spin, QtWidgets.QDoubleSpinBox)


def test_gspinbox_valuefromtext_parses_scientific_and_plain() -> None:
    spin = _GSpinBox()
    assert spin.valueFromText("1e-9") == pytest.approx(1e-9)
    assert spin.valueFromText("3") == pytest.approx(3.0)
    assert spin.valueFromText("not-a-number") == 0.0  # never raises


def test_gspinbox_validate_never_blocks_partial_scientific_input() -> None:
    spin = _GSpinBox()
    for text in ("", "-", "1e-", "1e", "1.", "3", "1e-09"):
        state, _, _ = spin.validate(text, len(text))
        assert state != QtGui.QValidator.State.Invalid


def test_float_spin_uses_compact_g_formatting_for_gls_defaults(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    sep_spin = form._readers["peak_separation_rayleigh"].__self__  # type: ignore[attr-defined]
    assert isinstance(sep_spin, _GSpinBox)
    assert sep_spin.text() == "3"  # not "3.000000"

    eps_spin = form._readers["nufft_eps"].__self__  # type: ignore[attr-defined]
    assert isinstance(eps_spin, _GSpinBox)
    assert eps_spin.text() == "1e-09"  # not "0.000000001000"


@pytest.mark.parametrize("model_cls", ALL_SETTINGS)
def test_form_float_values_roundtrip_defaults(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    """Every default float field, read back through its (now %g-formatted) spin box,
    still equals the pydantic default — the display format change must not lose
    precision or corrupt the round-trip.
    """
    defaults = model_cls()
    form = PydanticSettingsForm(model_cls, initial=defaults)
    qtbot.addWidget(form)
    built = form.build()
    for name, info in model_cls.model_fields.items():
        spec = field_to_spec(name, info)
        if spec is None or spec.kind not in ("float", "optional_float"):
            continue
        got = getattr(built, name)
        want = getattr(defaults, name)
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-9)


# -- int spinbox fallback bounds -----------------------------------------------------


def test_configure_int_spin_negative_fallback_when_lo_none(qtbot: QtBot) -> None:
    spin = QtWidgets.QSpinBox()
    qtbot.addWidget(spin)
    spec = FieldSpec(name="x", kind="int", default=0, lo=None, hi=None)
    _configure_int_spin(spin, spec)
    assert spin.minimum() == -10_000_000
    assert spin.maximum() == 10_000_000


# -- GUI-A-9: tuple-typed fields get a widget, so stop_criteria is reachable --------


@pytest.mark.parametrize("model_cls", FORM_MODELS)
def test_every_field_maps_to_a_widget(model_cls: type[BaseSettings]) -> None:
    for name, info in model_cls.model_fields.items():
        assert field_to_spec(name, info) is not None, f"{model_cls.__name__}.{name}"


def test_stop_criteria_can_be_switched_on(qtbot: QtBot) -> None:
    from cuperiod.prewhiten import prewhiten

    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    assert form.build().stop_criteria == ("snr",)  # type: ignore[attr-defined]
    boxes = form.findChildren(QtWidgets.QCheckBox, "stop_criteria:amplitude")
    assert len(boxes) == 1
    boxes[0].setChecked(True)
    opt = form._optional["min_amplitude"]
    opt._auto.setChecked(False)
    opt._spin.setValue(50.0)  # type: ignore[attr-defined]
    settings = form.build()
    assert settings.stop_criteria == ("snr", "amplitude")  # type: ignore[attr-defined]
    assert settings.min_amplitude == 50.0  # type: ignore[attr-defined]

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 30.0, 400))
    y = 0.3 * np.sin(2 * np.pi * 1.7 * t) + rng.normal(0.0, 0.01, t.size)
    lc = LightCurve.from_arrays(t, y, np.full(t.size, 0.01))
    run = settings.model_copy(update={"backend": "finufft", "max_frequencies": 2})
    assert prewhiten(lc, settings=run).n_components == 0  # 50 mag floor honoured


def test_thresholds_of_unticked_criteria_are_greyed_out(qtbot: QtBot) -> None:
    # Only the ticked criteria are tested by the engine; an editable FAP threshold
    # with 'fap' unticked looked like a knob that should change the result.
    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    fap = form.field_widget("fap_threshold")
    snr = form.field_widget("snr_threshold")
    assert fap is not None and snr is not None
    assert snr.isEnabled() and not fap.isEnabled()
    assert "'fap'" in fap.toolTip()
    box = form.findChild(QtWidgets.QCheckBox, "stop_criteria:fap")
    assert box is not None
    box.setChecked(True)
    assert fap.isEnabled()
    snr_box = form.findChild(QtWidgets.QCheckBox, "stop_criteria:snr")
    assert snr_box is not None
    snr_box.setChecked(False)
    assert not snr.isEnabled()
    form.reset_to_defaults()
    assert snr.isEnabled() and not fap.isEnabled()


def test_primary_spans_round_trip_and_validate(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(SuperSmootherSettings)
    qtbot.addWidget(form)
    default = SuperSmootherSettings().primary_spans
    assert form.build().primary_spans == default  # type: ignore[attr-defined]
    edit = form.findChild(QtWidgets.QLineEdit, "primary_spans")
    assert edit is not None
    edit.setText("0.1, 0.4")
    assert form.build().primary_spans == (0.1, 0.4)  # type: ignore[attr-defined]
    edit.setText("0.1, nonsense")
    with pytest.raises(ValidationError):
        form.build()


# -- GUI-A-12: CUPERIOD_<METHOD>_<FIELD> overrides seed the form --------------------


def test_form_defaults_honour_environment_overrides(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_DEVICE", "cpu")
    monkeypatch.setenv("CUPERIOD_GLS_PRECISION", "float64")
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "7")
    monkeypatch.setenv("CUPERIOD_PREWHITEN_MAX_FREQUENCIES", "4")
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    built = form.build()
    env = GLSSettings()
    for name in ("device", "precision", "samples_per_peak", "nyquist_factor"):
        assert getattr(built, name) == getattr(env, name), name
    pw = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(pw)
    assert pw.build().max_frequencies == 4  # type: ignore[attr-defined]


def test_an_invalid_environment_value_falls_back_to_defaults(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "0")
    with pytest.raises(ValidationError):
        GLSSettings()
    form = PydanticSettingsForm(GLSSettings)  # must not raise
    qtbot.addWidget(form)
    assert form.value_of("samples_per_peak") == 10


# -- GUI-A-19 / UX-12: what the form shows is what runs ----------------------------


@pytest.mark.parametrize("model_cls", FORM_MODELS)
def test_form_shows_the_gui_oversampling(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    if "samples_per_peak" not in model_cls.model_fields:
        pytest.skip("no frequency oversampling field")
    form = PydanticSettingsForm(model_cls)
    qtbot.addWidget(form)
    assert form.value_of("samples_per_peak") == 10
    assert form.build().samples_per_peak == 10  # type: ignore[attr-defined]


def test_an_explicit_initial_oversampling_is_kept(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings, initial=GLSSettings(samples_per_peak=5))
    qtbot.addWidget(form)
    assert form.build().samples_per_peak == 5  # type: ignore[attr-defined]


def test_gui_owned_fields_are_not_in_the_form(qtbot: QtBot) -> None:
    assert {"backend", "n_peaks", "store_spectra", "downsample_points"} <= set(
        GUI_OWNED_FIELDS
    )
    for model_cls in FORM_MODELS:
        form = PydanticSettingsForm(model_cls)
        qtbot.addWidget(form)
        assert not set(form.values()) & set(GUI_OWNED_FIELDS), model_cls.__name__


# -- GUI-A-25: float fields display a value that round-trips ------------------------


@pytest.mark.parametrize(
    "value", [11.123456789, 1234567.5, 12345.678, 0.1234567, 524.875031, 3.0, 1e-9]
)
def test_gspinbox_text_round_trips(value: float) -> None:
    spin = _GSpinBox()
    assert spin.valueFromText(spin.textFromValue(value)) == value


def test_gspinbox_second_return_does_not_truncate(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    spin = form._readers["snr_threshold"].__self__  # type: ignore[attr-defined]
    assert isinstance(spin, _GSpinBox)
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin, "1234567.5")
    qtbot.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == 1234567.5
    qtbot.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == 1234567.5
    assert form.build().snr_threshold == 1234567.5  # type: ignore[attr-defined]


def test_arrow_steps_show_no_float_noise(qtbot: QtBot) -> None:
    # Holding values at full precision made 1.1 + 0.1 read 1.2000000000000002.
    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    spin = form._readers["snr_window"].__self__  # type: ignore[attr-defined]
    assert isinstance(spin, _GSpinBox)
    spin.setValue(1.1)
    spin.setSingleStep(0.1)
    for expected in ("1.2", "1.3", "1.4"):
        qtbot.keyClick(spin, Qt.Key.Key_Up)
        assert spin.text() == expected
    assert spin.value() == 1.4
    spin.setValue(1.23456789)  # a typed value keeps its digits when stepped
    spin.stepBy(1)
    assert spin.value() == 1.33456789
    # stepping back to zero lands on zero, not on the sum's leftover 5.6e-17
    assert _snap_step(0.1 + 0.2 - 0.3, 0.1) == 0.0


def test_small_float_thresholds_are_not_rounded_to_zero(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(PreWhitenSettings)
    qtbot.addWidget(form)
    spin = form._readers["fap_threshold"].__self__  # type: ignore[attr-defined]
    spin.setValue(1e-12)
    built = form.build()
    assert built.fap_threshold == pytest.approx(1e-12)  # type: ignore[attr-defined]


# -- UX-11: fields are grouped into labelled sections, inert ones hidden ------------


@pytest.mark.parametrize("model_cls", FORM_MODELS)
def test_every_field_is_reachable_in_a_section(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    form = PydanticSettingsForm(model_cls)
    qtbot.addWidget(form)
    expected = set(model_cls.model_fields) - set(GUI_OWNED_FIELDS)
    assert set(form.values()) == expected
    sections = form.section_names()
    assert sections[-1] == "Advanced"
    assert len(sections) == len(set(sections))
    for name in expected:
        assert form.section_of(name) in sections, name


def test_search_range_comes_first_and_advanced_is_collapsed(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    form.show()
    assert form.section_names()[0] == "Search range"
    assert form.section_of("maximum_frequency") == "Search range"
    assert form.section_of("device") == "Advanced"
    assert form.section_of("mb_model") == "Multiband"
    device = form.field_widget("device")
    assert device is not None and not device.isVisible()
    spp = form.field_widget("samples_per_peak")
    assert spp is not None and spp.isVisible()
    form.set_section_expanded("Advanced", True)
    assert device.isVisible()


def test_multiband_section_hides_for_single_band_curves(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    form.show()
    assert "Multiband" in form.section_names()
    form.set_multiband(False)
    assert "Multiband" not in form.section_names()
    assert form.build().mb_model == "offsets"  # type: ignore[attr-defined]
    form.set_multiband(True)
    assert "Multiband" in form.section_names()


#: Width of the options dock's viewport at the window's minimum panel width (320 px
#: less the scroll area's frame, margins and scrollbar under the bundled theme).
_DOCK_VIEWPORT = 264


@pytest.mark.parametrize("model_cls", FORM_MODELS)
def test_every_form_fits_the_options_dock(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    # The stop-criteria check boxes used to sit in one row, 274 px wide.
    from cuperiod.gui.theme import apply_theme

    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    apply_theme(app, "dark")
    form = PydanticSettingsForm(model_cls)
    qtbot.addWidget(form)
    form.show()
    assert form.minimumSizeHint().width() <= _DOCK_VIEWPORT


def test_auto_spin_with_a_zero_lower_bound_keeps_a_usable_step(qtbot: QtBot) -> None:
    # minimum_frequency is bounded below by 0; sizing the step from that bound made
    # each arrow press move the value by 0.001.
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    widget = form.field_widget("minimum_frequency")
    assert widget is not None
    spins = widget.findChildren(QtWidgets.QDoubleSpinBox)
    assert spins and spins[0].singleStep() >= 0.1


def test_reset_to_defaults_restores_gui_defaults(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings, initial=GLSSettings(samples_per_peak=3))
    qtbot.addWidget(form)
    changed: list[None] = []
    form.changed.connect(lambda: changed.append(None))
    form.reset_to_defaults()
    assert form.value_of("samples_per_peak") == 10
    assert changed


# -- A11Y-04 / A11Y-18: every field has an accessible name and keyboard help --------

_INPUTS = (
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QCheckBox,
    QtWidgets.QLineEdit,
)


@pytest.mark.parametrize("model_cls", FORM_MODELS)
def test_every_input_has_an_accessible_name_and_help(
    model_cls: type[BaseSettings], qtbot: QtBot
) -> None:
    form = PydanticSettingsForm(model_cls)
    qtbot.addWidget(form)
    names: list[str] = []
    for widget in form.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, _INPUTS):
            continue
        if isinstance(widget.parent(), QtWidgets.QAbstractSpinBox):
            continue  # the spin box's own line edit
        name = widget.accessibleName()
        assert name, f"{model_cls.__name__}: unnamed {type(widget).__name__}"
        names.append(name)
        assert widget.whatsThis() and widget.statusTip(), name
    assert len(names) == len(set(names)), "accessible names must be unique"
    labels = [
        label
        for label in form.findChildren(QtWidgets.QLabel)
        if label.objectName() == "field_label"
    ]
    assert labels and all(label.buddy() is not None for label in labels)


# -- A11Y-17: the effective 'auto' value is readable, not drawn disabled ------------


def test_auto_value_is_read_only_not_disabled(qtbot: QtBot) -> None:
    form = PydanticSettingsForm(GLSSettings)
    qtbot.addWidget(form)
    form.set_auto_value("maximum_frequency", 50.0)
    opt = form._optional["maximum_frequency"]
    assert isinstance(opt, _OptionalSpin)
    spin = opt._spin
    assert spin.isEnabled() and spin.isReadOnly()
    assert spin.property("auto") is True
    assert spin.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert math.isclose(spin.value(), 50.0)  # type: ignore[attr-defined]
    assert opt.value() is None
    opt._auto.setChecked(False)
    assert not spin.isReadOnly()
    assert spin.property("auto") is False
    assert opt.value() == pytest.approx(50.0)
