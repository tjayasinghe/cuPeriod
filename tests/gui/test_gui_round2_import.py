"""The import dialog's second release-gate round (GUI-1 .. GUI-4).

* A headerless table whose first row lacks a value (``nan``, ``inf``, an empty CSV
  cell) is data, not a header row read as data: Load stays available, the header
  box starts unticked, and every row loads (GUI-1, GUI-4).
* What the next import starts from is what the user *chose*: a "none" error or band
  column of a file that had none, or a domain read off a ``flux`` column name, is not
  carried over to the next file (GUI-2, GUI-3).

Each sequence passes the last choice through the same JSON the main window keeps in
QSettings (:func:`import_choice_to_json` / :func:`import_choice_from_json`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.loaders import (
    NO_COLUMN,
    ImportChoice,
    guess_columns,
    is_text_cell,
    preview_file,
)
from cuperiod.gui.main_window import import_choice_from_json, import_choice_to_json
from cuperiod.gui.qt import QtCore, QtWidgets
from cuperiod.gui.widgets.preview_dialog import PreviewDialog

JD = [2458000.1 + 0.37 * i for i in range(30)]


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _table(
    path: Path,
    header: str | None = None,
    sep: str = " ",
    first: tuple[str, ...] | None = None,
    columns: int = 3,
) -> Path:
    """30 rows of ``JD mag err``: the first ``columns`` of JD, mag, err, band."""
    rows = []
    for i, t in enumerate(JD):
        cells = [f"{t}", f"{14 + 0.01 * i:.3f}", "0.020", "gr"[i % 2]]
        rows.append(sep.join(cells[:columns]))
    if first is not None:
        rows[0] = sep.join(first)
    return _write(path, rows if header is None else [header, *rows])


def _dialog(qtbot: QtBot, path: Path, last: str | None = None) -> PreviewDialog:
    """The import dialog for ``path``, starting from a remembered choice's JSON."""
    preview = preview_file(path)
    assert preview is not None
    remembered = import_choice_from_json(last)
    dialog = PreviewDialog(
        path.name,
        preview,
        path=path,
        columns=None if remembered is None else remembered.columns,
        domain=None if remembered is None else remembered.domain,
        read=None if remembered is None else remembered.read,
    )
    qtbot.addWidget(dialog)
    return dialog


def _pick(combo: QtWidgets.QComboBox, data: object) -> None:
    index = combo.findData(data)
    assert index >= 0, data
    combo.setCurrentIndex(index)


def _pick_jd_mag_err(dialog: PreviewDialog) -> None:
    """The user picks col1, col2, col3 as the time, brightness and error."""
    for role, column in (("time", "col1"), ("value", "col2"), ("error", "col3")):
        _pick(dialog._combos[role], column)


def _load(dialog: PreviewDialog, path: Path) -> object:
    assert dialog.can_load(), dialog._problem.text()
    return dialog.choice().load(path)


def _saved(dialog: PreviewDialog) -> str:
    """What the main window keeps for the next import after Load."""
    return import_choice_to_json(dialog.choice())


def _remembered(saved: str) -> ImportChoice:
    choice = import_choice_from_json(saved)
    assert choice is not None
    return choice


# -- GUI-1: a missing value in the first row is no header -----------------------------
_MISSING_FIRST = [
    ("nan_err.dat", " ", (f"{JD[0]}", "14.000", "nan")),
    ("NaN_err.dat", " ", (f"{JD[0]}", "14.000", "NaN")),
    ("inf_err.dat", " ", (f"{JD[0]}", "14.000", "inf")),
    ("nan_mag.dat", " ", (f"{JD[0]}", "nan", "0.020")),
    ("empty_err.csv", ",", (f"{JD[0]}", "14.000", "")),
    ("nan_err.csv", ",", (f"{JD[0]}", "14.000", "nan")),
    ("empty_mag.csv", ",", (f"{JD[0]}", "", "0.020")),
]


@pytest.mark.parametrize(("name", "sep", "first"), _MISSING_FIRST)
def test_a_missing_first_value_does_not_disable_load(
    qtbot: QtBot, tmp_path: Path, name: str, sep: str, first: tuple[str, ...]
) -> None:
    """GUI-1: 'nan' or an empty cell counted as text, so a headerless table whose
    first row lacked an error or magnitude was taken for a header row read as data,
    and Load was disabled with advice that would have lost that row."""
    path = _table(tmp_path / name, sep=sep, first=first)
    dialog = _dialog(qtbot, path)
    if dialog._header.isChecked():
        dialog._header.setChecked(False)
    _pick_jd_mag_err(dialog)
    assert dialog.can_load(), dialog._problem.text()
    assert dialog._problem.isHidden()
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve)
    assert len(lc.time) == len(JD) and lc.time[0] == JD[0]


@pytest.mark.parametrize(("name", "sep", "first"), _MISSING_FIRST)
def test_a_missing_first_value_reads_headerless_as_detected(
    qtbot: QtBot, tmp_path: Path, name: str, sep: str, first: tuple[str, ...]
) -> None:
    """GUI-1/GUI-4 as opened: no header row, the JD mag err guess, every row."""
    path = _table(tmp_path / name, sep=sep, first=first)
    dialog = _dialog(qtbot, path)
    assert not dialog._header.isChecked()
    assert dialog._preview.n_rows == len(JD)
    chosen = dialog.column_map()
    assert (chosen.time, chosen.value, chosen.error) == ("col1", "col2", "col3")
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve) and lc.time[0] == JD[0]
    assert len(lc.time) == len(JD)


def test_a_header_row_read_as_data_is_still_caught(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The check GUI-1 narrowed still catches the case it is for, and a stray word
    in a data row's error column (IRAF's INDEF) is not taken for a header row."""
    path = _table(tmp_path / "header.dat", header="JD mag err")
    dialog = _dialog(qtbot, path)
    assert dialog._header.isChecked()
    dialog._header.setChecked(False)
    _pick_jd_mag_err(dialog)
    assert not dialog.can_load()
    assert "'JD'" in dialog._problem.text() and "'err'" in dialog._problem.text()
    indef = _table(tmp_path / "indef.dat", first=(f"{JD[0]}", "14.000", "INDEF"))
    dialog = _dialog(qtbot, indef)
    assert not dialog._header.isChecked()
    _pick(dialog._combos["error"], "col3")
    assert "column names" not in dialog._problem.text()


def test_the_guess_counts_a_column_with_a_missing_value_as_numeric(
    tmp_path: Path,
) -> None:
    """A 'nan' first magnitude made col2 non-numeric, so the guess took the error
    column for the brightness."""
    path = _table(tmp_path / "nan_mag.dat", first=(f"{JD[0]}", "nan", "0.020"))
    preview = preview_file(path)
    assert preview is not None
    assert guess_columns(preview) == {"time": "col1", "value": "col2", "error": "col3"}
    assert not is_text_cell("nan") and not is_text_cell("") and not is_text_cell("inf")
    assert is_text_cell("mag") and is_text_cell("INDEF")
    # A placeholder the loader cannot read as a number is no error column to guess.
    dashed = _table(tmp_path / "dash.dat", first=(f"{JD[0]}", "14.000", "--"))
    preview = preview_file(dashed)
    assert preview is not None
    assert "error" not in guess_columns(preview)


def test_a_remembered_ogle_choice_opens_a_table_with_a_nan_first_row(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """GUI-1 sequence E: an OGLE table, then one whose first error is nan."""
    ogle = _table(tmp_path / "ogle.dat")
    first = _dialog(qtbot, ogle)
    _pick_jd_mag_err(first)  # the user confirms the guess
    saved = _saved(first)
    nan_first = _table(tmp_path / "nanfirst.dat", first=(f"{JD[0]}", "14.000", "nan"))
    dialog = _dialog(qtbot, nan_first, saved)
    assert dialog.column_map().error == "col3"
    lc = _load(dialog, nan_first)
    assert isinstance(lc, LightCurve) and len(lc.time) == len(JD)


# -- GUI-4: a headerless CSV with an empty cell ---------------------------------------
@pytest.mark.parametrize("after_ogle", [False, True])
def test_a_headerless_csv_with_an_empty_first_cell_loads_every_row(
    qtbot: QtBot, tmp_path: Path, after_ogle: bool
) -> None:
    """GUI-4 sequences G and H: pandas named the empty cell 'Unnamed: 2', which
    kept the header box ticked; Load was offered and then failed (the loader calls
    that column col2), or the first data row became the column names."""
    saved = None
    if after_ogle:
        saved = _saved(_dialog(qtbot, _table(tmp_path / "ogle.dat")))
    path = _table(
        tmp_path / "emptyerr_nohdr.csv", sep=",", first=(f"{JD[0]}", "14.0", "")
    )
    dialog = _dialog(qtbot, path, saved)
    assert not dialog._header.isChecked()
    assert dialog._preview.columns == ["col1", "col2", "col3"]
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve) and len(lc.time) == len(JD)
    assert lc.time[0] == JD[0]


def test_a_headerless_csv_repeating_a_value_in_its_first_row(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Two equal first-row cells come back renamed (pandas '0.02.1', the loader
    '0.02_1'): still a number, so still a data row."""
    rows = [f"{t},{14 + 0.01 * i:.3f},0.020,0.020" for i, t in enumerate(JD)]
    path = _write(tmp_path / "two_errors.csv", rows)
    dialog = _dialog(qtbot, path)
    assert not dialog._header.isChecked()
    assert dialog._preview.n_rows == len(JD)
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve) and len(lc.time) == len(JD)


def test_a_csv_header_with_an_empty_name_previews_the_loaders_name(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The preview (pandas) and the load (astropy) named an empty header cell
    differently, so the column picked in the dialog was not found at load."""
    path = _table(tmp_path / "noname.csv", header="time,mag,", sep=",")
    dialog = _dialog(qtbot, path)
    assert dialog._header.isChecked()
    assert dialog._preview.columns == ["time", "mag", "col2"]
    _pick(dialog._combos["error"], "col2")
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve) and lc.error is not None
    assert len(lc.time) == len(JD)


# -- GUI-2: a remembered "none" never drops a detected column -------------------------
def test_a_multiband_file_after_a_single_band_one_keeps_its_bands(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """GUI-2 sequence J: the band and error columns of a multiband file were
    replaced by the last file's "none": it loaded pooled, without its errors."""
    first = _dialog(qtbot, _table(tmp_path / "mag.csv", "time,mag", ",", columns=2))
    saved = _saved(first)
    assert _remembered(saved) == ImportChoice(columns=ColumnMap())
    path = _table(tmp_path / "mb.csv", "time,mag,mag_err,band", ",", columns=4)
    dialog = _dialog(qtbot, path, saved)
    assert dialog._combos["band"].currentData() == "band"
    assert dialog._combos["error"].currentData() == "mag_err"
    assert dialog.column_map().band is None  # as detected: left to the loader
    lc = _load(dialog, path)
    assert isinstance(lc, MultiBandLightCurve) and lc.n_bands == 2
    assert all(band.error is not None for band in lc.bands.values())


def test_an_error_column_after_a_file_without_one_is_kept(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """GUI-2 sequence K: jd,V (no error) then time,mag,mag_err ran unweighted."""
    v = _write(
        tmp_path / "v.csv",
        ["jd,V", *(f"{t},{13.0 + i / 100}" for i, t in enumerate(JD))],
    )
    saved = _saved(_dialog(qtbot, v))
    path = _table(tmp_path / "me.csv", "time,mag,mag_err", ",")
    dialog = _dialog(qtbot, path, saved)
    assert dialog.column_map().error == "mag_err"
    lc = _load(dialog, path)
    assert isinstance(lc, LightCurve) and lc.error is not None


def test_none_chosen_by_the_user_carries_over_but_yields_to_detection(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A "none" the user picked is remembered: it overrides the next headerless
    table's guess, but not a file whose error column is detected."""
    ogle = _table(tmp_path / "ogle.dat")
    first = _dialog(qtbot, ogle)
    _pick(first._combos["error"], NO_COLUMN)
    saved = _saved(first)
    assert _remembered(saved).columns.error == NO_COLUMN
    again = _dialog(qtbot, _table(tmp_path / "ogle2.dat"), saved)
    assert again.column_map().error == NO_COLUMN
    saved = _saved(again)  # untouched: the user's "none" is kept for the next one
    detected = _dialog(
        qtbot, _table(tmp_path / "me.csv", "time,mag,mag_err", ","), saved
    )
    assert detected.column_map().error == "mag_err"
    # Through a file that did not use it, the user's choice is still remembered.
    third = _dialog(qtbot, _table(tmp_path / "ogle3.dat"), _saved(detected))
    assert third.column_map().error == NO_COLUMN


def test_columns_the_user_picked_carry_over_to_the_next_such_file(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A JD err mag table: the user's col3 / col2 apply to the next one, also after
    a header file in between (where they did not apply)."""
    rows = [f"{t} 0.020 {14 + 0.01 * i:.3f}" for i, t in enumerate(JD)]
    first = _dialog(qtbot, _write(tmp_path / "a.dat", rows))
    _pick(first._combos["value"], "col3")
    _pick(first._combos["error"], "col2")
    saved = _saved(first)
    header = _dialog(qtbot, _table(tmp_path / "me.csv", "time,mag,mag_err", ","), saved)
    assert header.column_map().value == "mag"
    later = _dialog(qtbot, _write(tmp_path / "b.dat", rows), _saved(header))
    chosen = later.column_map()
    assert (chosen.value, chosen.error) == ("col3", "col2")


def test_a_reread_keeps_the_users_none_but_not_a_default_one(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Re-reading the file in the dialog (the header box) brought back the "none"
    the error combo showed by default, over the column the new reading detects."""
    path = _table(tmp_path / "kepler.csv", "time,flux,flux_err", ",")
    dialog = _dialog(qtbot, path)
    dialog._header.setChecked(False)  # no error column is found read this way
    assert dialog.column_map().error == NO_COLUMN
    dialog._header.setChecked(True)
    assert dialog.column_map().error == "flux_err"
    _pick(dialog._combos["error"], NO_COLUMN)  # the user's own "none" stays
    dialog._delimiter.setCurrentIndex(dialog._delimiter.findData(","))
    assert dialog.column_map().error == NO_COLUMN


# -- GUI-3: only a domain the user chose is remembered --------------------------------
def test_a_flux_file_does_not_make_the_next_ambiguous_file_flux(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """GUI-3 sequence B: the domain read off 'flux' was remembered, so an OGLE table
    (col2) and an HJD,V file loaded as fluxes."""
    flux = _dialog(qtbot, _table(tmp_path / "flux.csv", "time,flux,flux_err", ","))
    assert flux.domain() is Domain.FLUX and flux.choice().domain is Domain.FLUX
    saved = _saved(flux)
    assert _remembered(saved).domain is None
    ogle = _table(tmp_path / "ogle.dat")
    dialog = _dialog(qtbot, ogle, saved)
    assert dialog.domain() is Domain.MAGNITUDE
    lc = _load(dialog, ogle)
    assert isinstance(lc, LightCurve) and lc.domain is Domain.MAGNITUDE
    hdr = _table(tmp_path / "hdr.csv", "hjd,V,Verr", ",")
    dialog = _dialog(qtbot, hdr, _saved(dialog))
    assert dialog.domain() is Domain.MAGNITUDE


def test_a_domain_the_user_chose_applies_only_where_names_say_nothing(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Fluxes picked for a headerless table: the next one starts as fluxes, a 'mag'
    file reads as magnitudes, and the choice survives it for the file after."""
    rows = [f"{t} {500 + i:.3f}" for i, t in enumerate(JD)]
    first = _dialog(qtbot, _write(tmp_path / "a.dat", rows))
    assert first.domain() is Domain.MAGNITUDE
    _pick(first._domain_combo, Domain.FLUX)
    saved = _saved(first)
    assert _remembered(saved).domain is Domain.FLUX
    second = _dialog(qtbot, _write(tmp_path / "b.dat", rows), saved)
    assert second.domain() is Domain.FLUX
    mag = _dialog(
        qtbot, _table(tmp_path / "mag.csv", "time,mag,mag_err", ","), _saved(second)
    )
    assert mag.domain() is Domain.MAGNITUDE
    lc = _load(mag, tmp_path / "mag.csv")
    assert isinstance(lc, LightCurve) and lc.domain is Domain.MAGNITUDE
    third = _dialog(qtbot, _write(tmp_path / "c.dat", rows), _saved(mag))
    assert third.domain() is Domain.FLUX


# -- through the window ---------------------------------------------------------------
@pytest.fixture
def _isolated_qsettings(tmp_path: Path) -> None:
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )


@pytest.mark.usefixtures("_isolated_qsettings")
def test_the_window_remembers_only_what_the_user_chose(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """GUI-2 and GUI-3 end to end: Load in the real dialog, QSettings in between."""
    from cuperiod.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication)
    window = MainWindow(app)
    qtbot.addWidget(window)
    window.show()

    def load(path: Path) -> object:
        window._load_file(path)
        live = getattr(window._dialogs, "live_dialogs", ())
        dialogs = [d for d in live if isinstance(d, PreviewDialog)]
        assert len(dialogs) == 1
        dialog = dialogs[0]
        assert dialog.can_load(), dialog._problem.text()
        dialog.accept()
        return window.controller.state.current_lc

    lc = load(_table(tmp_path / "flux.csv", "time,flux,flux_err", ","))
    assert isinstance(lc, LightCurve) and lc.domain is Domain.FLUX
    lc = load(_table(tmp_path / "ogle.dat"))
    assert isinstance(lc, LightCurve) and lc.domain is Domain.MAGNITUDE
    lc = load(_table(tmp_path / "mag.csv", "time,mag", ",", columns=2))
    assert isinstance(lc, LightCurve)
    lc = load(_table(tmp_path / "mb.csv", "time,mag,mag_err,band", ",", columns=4))
    assert isinstance(lc, MultiBandLightCurve) and lc.n_bands == 2
    window.controller.shutdown()
