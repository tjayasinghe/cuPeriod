"""The import dialog (UX-03): choose columns, domain and text splitting, then load.

Everything runs offscreen and non-modally: the dialog is shown with ``open()`` and its
answer arrives through ``choice_made`` -- never ``exec()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytestqt.qtbot import QtBot

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.loaders import (
    NO_COLUMN,
    FilePreview,
    ImportChoice,
    ReadOptions,
    describe_time,
    guess_columns,
    is_text_table,
    load_path,
    preview_file,
    sniff_delimiter,
)
from cuperiod.gui.qt import Qt, QtWidgets
from cuperiod.gui.widgets.preview_dialog import ImportDialog, PreviewDialog

JD = (2458000.123456, 2458001.5, 2458002.987654, 2458003.25)


def _dialog(qtbot: QtBot, path: Path, **kwargs: object) -> PreviewDialog:
    preview = preview_file(path)
    assert preview is not None
    dialog = PreviewDialog(path.name, preview, path=path, **kwargs)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    return dialog


def _pick(combo: QtWidgets.QComboBox, data: object) -> None:
    index = combo.findData(data)
    assert index >= 0, data
    combo.setCurrentIndex(index)


def _dat(path: Path) -> Path:
    """An OGLE / Period04 style table: ``JD mag err``, no header row."""
    path.write_text(
        "".join(f"{t} {12.3 + 0.01 * i:.2f} 0.0{i + 1}\n" for i, t in enumerate(JD)),
        encoding="utf-8",
    )
    return path


# -- the loader side ------------------------------------------------------------------


def test_a_headerless_dat_file_is_previewed_and_guessed(tmp_path: Path) -> None:
    preview = preview_file(_dat(tmp_path / "star.dat"))
    assert preview is not None  # .dat used to load blind, without a preview
    assert preview.columns == ["col1", "col2", "col3"]
    assert preview.n_rows == 4
    assert preview.rows[0][0] == "2458000.123456"  # full precision (UX-21)
    assert not preview.mapping  # nothing detectable by name ...
    assert guess_columns(preview) == {  # ... but the numbers say JD mag err
        "time": "col1",
        "value": "col2",
        "error": "col3",
    }


def test_no_column_means_absent_not_detected(tmp_path: Path) -> None:
    path = tmp_path / "bands.csv"
    pd.DataFrame(
        {
            "time": np.arange(20.0),
            "mag": np.linspace(12, 13, 20),
            "mag_err": np.full(20, 0.01),
            "band": ["g", "r"] * 10,
        }
    ).to_csv(path, index=False)
    assert isinstance(load_path(path), MultiBandLightCurve)  # detected by default
    flat = load_path(
        path, columns=ColumnMap(error=NO_COLUMN, band=NO_COLUMN), domain=Domain.FLUX
    )
    assert isinstance(flat, LightCurve)
    assert flat.n == 20 and flat.error is None
    assert flat.domain is Domain.FLUX
    with_band = load_path(path, columns=ColumnMap(error=NO_COLUMN))
    assert isinstance(with_band, MultiBandLightCurve)
    assert all(b.error is None for b in with_band.bands.values())


def test_read_options_split_a_semicolon_file(tmp_path: Path) -> None:
    path = tmp_path / "eu.csv"
    path.write_text(
        "HJD;V;e_V\n2458000.1;12.3;0.01\n2458001.1;12.4;0.01\n2458002.1;12.5;0.02\n",
        encoding="utf-8",
    )
    assert sniff_delimiter(path) == ";"
    preview = preview_file(path, read=ReadOptions(delimiter=";"))
    assert preview is not None and preview.columns == ["HJD", "V", "e_V"]
    assert guess_columns(preview) == {"time": "HJD", "value": "V", "error": "e_V"}
    choice = ImportChoice(
        ColumnMap(time="HJD", value="V", error="e_V"),
        Domain.MAGNITUDE,
        ReadOptions(delimiter=";"),
    )
    lc = choice.load(path)
    assert isinstance(lc, LightCurve) and lc.n == 3
    np.testing.assert_array_equal(lc.error, [0.01, 0.01, 0.02])


def test_a_headerless_csv_keeps_its_first_row(tmp_path: Path) -> None:
    path = tmp_path / "nohdr.csv"
    path.write_text("2458000.1,12.3,0.01\n2458001.1,12.4,0.01\n", encoding="utf-8")
    lc = load_path(
        path,
        columns=ColumnMap(time="col1", value="col2", error="col3"),
        read=ReadOptions(header=False),
    )
    assert lc.n == 2 and lc.time[0] == 2458000.1


@pytest.mark.parametrize(
    ("name", "header", "sep"),
    [
        ("excel.csv", "time,mag,mag_err", ","),
        ("excel.tsv", "time\tmag\tmag_err", "\t"),
        ("excel.dat", "time mag mag_err", " "),
        ("nohdr.csv", None, ","),
        ("nohdr.dat", None, " "),
    ],
)
def test_a_file_with_a_byte_order_mark_loads_as_previewed(
    qtbot: QtBot, tmp_path: Path, name: str, header: str | None, sep: str
) -> None:
    """Excel's "CSV UTF-8" starts with a byte-order mark: the dialog's columns must
    be the ones the load finds (it detected ``time`` and then could not find it)."""
    rows = [sep.join((repr(t), f"12.{i}", "0.01")) for i, t in enumerate(JD)]
    lines = rows if header is None else [header, *rows]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    dialog = _dialog(qtbot, path)
    assert dialog.can_load(), dialog._problem.text()
    assert dialog.column_map().time == ("time" if header else "col1")
    lc = dialog.choice().load(path)
    assert isinstance(lc, LightCurve) and lc.n == len(JD)
    assert lc.time[0] == JD[0]


def test_lines_the_options_cannot_split_come_back_unloadable(tmp_path: Path) -> None:
    path = tmp_path / "ragged.txt"
    path.write_text("a b c\n1 2\n3 4 5 6\n", encoding="utf-8")
    preview = preview_file(path, read=ReadOptions(delimiter=" "))
    assert preview is not None
    assert not preview.loadable and "delimiter" in preview.note


@pytest.mark.parametrize(
    ("name", "suffix", "text"),
    [
        ("seconds", "time", "seconds"),
        ("years", "year", "decimal years"),
        ("mjd", "mjd", "modified Julian"),
        ("btjd", "btjd", "BTJD"),
        ("words", "id", "no numbers"),
    ],
)
def test_the_time_note_names_the_scale(name: str, suffix: str, text: str) -> None:
    cells = {
        "seconds": ["1600000000.5", "1600000100.5"],
        "years": ["2019.5", "2019.7"],
        "mjd": ["58000.5", "58001.5"],
        "btjd": ["1325.5", "1326.5"],
        "words": ["a", "b"],
    }[name]
    assert text in describe_time(suffix, cells)


def test_text_tables_are_told_apart_from_structured_formats() -> None:
    assert is_text_table("a.dat") and is_text_table("a.csv") and is_text_table("a")
    assert not is_text_table("a.parquet") and not is_text_table("a.fits")
    assert not is_text_table("a.csv.gz")


# -- the dialog -----------------------------------------------------------------------


def test_the_dialog_is_non_modal_and_hands_back_a_choice(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = _dat(tmp_path / "star.dat")
    dialog = _dialog(qtbot, path)
    assert ImportDialog is PreviewDialog
    finished: list[int] = []
    dialog.finished.connect(finished.append)
    dialog.open()  # never exec(): the event loop keeps running
    assert dialog.isVisible()
    with qtbot.waitSignal(dialog.choice_made, timeout=2000) as blocker:
        qtbot.mouseClick(dialog._load_button, Qt.MouseButton.LeftButton)
    choice = blocker.args[0]
    assert isinstance(choice, ImportChoice)
    assert finished == [QtWidgets.QDialog.DialogCode.Accepted.value]
    assert choice.columns == ColumnMap(
        time="col1", value="col2", error="col3", band=NO_COLUMN
    )
    lc = choice.load(path)
    assert isinstance(lc, LightCurve) and lc.n == 4
    assert lc.time[0] == JD[0]


def test_a_guess_is_offered_but_enter_still_cancels(
    qtbot: QtBot, tmp_path: Path
) -> None:
    dialog = _dialog(qtbot, _dat(tmp_path / "star.dat"))
    assert dialog.can_load()
    assert "could not tell which columns" in dialog._summary.text()
    assert "check it" in dialog._summary.text()
    assert dialog._cancel_button is not None and dialog._cancel_button.isDefault()
    # Once the user has confirmed a column, Enter loads.
    _pick(dialog._combos["error"], NO_COLUMN)
    assert dialog._load_button.isDefault()
    assert dialog.column_map().error == NO_COLUMN


def test_the_file_tab_names_each_columns_role(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, _dat(tmp_path / "star.dat"))
    header = [
        dialog._table.horizontalHeaderItem(c).text()
        for c in range(dialog._table.columnCount())
    ]
    assert header == ["col1\n(time)", "col2\n(brightness)", "col3\n(error)"]
    _pick(dialog._combos["error"], NO_COLUMN)
    assert dialog._table.horizontalHeaderItem(2).text() == "col3"


def test_the_loaded_tab_shows_full_precision_times(
    qtbot: QtBot, tmp_path: Path
) -> None:
    dialog = _dialog(qtbot, _dat(tmp_path / "star.dat"))
    loaded = dialog._loaded
    assert loaded.horizontalHeaderItem(0).text().startswith("time (d)")
    times = [loaded.item(r, 0).text() for r in range(loaded.rowCount())]
    assert times == [repr(t) for t in JD]
    assert len(set(times)) == len(JD)  # not four identical 2.458e+06
    assert "full Julian dates" in dialog._time_note.text()


def test_a_semicolon_file_is_split_on_opening(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "eu.csv"
    path.write_text(
        "HJD;V;e_V;filter\n2458000.1;12.3;0.01;V\n2458001.1;12.4;0.01;B\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, path)
    assert dialog._delimiter.isVisibleTo(dialog)
    assert dialog._delimiter.currentData() == ";"
    assert dialog._preview.columns == ["HJD", "V", "e_V", "filter"]
    choice = dialog.choice()
    assert choice.read == ReadOptions(delimiter=";")
    assert choice.columns.time == "HJD" and choice.columns.value == "V"
    lc = choice.load(path)
    assert isinstance(lc, MultiBandLightCurve) and lc.n_bands == 2


def test_a_headerless_csv_is_read_without_a_header(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "nohdr.csv"
    path.write_text(
        "2458000.1,12.3,0.01\n2458001.1,12.4,0.01\n2458002.1,12.5,0.01\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, path)
    assert not dialog._header.isChecked()
    assert dialog._preview.n_rows == 3  # the first row is data, not a header
    assert dialog.choice().read == ReadOptions(header=False)
    # Ticking it back re-reads the file with its first row as the column names.
    dialog._header.setChecked(True)
    assert dialog._preview.columns[0] == "2458000.1"
    assert dialog._preview.n_rows == 2


def test_impossible_choices_disable_load_and_say_why(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "ids.csv"
    path.write_text(
        "name,time,mag\nstar_a,1.5,12.3\nstar_b,2.5,12.4\n", encoding="utf-8"
    )
    dialog = _dialog(qtbot, path)
    assert dialog.can_load() and dialog._problem.isHidden()
    _pick(dialog._combos["value"], "time")
    assert not dialog.can_load()
    assert "different columns" in dialog._problem.text()
    _pick(dialog._combos["value"], "name")
    assert not dialog.can_load()
    assert "no numbers" in dialog._problem.text()
    _pick(dialog._combos["value"], "mag")
    assert dialog.can_load() and dialog._problem.isHidden()
    _pick(dialog._combos["time"], None)
    assert not dialog.can_load() and "time column" in dialog._problem.text()
    dialog.accept()  # Enter on a disabled Load does nothing
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted.value


def test_the_value_column_suggests_the_domain_until_the_user_says(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "both.csv"
    path.write_text("time,mag,flux\n1,12.3,100\n2,12.4,110\n", encoding="utf-8")
    dialog = _dialog(qtbot, path)
    assert dialog.column_map().value == "flux"  # detection prefers a flux column
    assert dialog.domain() is Domain.FLUX
    _pick(dialog._combos["value"], "mag")
    assert dialog.domain() is Domain.MAGNITUDE
    assert "brightness (mag)" in dialog._loaded.horizontalHeaderItem(1).text()
    _pick(dialog._domain_combo, Domain.FLUX)  # the user decides
    _pick(dialog._combos["value"], "flux")
    _pick(dialog._combos["value"], "mag")
    assert dialog.domain() is Domain.FLUX


def test_a_remembered_domain_yields_to_the_columns_name(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """The last file's domain is no choice made in this dialog: a ``flux`` column
    reads as flux, and picking a ``mag`` column still switches to magnitudes."""
    path = tmp_path / "both.csv"
    path.write_text("time,flux,mag,x\n1,100,12.3,5\n2,110,12.4,6\n", encoding="utf-8")
    dialog = _dialog(qtbot, path, domain=Domain.MAGNITUDE)
    assert dialog.column_map().value == "flux"
    assert dialog.domain() is Domain.FLUX
    assert dialog.choice().domain is Domain.FLUX
    _pick(dialog._combos["value"], "mag")
    assert dialog.domain() is Domain.MAGNITUDE
    # A column whose name says nothing keeps the domain shown.
    _pick(dialog._combos["value"], "x")
    assert dialog.domain() is Domain.MAGNITUDE
    # With no name to go by, the remembered domain is the starting point.
    flux_dialog = _dialog(qtbot, _dat(tmp_path / "star.dat"), domain=Domain.FLUX)
    assert flux_dialog.column_map().value == "col2"
    assert flux_dialog.domain() is Domain.FLUX


def test_remembered_read_options_yield_to_the_files_own_look(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Options remembered from the last file apply only where this file's own
    detection leaves one column; a file with a header keeps it."""
    header = tmp_path / "kepler.csv"
    header.write_text(
        "time,flux,flux_err\n1.5,100.0,0.1\n2.5,101.0,0.1\n3.5,99.0,0.1\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, header, read=ReadOptions(header=False))
    assert dialog._header.isChecked() and dialog.choice().read is None
    assert dialog._preview.columns == ["time", "flux", "flux_err"]
    assert dialog.can_load()
    dialog = _dialog(qtbot, header, read=ReadOptions(delimiter=";"))
    assert dialog._delimiter.currentData() is None and dialog.can_load()
    # Spaces in a .csv: detection finds no delimiter, so the remembered one helps.
    spaced = tmp_path / "spaced.csv"
    spaced.write_text(
        "time mag mag_err\n1.5 12.0 0.1\n2.5 12.1 0.1\n3.5 12.2 0.1\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, spaced, read=ReadOptions(delimiter=" "))
    assert dialog._delimiter.currentData() == " "
    assert dialog._preview.columns == ["time", "mag", "mag_err"]
    assert dialog.can_load()


def test_a_header_row_read_as_data_disables_load(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "kepler.csv"
    path.write_text(
        "time,flux,flux_err\n1.5,100.0,0.1\n2.5,101.0,0.1\n3.5,99.0,0.1\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, path)
    assert dialog.can_load()
    dialog._header.setChecked(False)  # the user unticks it by mistake
    assert dialog._preview.columns == ["col1", "col2", "col3"]
    _pick(dialog._combos["time"], "col1")  # as a remembered col1/col2 choice does
    _pick(dialog._combos["value"], "col2")
    assert not dialog.can_load()
    problem = dialog._problem.text()
    assert "'time'" in problem and "First row holds the column names" in problem
    dialog._header.setChecked(True)
    assert dialog.can_load()
    # A headerless file whose band column is text throughout is not flagged.
    bands = tmp_path / "bands.dat"
    bands.write_text(
        "".join(f"{t} 12.{i} 0.01 {'gr'[i % 2]}\n" for i, t in enumerate(JD)),
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, bands)
    assert not dialog._header.isChecked() and dialog.can_load()


def test_a_detected_band_stays_unpinned(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "bands.csv"
    path.write_text(
        "time,mag,band,camera\n1,12.3,g,A\n2,12.4,r,B\n3,12.5,g,A\n",
        encoding="utf-8",
    )
    dialog = _dialog(qtbot, path)
    # As detected: left to the loader, whose too-many-labels guard still applies.
    assert dialog.column_map().band is None
    _pick(dialog._combos["band"], "camera")
    assert dialog.column_map().band == "camera"
    _pick(dialog._combos["band"], NO_COLUMN)
    assert dialog.column_map().band == NO_COLUMN


def test_a_previous_choice_is_restored(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "both.csv"
    path.write_text("t,a,b,c\n1,12.3,100,0.1\n2,12.4,110,0.1\n", encoding="utf-8")
    dialog = _dialog(
        qtbot,
        path,
        columns=ColumnMap(time="t", value="b", error="missing"),
        domain=Domain.FLUX,
    )
    chosen = dialog.column_map()
    assert (chosen.time, chosen.value) == ("t", "b")
    assert chosen.error != "missing"  # a name the file lacks is ignored
    assert dialog.domain() is Domain.FLUX


def test_array_columns_cannot_be_picked(qtbot: QtBot) -> None:
    preview = FilePreview(
        columns=["id", "jd", "mag"],
        rows=[["1", "[5 values]", "[5 values]"]],
        n_rows=1,
        mapping={"time": "jd", "value": "mag"},
        domain="magnitude",
        multiband=False,
        note="Cannot load: wide table.",
        loadable=False,
        nested_columns=("jd", "mag"),
    )
    dialog = PreviewDialog("wide.parquet", preview)
    qtbot.addWidget(dialog)
    assert not dialog.can_load()
    assert not dialog._note.isHidden()
    combo = dialog._combos["time"]
    model = combo.model()
    row = combo.findData("jd")
    assert not (model.flags(model.index(row, 0)) & Qt.ItemFlag.ItemIsEnabled)
    assert not dialog._delimiter.isVisibleTo(dialog)  # no path: no text options


def test_every_control_is_named_and_explained(qtbot: QtBot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, _dat(tmp_path / "star.dat"))
    controls = [*dialog._combos.values(), dialog._domain_combo, dialog._delimiter]
    controls.append(dialog._header)
    for widget in controls:
        assert widget.accessibleName(), widget
        assert widget.toolTip(), widget
    labels = dialog.findChildren(QtWidgets.QLabel)
    buddies = {label.buddy() for label in labels if label.buddy()}
    for widget in [*dialog._combos.values(), dialog._domain_combo, dialog._delimiter]:
        assert widget in buddies  # a mnemonic label reaches every picker
    for table in (dialog._table, dialog._loaded):
        assert table.accessibleName() and not table.tabKeyNavigation()
