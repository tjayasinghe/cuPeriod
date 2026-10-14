"""Loading light curves for the GUI: arbitrary files plus the bundled demos.

:func:`load_path` reads a single light-curve file through the same table reader the CLI
uses, once, and splits it on a band column when there is one, in every supported
format. :func:`demo_sources` returns ready-to-explore :class:`SourceItem`s for the
example data in a source checkout's ``examples/data`` (a Kepler transit and six ASAS-SN
stars), plus an always-available synthetic multiband curve, so the app has something to
show on first launch even outside a source checkout.

The import dialog (:mod:`cuperiod.gui.widgets.preview_dialog`) lets the user say what
detection could not: which column holds what (:data:`NO_COLUMN` for "this file has no
error / band column"), the brightness domain, and for plain-text tables how a row is
split (:class:`ReadOptions`). Its answer is an :class:`ImportChoice`, which loads
through :func:`load_path` like everything else.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np

from cuperiod.batch.io import resolve_inputs
from cuperiod.core import lightcurve as _lightcurve
from cuperiod.core.columns import (
    BAND_NAMES,
    ERR_NAMES,
    FLUX_NAMES,
    MAG_NAMES,
    TIME_NAMES,
    ColumnMap,
    Domain,
    ResolvedColumns,
)
from cuperiod.core.errors import ColumnResolutionError
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.gui.models import LoadedCurve, SourceItem

_logger = logging.getLogger(__name__)

#: Environment override pointing at a directory of example data.
_ENV_DATA_DIR = "CUPERIOD_EXAMPLE_DATA"

#: Suffixes read line-by-line for a cheap row count in :func:`preview_file` (text
#: formats where ``nrows`` lets us avoid parsing the whole file just for a preview).
_TEXT_SUFFIXES = frozenset({".csv", ".tsv", ".tab"})

#: Parquet suffixes, previewed from their first record batch and footer row count.
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})

#: Formats with their own structure (column names, types), which :class:`ReadOptions`
#: does not apply to. Everything else is read as a plain-text table.
_STRUCTURED_SUFFIXES = frozenset(
    {".parquet", ".pq", ".fits", ".fit", ".fts", ".fz", ".ecsv"}
)

#: Compressed files are read whole by their reader; the preview cannot peek into them.
_COMPRESSED_SUFFIXES = frozenset({".gz", ".bz2", ".xz", ".zip"})

#: A :class:`ColumnMap` field set to this tells :func:`load_path` that the file has no
#: such column, so it must not be auto-detected either (``None`` means "detect it").
#: Only the optional roles, ``error`` and ``band``, may be absent.
NO_COLUMN: Final = ""

#: Delimiters the import dialog offers for plain-text tables, as ``(label, value)``;
#: ``None`` keeps the reader's own choice for the file's extension. A single space
#: means "any run of spaces or tabs", as in the fixed-width tables of OGLE or Period04.
DELIMITERS: Final[tuple[tuple[str, str | None], ...]] = (
    ("Automatic", None),
    ("Comma", ","),
    ("Semicolon", ";"),
    ("Tab", "\t"),
    ("Spaces", " "),
    ("Vertical bar", "|"),
)

#: Most lines read from the top of a text file to preview it (comment lines included),
#: so a file of comments or one enormous line cannot stall the dialog.
_MAX_HEAD_LINES: Final = 400


@dataclass(frozen=True)
class ReadOptions:
    """How to split a plain-text table into columns.

    The default (``delimiter=None, header=True``) reads the file exactly as the CLI and
    :meth:`~cuperiod.LightCurve.from_file` do. Anything else reads it with astropy's
    basic reader: ``header=False`` names the columns ``col1``, ``col2``, ... (the
    headerless ``JD mag err`` tables of OGLE and Period04), and ``delimiter`` splits a
    row on that character (``" "``: any run of whitespace).
    """

    delimiter: str | None = None
    header: bool = True

    @property
    def is_default(self) -> bool:
        """Whether these options leave the reader's own choice in place."""
        return self.delimiter is None and self.header


@dataclass(frozen=True)
class ImportChoice:
    """What the import dialog decided: columns, domain and how text is split.

    ``columns`` may carry :data:`NO_COLUMN` for the error and band roles. Pass the
    three to :func:`load_path`, or call :meth:`load`.
    """

    columns: ColumnMap
    domain: Domain | None = None
    read: ReadOptions | None = None

    def load(self, path: str | Path) -> LoadedCurve:
        """Load ``path`` with this choice (:func:`load_path`)."""
        return load_path(path, columns=self.columns, domain=self.domain, read=self.read)


def is_text_table(path: str | Path) -> bool:
    """Whether ``path`` is read as a plain-text table, so :class:`ReadOptions` apply.

    Parquet, FITS and ECSV carry their own column structure; compressed files are read
    whole by their reader and cannot be previewed line by line.
    """
    p = Path(path)
    suffixes = [s.lower() for s in p.suffixes]
    if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES:
        return False
    return (suffixes[-1] if suffixes else "") not in _STRUCTURED_SUFFIXES


def _default_delimiter(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".tab"}:
        return "\t"
    return None


def _read_text_table(source: Any, path: Path, read: ReadOptions) -> Any:
    """An astropy table of ``source`` (a path or a list of lines) split per ``read``."""
    from astropy.table import Table

    delimiter = read.delimiter if read.delimiter is not None else (
        _default_delimiter(path)
    )
    kwargs: dict[str, Any] = {
        "format": "ascii.basic" if read.header else "ascii.no_header"
    }
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    return Table.read(source, **kwargs)


def _read_table(path: Path, read: ReadOptions | None) -> Any:
    """The whole table of ``path``: the CLI's reader unless ``read`` says otherwise."""
    if read is None or read.is_default or not is_text_table(path):
        return _lightcurve._read_table_object(path)
    return _read_text_table(path, path, read)

#: Key in a loaded curve's ``meta`` under which :func:`load_path` lists what it had to
#: tell the user about the file (rows without a band label, an ignored band column).
LOAD_NOTES_KEY: Final = "load_notes"

#: Most distinct labels an *auto-detected* band column may hold before it is taken for
#: an identifier (per-exposure ids, per-row strings) and ignored. Real photometric
#: systems have a handful of filters; every band costs a scatter item, a legend row
#: and a combo entry built on the GUI thread, and thousands freeze the window.
MAX_AUTO_BANDS: Final = 32


def _read_dataframe(path: Path, *, nrows: int | None = None) -> Any:
    """Read a tabular file into a pandas DataFrame, or None if unreadable.

    ``nrows`` (honored for CSV/TSV) limits how many data rows are parsed — used by
    :func:`preview_file` so previewing a huge CSV doesn't read the whole thing.
    Whitespace after a delimiter is skipped, as the loader's reader does, so a
    ``time, mag, mag_err`` header previews with the columns it loads with.
    """
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows, skipinitialspace=True)
        if suffix in _PARQUET_SUFFIXES:
            return pd.read_parquet(path)
        if suffix in {".tsv", ".tab"}:
            return pd.read_csv(path, sep="\t", nrows=nrows, skipinitialspace=True)
    except Exception:  # noqa: BLE001 - fall back to single-band loading
        _logger.debug("failed to read %s as a tabular file", path, exc_info=True)
        return None
    return None


def _read_parquet_head(path: Path, nrows: int) -> tuple[Any, int] | None:
    """The first ``nrows`` rows of a Parquet file and its total row count, or None.

    Reads one record batch and the footer, never the whole file.
    """
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        total = int(parquet.metadata.num_rows)
        batch = next(parquet.iter_batches(batch_size=max(1, nrows)), None)
        table = (
            parquet.schema_arrow.empty_table()
            if batch is None
            else _pyarrow_table_from_batch(batch)
        )
        return table.to_pandas(), total
    except Exception:  # noqa: BLE001 - not previewable; the load reports any error
        _logger.debug("failed to preview %s as Parquet", path, exc_info=True)
        return None


def _pyarrow_table_from_batch(batch: Any) -> Any:
    import pyarrow as pa

    return pa.Table.from_batches([batch])


def _count_rows(path: Path) -> int | None:
    """Cheap data-row count for a text table: non-blank lines minus the header.

    Blank lines (trailing, interior, CRLF) are skipped, as the reader skips them.
    """
    try:
        with open(path, "rb") as fh:
            total = sum(1 for line in fh if line.strip())
    except OSError:
        return None
    return max(0, total - 1)


def _table_columns(table: Any) -> list[str]:
    """Column names of a pyarrow or astropy table, as strings."""
    names = getattr(table, "colnames", None)
    if names is None:
        names = getattr(table, "column_names", [])
    return [str(name) for name in names]


def _is_nested(table: Any, name: str) -> bool:
    """Whether column ``name`` holds an array per row (a wide, one-row-per-object
    table)."""
    schema = getattr(table, "schema", None)
    if schema is not None and hasattr(schema, "field"):  # pyarrow
        import pyarrow as pa

        kind = schema.field(name).type
        return bool(
            pa.types.is_list(kind)
            or pa.types.is_large_list(kind)
            or pa.types.is_fixed_size_list(kind)
        )
    column = table[name]
    if getattr(column, "ndim", 1) > 1:
        return True
    if getattr(getattr(column, "dtype", None), "kind", "") == "O" and len(column):
        return isinstance(column[0], (list, tuple, np.ndarray))
    return False


def _nested_message(name: str) -> str:
    return (
        f"this looks like a wide/nested table: column {name!r} holds an array of "
        "values in each row (one row per object). cuPeriod reads one row per "
        "observation; split the table into one light curve per object first"
    )


def _resolve(
    table: Any, columns: ColumnMap | None, domain: Domain | str | None
) -> ResolvedColumns | None:
    try:
        return (columns or ColumnMap()).resolve(_table_columns(table), domain=domain)
    except ColumnResolutionError:
        return None


def _build_curve(
    table: Any,
    columns: ColumnMap | None,
    domain: Domain | str | None,
    meta: dict[str, Any],
    notes: list[str],
) -> LoadedCurve:
    """The multiband split of ``table`` given a usable band column, else one band."""
    try:
        multi = MultiBandLightCurve.from_dataframe(
            table, columns=columns, domain=domain, meta=meta
        )
    except ColumnResolutionError:
        # No band column, or one without a single label: the whole table is one band.
        return LightCurve.from_dataframe(
            table, columns=columns, domain=domain, meta=meta
        )
    pinned = columns is not None and columns.band is not None
    if multi.n_bands > MAX_AUTO_BANDS and not pinned:
        resolved = _resolve(table, columns, domain)
        band = resolved.band if resolved is not None else "band"
        notes.append(
            f"band column {band!r} has {multi.n_bands} distinct labels, more than "
            f"the {MAX_AUTO_BANDS} a filter column plausibly holds, so it was taken "
            "for an identifier and the file was loaded as one band"
        )
        return LightCurve.from_dataframe(
            table, columns=columns, domain=domain, meta=meta
        )
    return multi


def _without_absent(
    table: Any, columns: ColumnMap | None, domain: Domain | str | None
) -> tuple[Any, ColumnMap | None]:
    """Honour :data:`NO_COLUMN`: ``(table, columns)`` with the absent roles removed.

    An absent role cannot simply be left ``None``, which means "detect it". The
    table handed on keeps only the columns of the present roles (by their own names,
    so notes and messages still name the file's columns), leaving detection nothing to
    find. A band column that was auto-detected stays unpinned, so the
    :data:`MAX_AUTO_BANDS` guard still applies to it.
    """
    if columns is None:
        return table, None
    no_error = columns.error == NO_COLUMN
    no_band = columns.band == NO_COLUMN
    if not (no_error or no_band):
        return table, columns
    if NO_COLUMN in (columns.time, columns.value):
        raise ColumnResolutionError("the time and value columns cannot be left out")
    probe = replace(
        columns,
        error=None if no_error else columns.error,
        band=None if no_band else columns.band,
    )
    resolved = probe.resolve(_table_columns(table), domain=domain)
    keep = [resolved.time, resolved.value]
    if not no_error and resolved.error is not None:
        keep.append(resolved.error)
    if not no_band and resolved.band is not None:
        keep.append(resolved.band)
    _, getter = _lightcurve._adapt_table(table)
    sub = {name: getter(name) for name in dict.fromkeys(keep)}
    pinned = ColumnMap(
        time=resolved.time,
        value=resolved.value,
        error=None if no_error else resolved.error,
        band=None if no_band else columns.band,
    )
    return sub, pinned


def load_path(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    read: ReadOptions | None = None,
) -> LoadedCurve:
    """Load a light curve from a file, returning multiband if a band column is present.

    The file is parsed once, by the reader the CLI and :meth:`LightCurve.from_file`
    use (astropy for text and FITS, pyarrow for Parquet), and split on a band/filter
    column in every format (CSV, TSV, Parquet, ECSV, FITS, ASCII), exactly as
    :meth:`MultiBandLightCurve.from_file` splits it. A band column without any label
    falls back to a single :class:`LightCurve`, as does an auto-detected one with more
    than :data:`MAX_AUTO_BANDS` labels (pin it with ``ColumnMap(band=...)`` to split
    it anyway).

    Anything the user should know about the load -- rows dropped for a missing band
    label, a band column taken for an identifier -- is logged and listed under
    :data:`LOAD_NOTES_KEY` in the returned curve's ``meta``.

    Parameters
    ----------
    path : str or Path
        The file.
    columns : ColumnMap, optional
        Columns to use; ``None`` fields are detected. :data:`NO_COLUMN` in ``error``
        or ``band`` says the file has none (nothing is detected for that role).
    domain : Domain, optional
        The brightness domain; inferred from the value column's name when omitted.
    read : ReadOptions, optional
        How to split a plain-text table (delimiter, header row). Ignored for
        Parquet, FITS and ECSV.

    Raises
    ------
    ColumnResolutionError
        If the time/value columns cannot be resolved, or the table is a wide/nested
        one (an array of observations in each row, one row per object).
    """
    p = Path(path)
    table = _read_table(p, read)
    # Resolving the absent roles away can warn about the data too (a domain override
    # that matches no column): it joins the notes like the reader's own warnings.
    with warnings.catch_warnings(record=True) as early:
        warnings.simplefilter("always")
        table, columns = _without_absent(table, columns, domain)
    resolved = _resolve(table, columns, domain)
    if resolved is not None:
        for name in (resolved.time, resolved.value, resolved.error):
            if name is not None and _is_nested(table, name):
                raise ColumnResolutionError(f"{p.name}: {_nested_message(name)}")
    notes: list[str] = []
    meta: dict[str, Any] = {"source": str(p)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = _build_curve(table, columns, domain, meta, notes)
    caught = [*early, *caught]
    # The reader's UserWarnings describe the data (rows without a band label, a
    # domain override that matches no column) and become notes for the user. Anything
    # else -- a library's DeprecationWarning, say -- is about the code, not the file,
    # so it is re-issued unchanged rather than shown as a fact about the light curve.
    for caught_warning in caught:
        if not issubclass(caught_warning.category, UserWarning):
            warnings.warn_explicit(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
            )
    notes[:0] = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    notes = list(dict.fromkeys(notes))
    if not notes:
        return curve
    for note in notes:
        _logger.warning("%s: %s", p.name, note)
    return replace(curve, meta={**curve.meta, LOAD_NOTES_KEY: tuple(notes)})


@dataclass(frozen=True)
class FilePreview:
    """A cheap preview of a tabular file: first rows + the detected column mapping.

    ``note`` explains anything that stops the file from loading as shown (no data
    rows, a wide/nested table, lines the reader cannot split); ``loadable`` is False
    when loading cannot succeed, so the dialog can disable Load. ``nested_columns``
    names the columns whose cells hold arrays (one row per object), which cannot be
    picked as the time, value or error.
    """

    columns: list[str]
    rows: list[list[str]]
    n_rows: int
    mapping: dict[str, str]  # role -> column (time/value/error/band)
    domain: str
    multiband: bool
    note: str = ""
    loadable: bool = True
    nested_columns: tuple[str, ...] = ()


def _fmt_cell(value: Any) -> str:
    """One preview cell: the shortest text that round-trips the stored value.

    Full-JD times keep every digit (``2458000.123456``, not ``2.458e+06``), and an
    array-valued cell (a wide table) is summarised by its length.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return f"[{len(value)} values]"
    return str(value)


def _format_column(values: Any) -> list[str]:
    """Format one column's cells, keeping the column's own dtype.

    Formatting row by row upcast an integer column to float whenever every column was
    numeric, so a 19-digit source id showed as ``4.56789e+18``. A float32 column is
    formatted as float32 (no ``12.345100402832031`` noise digits).
    """
    array = np.asarray(values)
    if array.dtype.kind in "fiub":
        return [str(cell) for cell in array]
    return [_fmt_cell(cell) for cell in array.tolist()]


def _preview_rows(head: Any) -> list[list[str]]:
    """Rows of preview text, built column by column (positional, so any label works)."""
    columns = [_format_column(head.iloc[:, j].to_numpy()) for j in range(head.shape[1])]
    return [list(row) for row in zip(*columns, strict=True)] if columns else []


def _nested_in_head(head: Any, names: Iterable[str]) -> str | None:
    """The first of ``names`` whose preview cells hold arrays, else None."""
    for name in names:
        if name not in head.columns:
            continue
        for cell in head[name].tolist():
            if isinstance(cell, (list, tuple, np.ndarray)):
                return name
    return None


def _head_lines(path: Path, data_rows: int) -> list[str]:
    """The top of a text file: non-blank lines up to ``data_rows`` non-comment ones."""
    lines: list[str] = []
    data = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.rstrip("\r\n")
            if not text.strip():
                continue
            lines.append(text)
            if not text.lstrip().startswith("#"):
                data += 1
            if data >= data_rows or len(lines) >= _MAX_HEAD_LINES:
                break
    return lines


@lru_cache(maxsize=8)
def _count_data_lines(path: str, mtime_ns: int, size: int) -> int | None:
    """Non-blank, non-comment lines of a text file (cached per file version: the
    dialog re-previews on every delimiter or header change)."""
    try:
        with open(path, "rb") as fh:
            return sum(
                1 for line in fh if line.strip() and not line.lstrip().startswith(b"#")
            )
    except OSError:
        return None


def sniff_delimiter(path: str | Path) -> str | None:
    """The delimiter the top of a text file appears to use, or None when unclear.

    Only a choice among :data:`DELIMITERS` is returned, and only when it splits every
    sampled line into the same number (two or more) of fields: a semicolon CSV from a
    European spreadsheet, a tab file named ``.txt``.
    """
    import csv

    try:
        lines = [
            line
            for line in _head_lines(Path(path), 13)
            if not line.lstrip().startswith("#")
        ]
    except OSError:
        return None
    if len(lines) < 2:
        return None
    candidates = "".join(d for _, d in DELIMITERS if d is not None and d != " ")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=candidates)
    except csv.Error:
        return None
    delimiter = str(dialect.delimiter)
    counts = {len(line.split(delimiter)) for line in lines}
    return delimiter if len(counts) == 1 and min(counts) >= 2 else None


def _generated_names(names: Sequence[str]) -> bool:
    """Whether ``names`` are the reader's ``col1, col2, ...`` (no header row)."""
    return bool(names) and all(n == f"col{i}" for i, n in enumerate(names, start=1))


def _text_head(
    path: Path, read: ReadOptions | None, max_rows: int
) -> tuple[Any, int | None, str]:
    """``(frame, n_rows, problem)`` of a plain-text table's first rows.

    Read with the options the load will use: the extension's own reader by default
    (astropy's guess for ``.dat`` / ``.txt``), the basic reader for explicit options.
    ``frame`` is None, and ``problem`` says why, when the lines do not split into a
    table.
    """
    from astropy.table import Table

    try:
        lines = _head_lines(path, max_rows + 1)
    except OSError as exc:
        return None, None, f"Cannot read the file: {exc}"
    try:
        if read is None or read.is_default:
            table = Table.read(lines, format="ascii")
            has_header = not _generated_names(list(table.colnames))
        else:
            table = _read_text_table(lines, path, read)
            has_header = read.header
    except Exception as exc:  # noqa: BLE001 - any reader error is advice, not a crash
        _logger.debug("failed to preview %s as text", path, exc_info=True)
        return (
            None,
            None,
            "Could not split the first lines into columns "
            f"({type(exc).__name__}). Try another delimiter, or say whether the "
            "first row holds the column names.",
        )
    stat = path.stat()
    counted = _count_data_lines(str(path), stat.st_mtime_ns, stat.st_size)
    n_rows = None if counted is None else max(0, counted - int(has_header))
    return table.to_pandas(), n_rows, ""


def preview_file(
    path: str | Path,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    max_rows: int = 12,
    read: ReadOptions | None = None,
) -> FilePreview | None:
    """First rows of a tabular file plus its auto-detected column mapping.

    Returns None for formats not cheaply previewable here (FITS, ECSV, compressed
    files); the caller then simply loads without a preview. Only ``max_rows`` rows are
    actually parsed: CSV/TSV are read with ``nrows`` and counted by a cheap line scan,
    Parquet from its first record batch and its footer's row count, other text tables
    (``.dat``, ``.txt``) from their first lines with the reader the load uses.
    ``read`` previews a plain-text table split another way (see :class:`ReadOptions`);
    a file the options cannot split comes back unloadable, with the reason as its note.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    n_rows: int | None = None
    if suffix in _PARQUET_SUFFIXES:
        head_and_count = _read_parquet_head(p, max_rows)
        if head_and_count is None:
            return None
        frame, n_rows = head_and_count
    elif (read is None or read.is_default) and suffix in _TEXT_SUFFIXES:
        frame = _read_dataframe(p, nrows=max_rows)
        if frame is None:
            return None
    elif is_text_table(p):
        frame, n_rows, problem = _text_head(p, read, max_rows)
        if frame is None:
            return FilePreview([], [], 0, {}, "?", False, problem, False)
    else:
        return None
    names = [str(c) for c in frame.columns]
    mapping: dict[str, str] = {}
    resolved_domain = "?"
    multiband = False
    try:
        resolved = (columns or ColumnMap()).resolve(names, domain=domain)
    except ColumnResolutionError:
        resolved = None
    if resolved is not None:
        mapping["time"] = resolved.time
        mapping["value"] = resolved.value
        if resolved.error:
            mapping["error"] = resolved.error
        if resolved.band:
            mapping["band"] = resolved.band
            multiband = True
        resolved_domain = resolved.domain.value
    head = frame.head(max_rows)
    rows = _preview_rows(head)
    if n_rows is None:
        n_rows = len(frame)
        if suffix in _TEXT_SUFFIXES:
            counted = _count_rows(p)
            if counted is not None:
                n_rows = counted
    note, loadable = _preview_verdict(head, mapping, int(n_rows))
    nested = tuple(
        name
        for j, name in enumerate(names)
        if any(isinstance(cell, (list, tuple, np.ndarray)) for cell in head.iloc[:, j])
    )
    return FilePreview(
        names,
        rows,
        int(n_rows),
        mapping,
        resolved_domain,
        multiband,
        note,
        loadable,
        nested,
    )


def _preview_verdict(
    head: Any, mapping: dict[str, str], n_rows: int
) -> tuple[str, bool]:
    """``(note, loadable)`` for a preview."""
    if n_rows == 0:
        return "The file has no data rows, so there is nothing to load.", False
    nested = _nested_in_head(
        head, [mapping[role] for role in ("time", "value", "error") if role in mapping]
    )
    if nested is not None:
        return f"Cannot load: {_nested_message(nested)}.", False
    # Undetected columns are no verdict on the file: the import dialog asks for them
    # (and says so in its summary), so there is nothing to add here.
    return "", True


def _float_or_none(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def numeric_cells(cells: Iterable[str]) -> list[float]:
    """The finite numbers among preview cells (text that ``float()`` accepts)."""
    out: list[float] = []
    for cell in cells:
        value = _float_or_none(cell)
        if value is not None:
            out.append(value)
    return out


def _column_cells(preview: FilePreview, column: str) -> list[str]:
    try:
        j = preview.columns.index(column)
    except ValueError:
        return []
    return [row[j] for row in preview.rows if j < len(row)]


def guess_columns(preview: FilePreview) -> dict[str, str]:
    """A starting point for the import dialog when detection found no mapping.

    Each role is taken by name where a column matches the detection lists; time and
    value otherwise fall to the first unused all-numeric columns, and the error to
    the third numeric column of a three-column numeric table (``JD mag err``, the
    OGLE / Period04 layout). A role nothing suits is left out.
    """
    if preview.mapping:
        return dict(preview.mapping)
    numeric = [
        c
        for c in preview.columns
        if (cells := _column_cells(preview, c))
        and len(numeric_cells(cells)) == len(cells)
    ]
    lower = {c.lower(): c for c in reversed(preview.columns)}
    out: dict[str, str] = {}

    def by_name(role: str, names: Iterable[str]) -> None:
        for name in names:
            column = lower.get(name)
            if column is not None and column not in out.values():
                out[role] = column
                return

    by_name("time", TIME_NAMES)
    by_name("value", (*MAG_NAMES, *FLUX_NAMES))
    for role in ("time", "value"):
        if role not in out:
            spare = [c for c in numeric if c not in out.values()]
            if spare:
                out[role] = spare[0]
    if "value" in out:  # the error of *this* value first, as ColumnMap.resolve does
        v = out["value"].lower()
        by_name(
            "error",
            (f"{v}_err", f"{v}err", f"{v}_error", f"{v}error", f"e_{v}", f"d{v}"),
        )
    if "error" not in out:
        by_name("error", ERR_NAMES)
    if "error" not in out and len(numeric) == 3 and len(preview.columns) == 3:
        spare = [c for c in numeric if c not in out.values()]
        if spare:
            out["error"] = spare[0]
    by_name("band", BAND_NAMES)
    return out


#: What a time column's name says about its scale, by lower-case name.
_TIME_SCALES: Final[dict[str, str]] = {
    "bjd_tdb": "barycentric Julian dates (BJD, TDB)",
    "bjd": "barycentric Julian dates (BJD)",
    "hjd": "heliocentric Julian dates (HJD)",
    "btjd": "TESS dates (BTJD = BJD - 2457000)",
    "bkjd": "Kepler dates (BKJD = BJD - 2454833)",
    "mjd": "modified Julian dates (MJD = JD - 2400000.5)",
    "hmjd": "heliocentric modified Julian dates (HMJD)",
    "midpointmjdtai": "modified Julian dates (MJD, TAI)",
    "jd": "Julian dates (JD)",
}


def describe_time(column: str, cells: Iterable[str]) -> str:
    """One or two sentences on what the time column holds, for the import dialog.

    Names the scale the column's name implies (BJD, HJD, MJD, BTJD, ...) and checks the
    magnitude of its numbers: cuPeriod reads time in days, so values that look like
    seconds (Unix time) or decimal years are called out before they turn into
    frequencies in the wrong unit.
    """
    values = numeric_cells(cells)
    if not values:
        return (
            f"Column {column!r} holds no numbers in the first rows, so it cannot be "
            "the time."
        )
    middle = float(np.median(values))
    span = float(max(values) - min(values))
    scale = _TIME_SCALES.get(column.lower())
    if 2.3e6 <= middle < 2.6e6:
        kind = scale or "full Julian dates"
    elif 3.0e4 <= middle < 1.0e5:
        kind = scale or "modified Julian dates (JD - 2400000.5)"
    elif 1.0e8 <= middle < 1.0e10:
        return (
            f"Warning: values near {middle:.4g} look like seconds (Unix or mission "
            "time). cuPeriod reads time in days, so frequencies would come out "
            "86400 times too high; convert the column to days first."
        )
    elif 1800.0 <= middle <= 2200.0 and span < 300.0 and any(
        v != int(v) for v in values
    ):
        return (
            f"Warning: values near {middle:.6g} look like decimal years. cuPeriod "
            "reads time in days; convert the column to days (e.g. JD) first."
        )
    else:
        kind = scale or "days from the data's own reference epoch"
    return (
        f"Time {column!r}: {kind}, read in days at full precision, so frequencies "
        "are in cycles per day."
    )


def example_data_dir() -> Path | None:
    """Locate the ``examples/data`` directory, or None if it is not on disk.

    Checks ``$CUPERIOD_EXAMPLE_DATA`` first, then the source checkout this module was
    imported from (``<root>/examples/data`` next to ``<root>/src/cuperiod`` and a
    ``pyproject.toml``, as in a clone or an editable install). Nothing above the
    checkout root is searched: from an installed package that walk reached the home
    directory and the drive root, where any user can plant an ``examples/data`` whose
    files the GUI would then parse at every launch.
    """
    override = os.environ.get(_ENV_DATA_DIR)
    if override and Path(override).is_dir():
        return Path(override)
    parents = Path(__file__).resolve().parents
    if len(parents) <= 3:
        return None
    root = parents[3]
    candidate = root / "examples" / "data"
    if (
        (root / "pyproject.toml").is_file()
        and (root / "src" / "cuperiod").is_dir()
        and candidate.is_dir()
    ):
        return candidate
    return None


# -- ASAS-SN wide-format adapter ---------------------------------------------------


@lru_cache(maxsize=1)
def _asassn_frame(parquet_path: str) -> Any:
    """Read the ASAS-SN demo parquet into a pandas DataFrame (cached)."""
    import pandas as pd

    return pd.read_parquet(parquet_path)


def _lightcurve_from_asassn_row(row: Any) -> LightCurve:
    """Build a :class:`LightCurve` from one wide-format ASAS-SN row (list columns)."""
    return LightCurve.from_arrays(
        np.asarray(row.jd, dtype=np.float64),
        np.asarray(row.mag, dtype=np.float64),
        np.asarray(row.mag_err, dtype=np.float64),
        domain=Domain.MAGNITUDE,
        meta={
            "id": str(row.asas_sn_id),
            "band": str(row.band),
            "vsx_type": str(row.vsx_type),
            "broad_class": str(row.broad_class),
            "vsx_period": float(row.vsx_period),
        },
    )


def asassn_sources(parquet_path: Path) -> list[SourceItem]:
    """One :class:`SourceItem` per row of the ASAS-SN demo parquet (lazy loaders)."""
    frame = _asassn_frame(str(parquet_path))
    items: list[SourceItem] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        key = f"asassn_{row.asas_sn_id}"
        label = f"{row.asas_sn_id} — {row.vsx_type} (P={float(row.vsx_period):.3f} d)"

        def _loader(r: Any = row) -> LoadedCurve:
            return _lightcurve_from_asassn_row(r)

        items.append(
            SourceItem(
                key=key,
                label=label,
                loader=_loader,
                meta={
                    "vsx_period": float(row.vsx_period),
                    "vsx_type": str(row.vsx_type),
                },
            )
        )
    return items


# -- synthetic multiband (always available) ---------------------------------------


def make_demo_multiband() -> MultiBandLightCurve:
    """A deterministic 3-band sinusoid (P=3.214 d) for the multiband overlay demo."""
    rng = np.random.default_rng(7)
    period = 3.214
    n, t0, baseline = 280, 2458000.0, 120.0
    specs = {
        "g": (0.90, 15.2, 0.020),
        "r": (0.62, 14.6, 0.015),
        "i": (0.41, 14.3, 0.012),
    }
    bands: dict[str, LightCurve] = {}
    for i, (name, (amp, base, noise)) in enumerate(specs.items()):
        time = np.sort(t0 + rng.uniform(0.0, baseline, n))
        mag = base + amp * np.sin(2.0 * np.pi * time / period + i * 0.35)
        mag = mag + rng.normal(0.0, noise, n)
        err = np.full(n, noise)
        bands[name] = LightCurve.from_arrays(
            time, mag, err, domain=Domain.MAGNITUDE, meta={"band": name}
        )
    return MultiBandLightCurve.from_light_curves(
        bands, meta={"id": "demo_multiband", "period": period}
    )


# -- demo registry -----------------------------------------------------------------


def _unique_labels(pairs: Sequence[tuple[str, Any]]) -> list[str]:
    """Display labels for batch items: the batch key, made unique where it clashes.

    Same-named files (``tgt.csv`` and ``tgt.tsv``, or one name in two folders) get
    their file name, then their full path, so the browser never lists two rows that
    look identical.
    """
    labels = [str(key) for key, _ in pairs]
    for widen in (
        lambda src: src.name,
        lambda src: str(src),
    ):
        counts = Counter(labels)
        labels = [
            widen(src) if counts[label] > 1 and isinstance(src, Path) else label
            for label, (_, src) in zip(labels, pairs, strict=True)
        ]
    return labels


def enumerate_sources(
    spec: Any,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    band_column: str | None = None,
) -> list[SourceItem]:
    """Lazily enumerate a batch input into :class:`SourceItem`s.

    ``spec`` is what :func:`cuperiod.batch.io.resolve_inputs` accepts: a glob string,
    a directory, a list of paths, or a ``(DataFrame, group_column)`` pair. Each item's
    ``loader`` reads its source only on selection, so a big folder isn't read up front.
    A file source is keyed by its path (unique) and labelled by its batch key, widened
    to the file name where two keys coincide.
    """
    pairs = resolve_inputs(
        spec, columns=columns, domain=domain, band_column=band_column
    )
    file_columns = columns
    if band_column is not None and (columns is None or columns.band is None):
        file_columns = replace(columns or ColumnMap(), band=band_column)
    items: list[SourceItem] = []
    for (key, source), label in zip(pairs, _unique_labels(pairs), strict=True):

        def _loader(src: Any = source) -> LoadedCurve:
            if isinstance(src, (LightCurve, MultiBandLightCurve)):
                return src
            return load_path(src, columns=file_columns, domain=domain)

        item_key = str(source) if isinstance(source, Path) else str(key)
        items.append(SourceItem(key=item_key, label=label, loader=_loader))
    return items


def demo_sources() -> list[SourceItem]:
    """All demo sources: bundled example files (if present) + the synthetic one.

    A demo file that cannot be read is logged and skipped: the synthetic curve is
    always offered, and the main window must start whatever is on disk.
    """
    items: list[SourceItem] = []
    data_dir = example_data_dir()
    if data_dir is not None:
        kepler = data_dir / "kepler_KIC7532973.csv"
        if kepler.is_file():

            def _load_kepler(path: Path = kepler) -> LoadedCurve:
                return load_path(path)

            items.append(
                SourceItem(
                    key="kepler_KIC7532973",
                    label="Kepler KIC 7532973 — hot-Jupiter transit (flux)",
                    loader=_load_kepler,
                    meta={"hint_method": "BLS"},
                )
            )
        asassn = data_dir / "asassn_examples.parquet"
        if asassn.is_file():
            try:
                items.extend(asassn_sources(asassn))
            except Exception:  # noqa: BLE001 - a bad demo file must not block start-up
                _logger.warning(
                    "skipping the ASAS-SN demos: could not read %s",
                    asassn,
                    exc_info=True,
                )
    items.append(
        SourceItem(
            key="demo_multiband",
            label="Synthetic multiband g,r,i — GLS demo (P=3.214 d)",
            loader=make_demo_multiband,
            meta={"hint_method": "GLS"},
        )
    )
    return items


__all__ = [
    "DELIMITERS",
    "LOAD_NOTES_KEY",
    "MAX_AUTO_BANDS",
    "NO_COLUMN",
    "FilePreview",
    "ImportChoice",
    "ReadOptions",
    "asassn_sources",
    "demo_sources",
    "describe_time",
    "enumerate_sources",
    "example_data_dir",
    "guess_columns",
    "is_text_table",
    "load_path",
    "make_demo_multiband",
    "numeric_cells",
    "preview_file",
    "sniff_delimiter",
]
