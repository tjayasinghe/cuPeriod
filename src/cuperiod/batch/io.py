"""Batch input resolution and result sinks.

Inputs can be an iterable of light curves, a glob pattern, a directory of light-curve
files, or a ``(DataFrame, group_column)`` pair; :func:`resolve_inputs` normalizes any of
them to ``(key, source)`` items, where ``source`` is loaded lazily in the worker so file
paths (not big arrays) cross the process boundary. Results are flattened to one row per
light curve and written as Parquet (default) or CSV.
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import numbers
import os
import secrets
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import _user_stacklevel
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, _scan_suffix
from cuperiod.core.result import Periodogram

#: File extensions treated as light-curve tables when scanning a directory: what
#: :meth:`~cuperiod.LightCurve.from_file` reads, also behind a ``.gz``/``.bz2``/``.xz``
#: compression suffix (``star.fits.gz``, ``star.csv.bz2``).
LIGHTCURVE_SUFFIXES = (
    ".parquet",
    ".pq",
    ".csv",
    ".ecsv",
    ".fits",
    ".fit",
    ".fts",
    ".fz",
    ".dat",
    ".txt",
    ".tsv",
    ".tab",
)


def _is_lightcurve_file(path: Path) -> bool:
    """Whether a directory scan takes ``path``: its format suffix, looked through a
    compression suffix as the reader does, is one of :data:`LIGHTCURVE_SUFFIXES`
    (a compressed Parquet file, which the reader cannot open, is not)."""
    return _scan_suffix(path) in LIGHTCURVE_SUFFIXES


#: Methods whose secondary peaks are chosen alias-diverse (harmonics and aliases of a
#: chosen peak skipped). A box/transit search puts strong peaks at every harmonic and
#: sub-harmonic of the true period, so a plain top-N list is mostly the same signal.
#: The one policy shared by batch rows, the ``cuperiod run`` printout and its JSON.
ALIAS_DIVERSE_METHODS: frozenset[str] = frozenset({"BLS", "TLS"})

#: Per-peak extras each built-in method reports, in column order. Every row of a batch
#: carries ``{extra}_{rank}`` for every declared extra of every method in the batch
#: (NaN where a peak or an extra is missing), so the row schema depends only on the
#: request -- methods, ``n_best``, ``store_raw`` -- and never on how many peaks one
#: light curve happened to have. Directory-sink part files then share one schema, which
#: dataset readers (pyarrow, pandas) need: they take the first part's columns.
PEAK_EXTRA_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "BLS": ("depth", "depth_snr", "duration", "t0", "sde"),
    "TLS": ("sr", "depth", "duration", "t0"),
    "GLS": ("fap",),
}

#: Extras that only multi-band results carry, declared when a batch may hold any.
MULTIBAND_PEAK_EXTRA_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "BLS": ("band_index",),
}

#: First characters that make a spreadsheet evaluate a CSV cell as a formula.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

Source = LightCurve | MultiBandLightCurve | Path
InputItem = tuple[str, Source]


def _canonical_method(name: str) -> str:
    """A method name the way the registry spells it (case and separators ignored)."""
    return "".join(ch for ch in str(name) if ch.isalnum()).upper()


def alias_diverse_for(method: str) -> bool:
    """Whether ``method``'s peaks are listed alias-diverse (see ALIAS_DIVERSE_METHODS).

    Parameters
    ----------
    method : str
        A method name, in any case or spelling the registry accepts.

    Returns
    -------
    bool
    """
    return _canonical_method(method) in ALIAS_DIVERSE_METHODS


def peak_extra_columns(
    methods: Sequence[str], *, multiband: bool = False
) -> tuple[str, ...]:
    """The ordered union of the declared per-peak extras of ``methods``.

    Parameters
    ----------
    methods : sequence of str
        The batch's methods.
    multiband : bool, default False
        Also declare the extras that only multi-band results carry.

    Returns
    -------
    tuple of str
        Extra names, first-seen order, without duplicates.
    """
    seen: dict[str, None] = {}
    for name in methods:
        key = _canonical_method(name)
        for extra in PEAK_EXTRA_COLUMNS.get(key, ()):
            seen.setdefault(extra, None)
        if multiband:
            for extra in MULTIBAND_PEAK_EXTRA_COLUMNS.get(key, ()):
                seen.setdefault(extra, None)
    return tuple(seen)


def source_label(source: Source) -> str:
    """The provenance string a result row records: an absolute path, or ``""``.

    In-memory light curves have no path, so they record the empty string, which is
    never compared when a resumed sink checks that a key still names the same file.
    """
    if isinstance(source, (LightCurve, MultiBandLightCurve)):
        return ""
    return os.path.abspath(os.fspath(source))


def _is_dataframe(obj: Any) -> bool:
    return hasattr(obj, "groupby") and hasattr(obj, "columns")


def resolve_inputs(
    inputs: Any,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    band_column: str | None = None,
) -> list[InputItem]:
    """Normalize a batch input into ``(key, source)`` items.

    Parameters
    ----------
    inputs : various
        Iterable of light curves / ``(key, lc)`` pairs / paths, a glob string, a
        directory path, or a ``(DataFrame, group_column)`` tuple. A string or path
        that exists is taken literally (so a folder named ``run[1]`` is that folder),
        and only otherwise read as a glob pattern.
    columns, domain : optional
        Forwarded to file/table loading.
    band_column : str, optional
        If given with a ``(DataFrame, group_column)`` input, each group is loaded as a
        :class:`MultiBandLightCurve` split on this column. ``columns.band`` does the
        same, as it does for file inputs.

    Returns
    -------
    list of (str, Source)
        Stable keys paired with a light curve or a path to load lazily, each key
        distinct. A path's key is its file stem; files that share a stem
        (``fieldA/lc.csv`` and ``fieldB/lc.csv``) get the shortest trailing path
        that tells them apart (``fieldA/lc``, ``fieldB/lc``). A key that still
        repeats -- given twice, or two light curves with one ``meta["id"]`` -- keeps
        its first occurrence and gets ``#2``, ``#3``, ... on the later ones, so no two
        inputs share a result row or a resume entry in any sink.

    Raises
    ------
    FileNotFoundError
        If a path does not exist and, read as a glob pattern, matches nothing.
    TypeError
        For an unsupported input or entry type, or a ``(key, source)`` pair whose key
        is neither a string nor an integer.

    Warns
    -----
    UserWarning
        When file stems collide (keys were path-qualified), or when keys repeat (the
        repeats were renamed).
    """
    # (DataFrame, group_column)
    if isinstance(inputs, tuple) and len(inputs) == 2 and _is_dataframe(inputs[0]):
        df, group_col = inputs
        # The same predicate load_source uses for files: ColumnMap(band=...) must not
        # mean a joint multi-band fit for files and a pooled single band for a table.
        multiband = band_column is not None or (
            columns is not None and columns.band is not None
        )
        items: list[InputItem] = []
        for key, sub in df.groupby(group_col, sort=False):
            if multiband:
                lc: Source = MultiBandLightCurve.from_dataframe(
                    sub, band_column=band_column, columns=columns, domain=domain
                )
            else:
                lc = LightCurve.from_dataframe(sub, columns=columns, domain=domain)
            items.append((str(key), lc))
        # Distinct group values can share a text form (1 and "1" in one column).
        return _unique_keys(items)

    # Glob / directory / single file given as a string or Path.
    if isinstance(inputs, (str, os.PathLike)):
        paths = _resolve_path_like(inputs)
        return _unique_keys(list(zip(_path_keys(paths), paths, strict=True)))

    # An iterable of light curves / (key, lc) pairs / paths.
    if isinstance(inputs, Iterable):
        return _resolve_iterable(inputs)

    raise TypeError(f"unsupported batch input type: {type(inputs)!r}")


def _resolve_path_like(inputs: str | os.PathLike[str]) -> list[Path]:
    """Expand a file, a directory, or a glob pattern into a sorted list of paths.

    The literal path wins: ``[``, ``?`` and ``*`` are legal in file names (``[`` even on
    Windows), so an existing ``survey [v2]`` directory must not be read as a character
    class that matches nothing. Only a path that does not exist is globbed, and a
    pattern that matches nothing is an error -- a typo in a suffix used to look like a
    successful batch of zero inputs (and, with ``resume=False``, empty the sink).
    """
    text = os.fspath(inputs)
    path = Path(text)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if _is_lightcurve_file(p))
    if path.is_file():
        return [path]
    if _glob.has_magic(text):
        matches = _glob.glob(text, recursive=True)
        if not matches:
            # A pattern below a bracketed directory ('survey [v2]/*.csv'): escape the
            # longest leading part that exists literally and glob only the rest.
            escaped = _escape_existing_prefix(text)
            if escaped is not None:
                matches = _glob.glob(escaped, recursive=True)
        if not matches:
            raise FileNotFoundError(f"glob pattern {text!r} matched no files")
        return sorted(Path(p) for p in matches)
    raise FileNotFoundError(f"no such file, directory, or glob match: {text!r}")


#: Suffixes that make a sink a single file; any other sink is a directory.
_FILE_SINK_SUFFIXES = frozenset({".parquet", ".pq", ".csv"})

#: What a directory sink writes inside itself: its part files, its manifest and the
#: CLI's errors list, each also as a dot-prefixed temporary or ``.corrupt-*`` copy.
_DIR_SINK_FILES = ("part-*", "_manifest.json*", "_errors.csv*")


def _resolve_input_files(
    inputs: str | os.PathLike[str],
    *,
    outputs: Iterable[str | os.PathLike[str]] = (),
) -> list[Path]:
    """The files a path-like batch input names, less the batch's own ``outputs``.

    A batch whose sink sits inside its own input (``"*.csv"`` with ``--out
    results.csv``) otherwise reads its previous results, and the errors list beside
    them, as light curves when it is run again. A file output excludes itself and,
    beside it, the ``.corrupt-*`` copies a resume sets aside and the dot-prefixed
    temporary files of its write-then-rename. A directory output (no file suffix)
    excludes what it writes inside itself (:data:`_DIR_SINK_FILES`), not the rest of
    its contents: a directory sink may be the input folder itself. Folders are
    compared resolved, each one once.

    Raises
    ------
    FileNotFoundError
        As :func:`_resolve_path_like`.
    """
    paths = _resolve_path_like(inputs)
    files: list[Path] = []
    dirs: list[Path] = []
    for output in outputs:
        out = Path(output)
        kind = files if out.suffix.lower() in _FILE_SINK_SUFFIXES else dirs
        kind.append(out.resolve())
    if not files and not dirs:
        return paths
    folders: dict[Path, tuple[tuple[str, ...], bool]] = {}

    def owned(path: Path) -> bool:
        info = folders.get(path.parent)
        if info is None:
            folder = path.parent.resolve()
            names = tuple(os.path.normcase(f.name) for f in files if f.parent == folder)
            info = folders[path.parent] = (names, folder in dirs)
        names, in_dir_sink = info
        name = os.path.normcase(path.name)
        if in_dir_sink and any(
            fnmatch.fnmatch(name.removeprefix("."), pattern)
            for pattern in _DIR_SINK_FILES
        ):
            return True
        return any(
            name == own or name.startswith((f"{own}.corrupt-", f".{own}."))
            for own in names
        )

    return [path for path in paths if not owned(path)]


def _escape_existing_prefix(text: str) -> str | None:
    """``text`` with its longest literally existing leading path glob-escaped.

    Returns ``None`` when no leading part both exists and contains pattern characters
    (escaping would change nothing).
    """
    parts = Path(text).parts
    for k in range(len(parts) - 1, 0, -1):
        prefix = Path(*parts[:k])
        if _glob.has_magic(str(prefix)) and prefix.exists():
            return os.path.join(_glob.escape(str(prefix)), *parts[k:])
    return None


def _path_keys(paths: Sequence[Path]) -> list[str]:
    """Batch keys for file paths: the stem, path-qualified where stems collide.

    Two light curves with one key are indistinguishable in the result rows, and a
    resumed or merged sink treats the second as already done. Unique stems stay bare
    (existing sinks and ``{key: period}`` maps keep working); colliding ones get the
    shortest trailing path that is unique among them -- ``fieldB/lc_0000`` for the same
    name in two folders, ``a.parquet`` for ``a.csv`` beside ``a.parquet``.
    """
    stems = [p.stem for p in paths]
    counts = Counter(stems)
    clashing = {stem for stem, n in counts.items() if n > 1}
    if not clashing:
        return stems
    keys = list(stems)
    for stem in clashing:
        idx = [i for i, s in enumerate(stems) if s == stem]
        tails = _shortest_unique_tails([paths[i] for i in idx])
        for i, key in zip(idx, tails, strict=True):
            keys[i] = key
    final = Counter(keys)
    for i, key in enumerate(keys):
        if final[key] > 1:  # a pathological residue: fall back to the whole path
            keys[i] = Path(os.path.abspath(paths[i])).as_posix()
    renamed = [keys[i] for i, s in enumerate(stems) if s in clashing]
    warnings.warn(
        f"{len(renamed)} input files share a file stem with another input, so their "
        f"keys were path-qualified to stay distinct (e.g. {', '.join(renamed[:3])})",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )
    return keys


def _shortest_unique_tails(paths: Sequence[Path]) -> list[str]:
    """For each path, the shortest trailing posix path unique within ``paths``.

    Candidates are tried shortest first, the extension-less form before the file
    name: ``x``, ``x.csv``, ``dir/x``, ``dir/x.csv``, ... A candidate is taken when no
    other path in the group produces the same string at that position.
    """
    parts = [Path(os.path.abspath(p)).parts for p in paths]
    depth = max(len(p) for p in parts)
    columns: list[list[str]] = []
    for d in range(1, depth + 1):
        for with_suffix in (False, True):
            column = []
            for p, path in zip(parts, paths, strict=True):
                tail = list(p[-d:])
                if not with_suffix:
                    tail[-1] = path.stem
                column.append("/".join(tail))
            columns.append(column)
    out: list[str] = []
    for i in range(len(paths)):
        for column in columns:
            if column.count(column[i]) == 1:
                out.append(column[i])
                break
        else:
            out.append(Path(os.path.abspath(paths[i])).as_posix())
    return out


def _is_key(value: object) -> bool:
    """A usable explicit batch key: a string or an integer (NumPy integers included).

    NumPy integer scalars register as :class:`numbers.Integral`, so the IDs read off a
    catalogue column (``np.int64``) work as they do as Python ints; ``str()`` gives the
    same text for both. Floats are refused: their text is not a stable identifier.
    """
    return isinstance(value, str) or (
        isinstance(value, numbers.Integral) and not isinstance(value, bool)
    )


def _resolve_iterable(inputs: Iterable[Any]) -> list[InputItem]:
    items: list[InputItem] = []
    path_rows: list[int] = []
    sources = (LightCurve, MultiBandLightCurve, str, os.PathLike)
    for i, entry in enumerate(inputs):
        pair = isinstance(entry, tuple) and len(entry) == 2
        if pair and isinstance(entry[1], sources):
            key, src = entry
            if not _is_key(key):
                raise TypeError(
                    f"batch key at index {i} must be a str or an integer, got "
                    f"{type(key).__name__}"
                )
            source = src if isinstance(src, (LightCurve, MultiBandLightCurve)) else (
                Path(src)
            )
            items.append((str(key), source))
        elif isinstance(entry, (LightCurve, MultiBandLightCurve)):
            key = str(entry.meta.get("id", i)) if entry.meta else str(i)
            items.append((key, entry))
        elif isinstance(entry, (str, os.PathLike)):
            path_rows.append(len(items))
            items.append((Path(entry).stem, Path(entry)))
        else:
            raise TypeError(f"unsupported batch entry at index {i}: {type(entry)!r}")
    if len(path_rows) > 1:
        paths = [items[i][1] for i in path_rows]
        keys = _path_keys([p for p in paths if isinstance(p, Path)])
        for row, key in zip(path_rows, keys, strict=True):
            items[row] = (key, items[row][1])
    return _unique_keys(items)


def _unique_keys(items: list[InputItem]) -> list[InputItem]:
    """``items`` with every repeated key made distinct (``key#2``, ``key#3``, ...).

    A key is a light curve's identity in every sink: its result rows, the resume
    check (``key`` and ``(key, method)`` sets), the period catalogue of
    ``batch_pdot``. Two inputs under one key used to share all of that -- rows that
    could not be told apart, and a resumed sink that counted the second as done when
    only the first had run. The first occurrence keeps its key, so the common case
    (keys that are already unique) is untouched. Keys are compared in the form a CSV
    sink stores them (:func:`csv_safe`): ``"=x"`` and ``"'=x"`` are distinct keys
    that a CSV sink would write identically.
    """
    stored = [str(csv_safe(key)) for key, _ in items]
    counts = Counter(stored)
    if all(n == 1 for n in counts.values()):
        return items
    taken = set(stored)
    seen: Counter[str] = Counter()
    out: list[InputItem] = []
    renamed: list[str] = []
    for (key, source), form in zip(items, stored, strict=True):
        seen[form] += 1
        if seen[form] == 1:
            out.append((key, source))
            continue
        n = seen[form]
        while str(csv_safe(f"{key}#{n}")) in taken:  # never onto another real key
            n += 1
        new = f"{key}#{n}"
        taken.add(str(csv_safe(new)))
        out.append((new, source))
        renamed.append(new)
    warnings.warn(
        f"{len(renamed)} batch keys occur more than once; the repeats were renamed "
        f"({', '.join(repr(k) for k in renamed[:3])}) so each light curve keeps its "
        "own result rows and resume entry. Give every light curve its own key to "
        "choose the names (a key-based catalogue, such as batch_pdot's period=, "
        "matches only the first occurrence).",
        UserWarning,
        # Reached from a list, a glob or a table, at different depths.
        stacklevel=_user_stacklevel(),
    )
    return out


def load_source(
    source: Source,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    band_column: str | None = None,
) -> LightCurve | MultiBandLightCurve:
    """Materialize a source: return a light curve unchanged, or load it from a path.

    A file source loads as a :class:`MultiBandLightCurve` split on the band
    column when ``band_column`` (or ``columns.band``) is set; otherwise as a
    single-band :class:`LightCurve`.
    """
    if isinstance(source, (LightCurve, MultiBandLightCurve)):
        return source
    if band_column is not None or (columns is not None and columns.band is not None):
        return MultiBandLightCurve.from_file(
            source, band_column=band_column, columns=columns, domain=domain
        )
    return LightCurve.from_file(source, columns=columns, domain=domain)


def periodogram_to_row(
    key: str,
    pg: Periodogram,
    *,
    n_best: int,
    store_raw: bool,
    extra_keys: Sequence[str] | None = None,
    alias_diverse: bool | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Flatten a :class:`Periodogram` into one result row (peaks + optional raw).

    Parameters
    ----------
    key : str
        The light curve's batch key.
    pg : Periodogram
        The result to flatten.
    n_best : int
        Peaks stored: ``period_k``/``power_k`` for ``k = 1..n_best``, NaN past the
        last peak found.
    store_raw : bool
        Also store the peak-preserving downsampled spectrum (list-valued columns).
    extra_keys : sequence of str, optional
        Per-peak extras written as ``{extra}_{k}`` for every rank, NaN where absent,
        so rows of curves with fewer peaks (or another method) keep the same columns
        (see :func:`peak_extra_columns`). Extras the result has beyond these are still
        written, for the ranks that have them.
    alias_diverse : bool, optional
        Peak selection; defaults to :func:`alias_diverse_for` the result's method.
    source : str, optional
        Provenance written as a ``source`` column (see :func:`source_label`).

    Returns
    -------
    dict
    """
    row: dict[str, Any] = {"key": key, "method": pg.method}
    if source is not None:
        row["source"] = source
    row["backend"] = pg.backend
    row["n_samples"] = pg.n_samples
    row["baseline"] = pg.baseline
    diverse = alias_diverse_for(pg.method) if alias_diverse is None else alias_diverse
    # Rank once: the first pick of either peak selector is the global best candidate,
    # so best_period comes from the same list as period_1.
    peaks = pg.best_periods(max(int(n_best), 1), alias_diverse=diverse)
    row["best_period"] = peaks[0].period if peaks else float("nan")
    declared = tuple(extra_keys) if extra_keys is not None else ()
    for rank in range(1, n_best + 1):
        peak = peaks[rank - 1] if rank <= len(peaks) else None
        row[f"period_{rank}"] = peak.period if peak else float("nan")
        row[f"power_{rank}"] = peak.power if peak else float("nan")
        extra = peak.extra if peak is not None else {}
        for ekey in declared:
            value = extra.get(ekey)
            row[f"{ekey}_{rank}"] = float("nan") if value is None else float(value)
        for ekey, eval_ in extra.items():
            if ekey not in declared:
                row[f"{ekey}_{rank}"] = eval_
    if store_raw:
        freq, power = pg.downsample()
        row["pgram_frequency"] = freq.tolist()
        row["pgram_power"] = power.tolist()
    return row


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write result rows to ``path`` as Parquet (``.parquet``) or CSV (``.csv``).

    The table is written to a fresh, exclusively created temporary file beside
    ``path`` and moved over it, so a reader never sees a half-written file and a
    pre-planted temporary name (or symlink) is never written through.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        _write_parquet(rows, path)
    elif path.suffix.lower() == ".csv":
        _write_csv(rows, path)
    else:
        raise ValueError(
            f"unsupported sink format {path.suffix!r}; use .parquet or .csv"
        )


def _union_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _temp_path(path: Path) -> Path:
    """A unique, dot-prefixed sibling of ``path`` for the write-then-rename.

    The random part means two runs never share (and clobber) one temporary file; the
    leading dot keeps pyarrow/pandas dataset discovery from reading a crash leftover
    as a part of a directory sink.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def _move_into_place(tmp: Path, path: Path) -> None:
    """Rename the finished temporary file over ``path``, naming it if that fails."""
    try:
        os.replace(tmp, path)
    except OSError as exc:
        raise type(exc)(
            f"could not replace {path} ({exc}); the complete result is kept in {tmp}"
        ) from exc


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = _union_columns(rows)
    data = {col: [row.get(col) for row in rows] for col in columns}
    table = pa.table(data)
    tmp = _temp_path(path)
    try:
        # "xb": O_EXCL, so an existing file or symlink at the name is never followed.
        with tmp.open("xb") as fh:
            pq.write_table(table, fh)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _move_into_place(tmp, path)


def csv_safe(value: Any) -> Any:
    """Neutralise a CSV text cell that a spreadsheet would run as a formula.

    Keys and statuses come from file names and table cells, i.e. from whoever
    supplied the data. A text cell starting with ``=``, ``+``, ``-``, ``@``, tab or
    carriage return is prefixed with a single quote (the OWASP CSV-injection rule),
    unless it is a plain number such as ``-5`` or ``+7``, which stays as it was.
    Non-text cells are returned unchanged, and the function is idempotent.
    """
    if isinstance(value, str) and value and value[0] in _CSV_FORMULA_PREFIXES:
        try:
            float(value)
        except ValueError:
            return "'" + value
    return value


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import csv

    columns = _union_columns(rows)
    list_cols = [
        col for col in columns if any(isinstance(row.get(col), list) for row in rows)
    ]
    if list_cols:
        raise ValueError(
            f"CSV sink cannot store array-valued columns {list_cols} (e.g. from "
            "store_raw); use a .parquet sink."
        )
    tmp = _temp_path(path)
    try:
        with tmp.open("x", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(
                {col: csv_safe(value) for col, value in row.items()} for row in rows
            )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _move_into_place(tmp, path)


__all__ = [
    "ALIAS_DIVERSE_METHODS",
    "MULTIBAND_PEAK_EXTRA_COLUMNS",
    "PEAK_EXTRA_COLUMNS",
    "InputItem",
    "Source",
    "alias_diverse_for",
    "csv_safe",
    "load_source",
    "peak_extra_columns",
    "periodogram_to_row",
    "resolve_inputs",
    "source_label",
    "write_rows",
]
