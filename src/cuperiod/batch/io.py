"""Batch input resolution and result sinks.

Inputs can be an iterable of light curves, a glob pattern, a directory of light-curve
files, or a ``(DataFrame, group_column)`` pair; :func:`resolve_inputs` normalizes any of
them to ``(key, source)`` items, where ``source`` is loaded lazily in the worker so file
paths (not big arrays) cross the process boundary. Results are flattened to one row per
light curve and written as Parquet (default) or CSV.
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram

#: File extensions treated as light-curve tables when scanning a directory.
LIGHTCURVE_SUFFIXES = (
    ".parquet",
    ".pq",
    ".csv",
    ".ecsv",
    ".fits",
    ".fit",
    ".dat",
    ".txt",
    ".tsv",
)

Source = LightCurve | MultiBandLightCurve | Path
InputItem = tuple[str, Source]


def _is_dataframe(obj: Any) -> bool:
    return hasattr(obj, "groupby") and hasattr(obj, "columns")


def resolve_inputs(
    inputs: Any,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    band_column: str | None = None,
) -> list[InputItem]:
    """Normalize a batch input into ``(key, source)`` items.

    Parameters
    ----------
    inputs : various
        Iterable of light curves / ``(key, lc)`` pairs / paths, a glob string, a
        directory path, or a ``(DataFrame, group_column)`` tuple.
    columns, domain : optional
        Forwarded to file/table loading.
    band_column : str, optional
        If given with a ``(DataFrame, group_column)`` input, each group is loaded as a
        :class:`MultiBandLightCurve` split on this column.

    Returns
    -------
    list of (str, Source)
        Stable keys paired with a light curve or a path to load lazily.
    """
    # (DataFrame, group_column)
    if isinstance(inputs, tuple) and len(inputs) == 2 and _is_dataframe(inputs[0]):
        df, group_col = inputs
        items: list[InputItem] = []
        for key, sub in df.groupby(group_col, sort=False):
            if band_column is not None:
                lc: Source = MultiBandLightCurve.from_dataframe(
                    sub, band_column=band_column, columns=columns, domain=domain
                )
            else:
                lc = LightCurve.from_dataframe(sub, columns=columns, domain=domain)
            items.append((str(key), lc))
        return items

    # Glob / directory / single file given as a string or Path.
    if isinstance(inputs, (str, Path)):
        return _resolve_path_like(inputs)

    # An iterable of light curves / (key, lc) pairs / paths.
    if isinstance(inputs, Iterable):
        return _resolve_iterable(inputs, columns=columns, domain=domain)

    raise TypeError(f"unsupported batch input type: {type(inputs)!r}")


def _resolve_path_like(inputs: str | Path) -> list[InputItem]:
    text = str(inputs)
    if any(ch in text for ch in "*?[") or "**" in text:
        paths = sorted(Path(p) for p in _glob.glob(text, recursive=True))
        return [(p.stem, p) for p in paths]
    path = Path(inputs)
    if path.is_dir():
        paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in LIGHTCURVE_SUFFIXES
        )
        return [(p.stem, p) for p in paths]
    if path.is_file():
        return [(path.stem, path)]
    raise FileNotFoundError(f"no such file, directory, or glob match: {inputs!r}")


def _resolve_iterable(
    inputs: Iterable[Any], *, columns: ColumnMap | None, domain: Domain | None
) -> list[InputItem]:
    items: list[InputItem] = []
    for i, entry in enumerate(inputs):
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], (str, int))
            and isinstance(entry[1], (LightCurve, MultiBandLightCurve, str, Path))
        ):
            key, src = entry
            src = Path(src) if isinstance(src, str) else src
            items.append((str(key), src))
        elif isinstance(entry, (LightCurve, MultiBandLightCurve)):
            key = str(entry.meta.get("id", i)) if entry.meta else str(i)
            items.append((key, entry))
        elif isinstance(entry, (str, Path)):
            items.append((Path(entry).stem, Path(entry)))
        else:
            raise TypeError(f"unsupported batch entry at index {i}: {type(entry)!r}")
    return items


def load_source(
    source: Source,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> LightCurve | MultiBandLightCurve:
    """Materialize a source: return a light curve unchanged, or load it from a path."""
    if isinstance(source, (LightCurve, MultiBandLightCurve)):
        return source
    return LightCurve.from_file(source, columns=columns, domain=domain)


def periodogram_to_row(
    key: str, pg: Periodogram, *, n_best: int, store_raw: bool
) -> dict[str, Any]:
    """Flatten a :class:`Periodogram` into one result row (peaks + optional raw)."""
    row: dict[str, Any] = {
        "key": key,
        "method": pg.method,
        "backend": pg.backend,
        "n_samples": pg.n_samples,
        "baseline": pg.baseline,
        "best_period": pg.best_period() if pg.size else float("nan"),
    }
    peaks = pg.best_periods(n_best, alias_diverse=(pg.method == "BLS"))
    for rank in range(1, n_best + 1):
        peak = peaks[rank - 1] if rank <= len(peaks) else None
        row[f"period_{rank}"] = peak.period if peak else float("nan")
        row[f"power_{rank}"] = peak.power if peak else float("nan")
        if peak is not None:
            for ekey, eval_ in peak.extra.items():
                row[f"{ekey}_{rank}"] = eval_
    if store_raw:
        freq, power = pg.downsample()
        row["pgram_frequency"] = freq.tolist()
        row["pgram_power"] = power.tolist()
    return row


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write result rows to ``path`` as Parquet (``.parquet``) or CSV (``.csv``)."""
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


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = _union_columns(rows)
    data = {col: [row.get(col) for row in rows] for col in columns}
    table = pa.table(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import csv

    columns = [
        col
        for col in _union_columns(rows)
        if not any(isinstance(row.get(col), list) for row in rows)
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


__all__ = [
    "InputItem",
    "Source",
    "load_source",
    "periodogram_to_row",
    "resolve_inputs",
    "write_rows",
]
