"""Batch orchestration: :func:`batch_periodograms`.

Scales a method (or several) over many light curves with user-chosen workers:

* ``device="cpu"`` — a process pool on the CPU backends: BLAS/OpenMP pinned to one
  thread per worker and numba to its share of the cores (``cores // workers``).
* ``device="gpu"`` with one worker — a single process holding reusable plan/kernel
  engines, the throughput path for one device.
* ``device="gpu"`` with several workers — a process pool, each worker building its own
  engines once, parallelizing the CPU-side work over an otherwise-idle GPU.

Results are flattened to one row per light curve and method and either returned in
memory (``sink=None``) or written to Parquet/CSV, in input order. A directory sink
writes one part file per chunk. Resuming (the default) reads the ``(key, method)``
pairs already in the sink and computes only what is missing, so adding inputs or a
method to a finished run processes just the new work.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

from cuperiod.api import _multiband_grid
from cuperiod.batch.io import (
    InputItem,
    csv_safe,
    load_source,
    peak_extra_columns,
    periodogram_to_row,
    resolve_inputs,
    source_label,
    write_rows,
)
from cuperiod.batch.sizing import (
    clamp_pool_workers,
    cpu_worker_count,
    init_worker_threads,
    numba_threads_per_worker,
    pinned_worker_threads,
)
from cuperiod.core.backend import (
    cpu_pool_backend,
    is_gpu_selector,
    normalize_backend,
    requested_backend,
)
from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.errors import _user_stacklevel
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, raise_if_unusable
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import SettingsInput, get_method, select_settings

#: Per-process engine cache (method name -> reusable plan/kernel engine), set for the
#: inline paths or by the pool initializer.
_WORKER_ENGINES: dict[str, object] = {}

#: Light curves per chunk when no pool is used, and the cap of the automatic size.
DEFAULT_CHUNK_SIZE = 256

#: Automatic chunking aims at this many tasks per pool worker, so a slow light curve
#: cannot leave the rest of the pool idle while one worker grinds through its chunk.
_TASKS_PER_WORKER = 4

#: A directory sink's part files (``.tmp``: an interrupted write of an older version),
#: and this version's in-flight temporaries for them. Nothing else is ever removed.
_PART_RE = re.compile(r"^part-(\d{5,})\.parquet$")
_STALE_PART_RE = re.compile(r"^part-\d{5,}\.parquet\.tmp$")
_PART_TEMP_RE = re.compile(r"^\.part-\d{5,}\.parquet\.\d+\.[0-9a-f]+\.tmp$")

#: Settings fields that choose hardware or batch sizes rather than results. They are
#: left out of a directory sink's fingerprint, so a run can resume on another machine.
_RUNTIME_SETTINGS_FIELDS = frozenset(
    {"backend", "device", "precision", "batch_periods", "direct_freq_batch",
     "period_batch"}
)

#: Request fields a resumed directory sink must share with the run that wrote it.
_FINGERPRINT_FIELDS = (
    "n_best", "store_raw", "columns", "domain", "band_column", "grid"
)

_GUARD_HINT = (
    "A batch with workers > 1 runs in spawned worker processes, which import the "
    "calling script again: a script that starts a batch at module level must do so "
    "under `if __name__ == '__main__':` (or pass workers=1 to run in this process)."
)
_POOL_HINT = (
    _GUARD_HINT + " If the script is guarded, a worker process died, for example by "
    "running out of memory."
)

#: Environment prefix of the deprecated :class:`~cuperiod.BatchSettings`.
_BATCH_ENV_PREFIX = "CUPERIOD_BATCH_"


def _warn_dead_batch_env() -> None:
    """Warn that ``CUPERIOD_BATCH_*`` variables change nothing.

    They look like every other ``CUPERIOD_<MODEL>_<FIELD>`` override, but their model
    (``BatchSettings``, deprecated) was never read by any driver, so a job configured
    through them ran with the defaults without a word. Matched case-insensitively,
    as pydantic-settings matches the variables it does read.
    """
    names = sorted(k for k in os.environ if k.upper().startswith(_BATCH_ENV_PREFIX))
    if names:
        warnings.warn(
            f"{', '.join(names)} has no effect: BatchSettings is deprecated and no "
            "batch driver reads CUPERIOD_BATCH_* variables; pass device=, workers=, "
            "n_best=, store_raw= and resume= to the batch function instead.",
            DeprecationWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class _ChunkConfig:
    """Immutable, picklable settings shared by every chunk task."""

    methods: tuple[str, ...]
    backends: Mapping[str, str]
    settings_map: Mapping[str, BaseSettings]
    columns: ColumnMap | None
    domain: Domain | None
    band_column: str | None
    n_best: int
    store_raw: bool
    grid: GridSpec | None = None
    extra_keys: tuple[str, ...] = ()


@dataclass
class BatchSummary:
    """Outcome of a :func:`batch_periodograms` run.

    Attributes
    ----------
    n_inputs, n_done, n_failed, n_skipped : int
        Input count and outcome tallies: ``n_done`` result rows produced by this run,
        ``n_failed`` light curves that raised, ``n_skipped`` light curves already
        complete in a resumed sink.
    methods : tuple of str
        Methods that were run.
    device : str
        ``"cpu"``/``"gpu"``/``"hybrid"``.
    sink : str or None
        Output location, or ``None`` for in-memory.
    errors : list of (str, str)
        ``(key, message)`` for light curves that failed, in input order.
    rows : list of dict or None
        Result rows when ``sink is None``, in input order.
    n_workers : int or None
        Worker processes the run used (``1``: it ran in the calling process; ``0``:
        nothing needed computing). ``None`` when the driver does not report it.
    """

    n_inputs: int
    n_done: int
    n_failed: int
    n_skipped: int
    methods: tuple[str, ...]
    device: str
    sink: str | None
    errors: list[tuple[str, str]] = field(default_factory=list)
    rows: list[dict[str, Any]] | None = None
    n_workers: int | None = None


def _compute_one(
    method_name: str,
    lc: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
    backend_request: str,
    engine: object | None,
    grid: GridSpec | None = None,
) -> Periodogram:
    method = get_method(method_name)
    backend = method.resolve_backend_for(backend_request, settings)
    if isinstance(lc, MultiBandLightCurve):
        if not method.supports_multiband:
            raise ValueError(f"{method.name} does not support multi-band input")
        # Name an unusable error column instead of the empty baseline it leaves.
        raise_if_unusable(lc, method.name)
        g = grid if grid is not None else _multiband_grid(method, lc, settings)
        pg = method.multiband_power(g, lc, settings, backend, engine=engine)
    else:
        raise_if_unusable(lc, method.name)
        single = lc.in_domain(method.natural_domain) if method.natural_domain else lc
        g = grid if grid is not None else method.default_grid(single, settings)
        pg = method.power(g, single, settings, backend, engine=engine)
    # As cup.periodogram does: the settings' peak fields (n_peaks, separations, alias
    # tolerance, downsample_points) become the result's defaults.
    return pg.with_peak_defaults(settings)


def _process_chunk(
    items: Sequence[InputItem], cfg: _ChunkConfig
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Run every method on every light curve in a chunk; collect rows and errors."""
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    for key, source in items:
        try:
            lc = load_source(
                source,
                columns=cfg.columns,
                domain=cfg.domain,
                band_column=cfg.band_column,
            )
            label = source_label(source)
            for name in cfg.methods:
                pg = _compute_one(
                    name,
                    lc,
                    cfg.settings_map[name],
                    cfg.backends[name],
                    _WORKER_ENGINES.get(name),
                    cfg.grid,
                )
                rows.append(
                    periodogram_to_row(
                        key,
                        pg,
                        n_best=cfg.n_best,
                        store_raw=cfg.store_raw,
                        extra_keys=cfg.extra_keys,
                        source=label,
                    )
                )
        except Exception as exc:  # one bad light curve must not kill the batch
            errors.append((key, f"{type(exc).__name__}: {exc}"))
    return rows, errors


def _build_engines(
    method_names: Sequence[str],
    backends: Mapping[str, str],
    settings_map: Mapping[str, BaseSettings],
) -> dict[str, object]:
    """Each method's reusable engine for the backend its selector resolves to here."""
    engines: dict[str, object] = {}
    for name in method_names:
        method = get_method(name)
        settings = settings_map[method.name]
        backend = method.resolve_backend_for(backends[method.name], settings)
        engine = method.make_engine(backend, settings)
        if engine is not None:
            engines[method.name] = engine
    return engines


def _worker_init(
    method_names: tuple[str, ...],
    backends: Mapping[str, str],
    settings_map: Mapping[str, BaseSettings],
) -> None:
    """Pool initializer: build this worker's reusable engines once."""
    global _WORKER_ENGINES
    _WORKER_ENGINES = _build_engines(method_names, backends, settings_map)


def _worker_backend(device: str, backend: str | None, settings: BaseSettings) -> str:
    """The backend selector a worker resolves for one method under ``device``.

    ``device="cpu"`` asks for the CPU: ``"auto"`` becomes ``"cpu"`` (resolving it in
    each worker would put a CPU pool of dozens of processes on the GPU), and a GPU-only
    backend raises. ``device="gpu"`` maps ``"auto"`` to ``"gpu"`` but keeps a concrete
    GPU request (``torch``, ``torch:<device>``, ``cupy``, ``cufinufft``), and refuses
    an explicit CPU backend. A method's ``settings.backend`` pin is honoured either
    way, as :func:`cuperiod.periodogram` honours it.
    """
    if device == "cpu":
        return cpu_pool_backend(backend, settings)
    explicit = "auto" if backend is None else normalize_backend(backend)
    if explicit == "auto":
        pinned = requested_backend(None, settings)
        return "gpu" if pinned == "auto" else pinned
    if explicit == "torch" or is_gpu_selector(explicit):
        return explicit
    raise ValueError(
        f"backend {explicit!r} runs on the CPU, but device='gpu' asks for GPU workers; "
        "use device='cpu', or a GPU backend ('auto', 'gpu', 'torch', "
        "'torch:<device>', or a concrete GPU name)"
    )


def _chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _gpu_worker_count(
    methods: Sequence[str], grid: GridSpec | None, workers: int | None
) -> int:
    """Workers for a GPU pool: the caller's count, else what every method fits.

    Every worker runs every method, so the most demanding method sets the count (not
    the first one listed); an explicit ``grid`` is the real per-worker footprint,
    otherwise :func:`~cuperiod.suggest_gpu_workers` assumes a survey-sized grid.
    """
    if workers is not None:
        return max(1, int(workers))
    grid_size = int(grid.values.size) if grid is not None else None
    return max(
        1, min(suggest_gpu_workers(name, grid_size=grid_size) for name in methods)
    )


def _auto_chunk_size(n_items: int, n_workers: int) -> int:
    """Chunk size for a pool: about four tasks per worker, at most the default."""
    if n_workers <= 1:
        return DEFAULT_CHUNK_SIZE
    per_task = math.ceil(n_items / (_TASKS_PER_WORKER * n_workers))
    return max(1, min(DEFAULT_CHUNK_SIZE, per_task))


def _part_path(sink_dir: Path, idx: int) -> Path:
    return sink_dir / f"part-{idx:05d}.parquet"


# --- sink planning -----------------------------------------------------------------


def _quarantine(path: Path, *, hidden: bool) -> Path:
    """Move an unreadable sink file aside (never delete it) and return the new name."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    prefix = "." if hidden else ""
    target = path.with_name(f"{prefix}{path.name}.corrupt-{stamp}")
    n = 1
    while target.exists():
        target = path.with_name(f"{prefix}{path.name}.corrupt-{stamp}-{n}")
        n += 1
    os.replace(path, target)
    return target


def _corrupt_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "not a valid Parquet/CSV file" (as opposed to I/O)."""
    import pyarrow as pa

    return (pa.ArrowInvalid, UnicodeDecodeError, csv.Error)


def _sink_key(value: Any, *, csv_sink: bool) -> str | None:
    """A key cell as a sink stores it (CSV text cells are formula-escaped)."""
    if value is None:
        return None
    text = str(value)
    return str(csv_safe(text)) if csv_sink else text


def _source_clash(stored: Any, label: str, *, csv_sink: bool) -> bool:
    """Whether a sink row stored for a key came from another file than ``label``.

    In-memory inputs (``label == ""``) and rows written without a ``source`` cell
    carry no provenance and never clash.
    """
    if not stored or not label:
        return False
    return str(stored) != str(csv_safe(label) if csv_sink else label)


def _warn_source_clashes(clashes: Sequence[tuple[str, str, str]]) -> None:
    """Report ``(key, stored source, input source)`` triples skipped as done.

    Resuming after the data moved must not recompute (nor duplicate) a catalogue, so
    such inputs are still skipped, but two different light curves sharing one row
    must not pass without a word.
    """
    if not clashes:
        return
    shown = "; ".join(f"{k!r}: sink has {a}, input is {b}" for k, a, b in clashes[:3])
    warnings.warn(
        f"{len(clashes)} input rows share a key with rows already in the sink that "
        f"came from a different file, and were skipped as done ({shown}). If they "
        "are different light curves, give them distinct keys or use another sink.",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _read_key_table(
    path: Path, fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Rows of ``path`` with only ``fields`` (and ``source``/``status``, if present)."""
    wanted = [*fields, "source", "status"]
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        # Our own handle, closed even when the file is not valid Parquet: a handle
        # left to the garbage collector keeps Windows from moving the file aside.
        with path.open("rb") as fh:
            parquet = pq.ParquetFile(fh)
            names = set(parquet.schema_arrow.names)
            cols = [c for c in dict.fromkeys(wanted) if c in names]
            if not cols:
                return []
            table = parquet.read(columns=cols)
        # Column-wise conversion: several times faster than Table.to_pylist().
        values = [table.column(c).to_pylist() for c in cols]
        return [dict(zip(cols, row, strict=True)) for row in zip(*values, strict=True)]
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {c: row.get(c) for c in wanted if c in row} for row in csv.DictReader(fh)
        ]


#: ``done_if(row) -> bool``: whether a row already in a sink marks its key as done
#: (``batch_pdot`` keeps failed stars, status not ``"ok"``, out of the done set).
DoneFilter = Callable[[Mapping[str, Any]], bool]


def _done_map(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    csv_sink: bool,
    done_if: DoneFilter | None = None,
) -> dict[tuple[str | None, ...], str]:
    """``{key tuple: source}`` for the rows already in a sink."""
    done: dict[tuple[str | None, ...], str] = {}
    for row in rows:
        if done_if is not None and not done_if(row):
            continue
        key = tuple(_sink_key(row.get(f), csv_sink=csv_sink) for f in fields)
        done.setdefault(key, str(row.get("source") or ""))
    return done


def _file_sink_done(
    path: Path, fields: Sequence[str], *, done_if: DoneFilter | None = None
) -> dict[tuple[str | None, ...], str]:
    """The key tuples a file sink already holds (an unreadable file is moved aside).

    Only the ``fields`` columns (plus ``source`` and ``status``, which ``done_if`` may
    read) are loaded. Keys map to the ``source`` recorded with them (``""`` if none).
    """
    if not path.exists():
        return {}
    try:
        rows = _read_key_table(path, fields)
    except _corrupt_errors() as exc:
        moved = _quarantine(path, hidden=False)
        warnings.warn(
            f"existing sink {path} is not a readable {path.suffix} file ({exc}); it "
            f"was moved to {moved} and a new file will be written",
            UserWarning,
            stacklevel=3,
        )
        return {}
    return _done_map(
        rows, fields, csv_sink=path.suffix.lower() == ".csv", done_if=done_if
    )


def _scan_parts(
    sink_dir: Path, fields: Sequence[str], *, done_if: DoneFilter | None = None
) -> tuple[dict[tuple[str | None, ...], str], int]:
    """The key tuples a directory sink's parts hold, and the next free part index.

    Only the key columns are read. An unreadable part is moved aside (dot-prefixed,
    so dataset readers skip it) and its light curves are computed again; an
    interrupted write's ``part-N.parquet.tmp`` from an older version is removed.
    """
    done: dict[tuple[str | None, ...], str] = {}
    next_index = 0
    if not sink_dir.is_dir():
        return done, next_index
    for entry in sorted(sink_dir.iterdir()):
        name = entry.name
        if _STALE_PART_RE.match(name):
            entry.unlink(missing_ok=True)
            continue
        match = _PART_RE.match(name)
        if match is None:
            continue
        next_index = max(next_index, int(match.group(1)) + 1)
        try:
            rows = _read_key_table(entry, fields)
        except _corrupt_errors() as exc:
            moved = _quarantine(entry, hidden=True)
            warnings.warn(
                f"part file {entry} is unreadable ({exc}); it was moved to "
                f"{moved.name} and its light curves will be computed again",
                UserWarning,
                stacklevel=3,
            )
            continue
        for key, src in _done_map(
            rows, fields, csv_sink=False, done_if=done_if
        ).items():
            done.setdefault(key, src)
    return done, next_index


def _purge_parts(sink_dir: Path) -> None:
    """Remove a directory sink's part files (and their temporaries), nothing else."""
    if not sink_dir.is_dir():
        return
    for entry in sink_dir.iterdir():
        name = entry.name
        if (
            _PART_RE.match(name) or _STALE_PART_RE.match(name)
            or _PART_TEMP_RE.match(name)
        ) and entry.is_file():
            entry.unlink()


def _columns_fingerprint(columns: ColumnMap | None) -> dict[str, Any] | None:
    if columns is None:
        return None
    return {f.name: getattr(columns, f.name) for f in dataclasses.fields(columns)}


def _grid_fingerprint(grid: GridSpec | None) -> dict[str, Any] | None:
    if grid is None:
        return None
    digest = hashlib.sha256(grid.values.tobytes()).hexdigest()[:16]
    return {"kind": str(grid.kind), "size": int(grid.size), "sha256": digest}


def _settings_fingerprint(settings: BaseSettings) -> dict[str, Any]:
    dumped: dict[str, Any] = settings.model_dump(
        mode="json", exclude=set(_RUNTIME_SETTINGS_FIELDS)
    )
    return dumped


def _run_fingerprint(
    methods: Sequence[str],
    settings_map: Mapping[str, BaseSettings],
    *,
    n_best: int,
    store_raw: bool,
    columns: ColumnMap | None,
    domain: Domain | None,
    band_column: str | None,
    grid: GridSpec | None,
    extra_keys: Sequence[str],
) -> dict[str, Any]:
    """What a directory sink records about the request that produced its rows."""
    raw = {
        "n_best": int(n_best),
        "store_raw": bool(store_raw),
        "columns": _columns_fingerprint(columns),
        "domain": domain.value if domain is not None else None,
        "band_column": band_column,
        "grid": _grid_fingerprint(grid),
        "extra_keys": list(extra_keys),
        "settings": {
            name: _settings_fingerprint(settings_map[name]) for name in methods
        },
    }
    # Round-trip through JSON so it compares equal to what the manifest reads back.
    normalised: dict[str, Any] = json.loads(json.dumps(raw))
    return normalised


def _merge_config(
    stored: Mapping[str, Any], current: Mapping[str, Any], sink_dir: Path
) -> dict[str, Any]:
    """Check a resumed request against the sink's record; return the updated record.

    A method new to the sink is fine (its rows are added). Anything that changes the
    rows of methods already there -- their settings, ``n_best``, ``store_raw``, the
    column/domain/band handling or the grid -- would leave a silently mixed directory
    and is refused. Settings fields one side lacks (added by an upgrade) are skipped.

    The per-peak extra columns are the directory's schema: a run declaring a subset of
    them writes the recorded set (NaN where it has none), so every part keeps one
    schema; a new method that would add columns the existing parts lack is refused,
    because dataset readers keep only the first part's columns.
    """
    changed = [
        f for f in _FINGERPRINT_FIELDS if f in stored and stored[f] != current.get(f)
    ]
    extra = list(current.get("extra_keys", []))
    old_extra = stored.get("extra_keys")
    if isinstance(old_extra, list):
        added = [k for k in extra if k not in old_extra]
        if added:
            changed.append(
                f"new per-peak columns {added} that the existing part files lack; "
                "write that method to a directory of its own"
            )
        extra = list(old_extra)
    old_settings = stored.get("settings")
    old_settings = old_settings if isinstance(old_settings, Mapping) else {}
    for name, new in current.get("settings", {}).items():
        old = old_settings.get(name)
        if not isinstance(old, Mapping):
            continue
        diff = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        if diff:
            changed.append(f"{name} settings ({', '.join(diff)})")
    if changed:
        raise ValueError(
            f"directory sink {sink_dir} was written with a different request "
            f"({'; '.join(changed)}); resuming would mix incompatible rows. Repeat the "
            "original request, write to a fresh directory, or pass resume=False to "
            "replace the directory's part files."
        )
    return {
        **current,
        "extra_keys": extra,
        "settings": {**old_settings, **current.get("settings", {})},
    }


def _dir_manifest_guard(
    sink_dir: Path,
    chunk_size: int | None,
    resume: bool,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and record a directory sink's manifest before any work starts.

    ``resume=False`` replaces the directory's results: its part files (and only
    those, matched by name) are removed first, so no part of an earlier, larger or
    differently chunked run survives next to the new ones.

    With ``resume=True`` and part files already in the directory:

    * ``chunk_size`` (drivers that resume by chunk index): a different size than the
      one recorded is refused, as it would realign part indices. ``None`` (drivers
      that resume by key) records and checks nothing.
    * ``config`` (see :func:`_run_fingerprint`): a request whose rows would differ
      from the ones already in the directory is refused; new methods are recorded.
      A manifest written before fingerprints existed is accepted.

    A directory without part files has no rows to keep consistent, so its record is
    simply replaced: a run in which every light curve failed (wrong column names,
    say) must not lock the directory to the request that failed.

    Returns
    -------
    dict
        The manifest as written; its ``config`` is the merged record, whose
        ``extra_keys`` the run must use to keep every part on one schema.
    """
    manifest = sink_dir / "_manifest.json"
    stored: dict[str, Any] = {}
    if manifest.exists():
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            stored = loaded if isinstance(loaded, dict) else {}
        except Exception:  # noqa: BLE001 - a corrupt manifest must not abort the run
            stored = {}
    record: dict[str, Any] = {}
    if not resume:
        _purge_parts(sink_dir)
    elif sink_dir.is_dir() and any(
        _PART_RE.match(entry.name) for entry in sink_dir.iterdir()
    ):
        if chunk_size is not None and "chunk_size" in stored:
            try:
                prev = int(stored["chunk_size"])
            except (TypeError, ValueError):
                prev = chunk_size
            if prev != chunk_size:
                raise ValueError(
                    f"directory sink {sink_dir} was written with chunk_size={prev}; "
                    f"resuming requires the same chunk_size (got {chunk_size}). Use "
                    "the same chunk_size, a fresh directory, or resume=False."
                )
        if config is not None and isinstance(stored.get("config"), Mapping):
            config = _merge_config(stored["config"], config, sink_dir)
    sink_dir.mkdir(parents=True, exist_ok=True)
    if chunk_size is not None:
        record["chunk_size"] = chunk_size
    if config is not None:
        record["config"] = config
    manifest.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return record


def _pending_chunks(
    chunks: list[list[InputItem]], sink_kind: str, sink_dir: Path, resume: bool
) -> list[int]:
    """Chunks still to compute for a resumed directory sink; re-plans ``chunks``.

    Resuming by chunk index alone trusted ``part-N`` for whatever chunk ``N`` holds
    now, so a changed input list (a new file sorting between old ones, a removed one,
    a reordered iterable) silently dropped some light curves and duplicated others.
    This reads the keys already written instead: the light curves whose key is in a
    part are done, and the rest are re-chunked (at the size of the first chunk) into
    part indices past the highest existing one. ``chunks`` is rewritten **in place**
    -- empty lists at the indices of finished parts, the new chunks after them -- so a
    caller that indexes ``chunks[idx]`` for each returned ``idx`` and writes
    ``part-idx`` keeps working unchanged. A fresh directory keeps its layout.

    Light curves that failed left no row, so a resume tries them again. A key whose
    part rows came from a different file than this input is still taken as done,
    with a warning (as :func:`batch_periodograms` does).
    """
    if sink_kind != "dir" or not resume:
        return list(range(len(chunks)))
    done, next_index = _scan_parts(sink_dir, ("key",))
    if next_index == 0:
        return list(range(len(chunks)))
    size = max(1, max((len(c) for c in chunks), default=1))
    todo: list[InputItem] = []
    clashes: list[tuple[str, str, str]] = []
    for chunk in chunks:
        for key, src in chunk:
            stored = done.get((key,))
            if stored is None:
                todo.append((key, src))
            elif _source_clash(stored, label := source_label(src), csv_sink=False):
                clashes.append((key, stored, label))
    _warn_source_clashes(clashes)
    replanned = _chunked(todo, size)
    chunks[:] = [[] for _ in range(next_index)] + replanned
    return list(range(next_index, next_index + len(replanned)))


def _classify_sink(sink: str | Path | None) -> tuple[str, Path, Path]:
    """``("memory" | "file" | "dir", directory, file)`` for a ``sink=`` argument.

    Checked before any light curve is computed: a file-suffixed path that is an
    existing directory (a Spark/Dask dataset, say) is refused, and a file sink's parent
    directory is created, so an unusable location fails now rather than after the
    whole batch.
    """
    if sink is None:
        return "memory", Path(), Path()
    path = Path(sink)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq", ".csv"}:
        if path.is_dir():
            raise ValueError(
                f"sink {str(sink)!r} is an existing directory, but its {suffix} suffix "
                "asks for a single-file sink; pass a file path, or a directory path "
                "without an extension for a resumable multi-part sink."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        return "file", path.parent, path
    if suffix and not path.is_dir():
        raise ValueError(
            f"unsupported sink {str(sink)!r}: a file sink must end in .parquet or "
            ".csv; pass a directory (no extension) for a resumable multi-part sink."
        )
    return "dir", path, path


def batch_periodograms(
    inputs: Any,
    method: str | Sequence[str] = "GLS",
    *,
    backend: str | None = "auto",
    device: str = "cpu",
    workers: int | None = None,
    grid: GridSpec | None = None,
    settings: SettingsInput = None,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    band_column: str | None = None,
    sink: str | Path | None = None,
    n_best: int = 10,
    store_raw: bool = False,
    resume: bool = True,
    chunk_size: int | None = None,
) -> BatchSummary:
    """Compute periodograms for many light curves.

    Parameters
    ----------
    inputs : various
        Iterable of light curves / ``(key, lc)`` pairs / paths, a glob string, a
        directory, or a ``(DataFrame, group_column)`` tuple (see
        :func:`cuperiod.batch.io.resolve_inputs`). A glob that matches nothing raises
        ``FileNotFoundError``.
    method : str or sequence of str, default "GLS"
        Method(s) to run on each light curve.
    backend : str, default "auto"
        Backend selector (``"auto"``/``"cpu"``/``"gpu"``/concrete). ``"auto"`` (or
        ``None``) defers to each method's ``settings.backend``, then follows
        ``device``: the CPU backends for ``device="cpu"``, the GPU for
        ``device="gpu"``. A GPU-only backend with ``device="cpu"``, or a CPU-only one
        with ``device="gpu"``, raises ``ValueError``.
    device : {"cpu", "gpu", "hybrid"}, default "cpu"
        Where to run workers. ``"hybrid"`` is not yet implemented.
    workers : int, optional
        Worker count. ``None`` → all-but-one usable core (CPU) or
        :func:`~cuperiod.suggest_gpu_workers` (GPU); ``0`` or ``1`` runs in this
        process. A pool spawns processes that import the calling script again, so a
        script must call this under ``if __name__ == "__main__":``.
    grid : GridSpec, optional
        Custom grid applied to every light curve and method (rare; per-LC defaults
        are typical).
    settings : settings model or mapping, optional
        Per-method settings: one settings object (applied to the methods it belongs
        to), or a ``{method: settings}`` mapping whose keys are matched like method
        names. A settings object of the wrong class, or a key naming no method,
        raises.
    columns, domain : optional
        Column / domain handling for file and table inputs.
    band_column : str, optional
        Multi-band split column for file and ``(DataFrame, group_column)`` inputs
        (``columns=ColumnMap(band=...)`` does the same).
    sink : str or Path, optional
        ``.parquet``/``.csv`` file, or a directory (one part file per chunk).
        ``None`` returns rows in memory. Rows are in input order.
    n_best : int, default 10
        Peaks stored per light curve.
    store_raw : bool, default False
        Also store the peak-preserving downsampled spectrum.
    resume : bool, default True
        Skip work already in the sink: light curves whose ``(key, method)`` rows a
        file or directory sink holds are not computed again, and only new rows are
        added. A directory sink also refuses to resume a request whose rows would
        differ (other settings, ``n_best``, ``store_raw``, columns or grid); adding a
        method is fine. ``False`` replaces the sink (a directory's part files are
        removed first).
    chunk_size : int, optional
        Light curves per chunk/task (and per directory part file). ``None`` picks
        about four tasks per pool worker (at most 256), or 256 without a pool.

    Returns
    -------
    BatchSummary

    Raises
    ------
    ValueError
        For an unknown device, a backend that contradicts ``device``, an unsupported
        sink, or a directory sink written by a different request.
    BackendUnavailableError
        If a method's backend cannot run here (checked before any work).
    FileNotFoundError
        If ``inputs`` names a path that does not exist or a glob that matches nothing.

    Warns
    -----
    UserWarning
        When there is nothing to process (the sink is left untouched), when input
        keys repeat (the repeats are renamed), or when a resumed sink holds a key
        from a different source file.
    DeprecationWarning
        When a ``CUPERIOD_BATCH_*`` variable is set: nothing reads them.
    """
    if device not in {"cpu", "gpu", "hybrid"}:
        raise ValueError("device must be 'cpu', 'gpu', or 'hybrid'")
    if device == "hybrid":
        raise NotImplementedError(
            "hybrid device is planned; use device='cpu' or device='gpu'"
        )
    _warn_dead_batch_env()
    if grid is not None and not isinstance(grid, GridSpec):
        raise TypeError(f"grid must be a GridSpec or None, got {type(grid).__name__}")
    if chunk_size is not None and int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    domain = coerce_domain(domain)

    method_names = (method,) if isinstance(method, str) else tuple(method)
    resolved = [get_method(m) for m in method_names]
    method_objs = list({m.name: m for m in resolved}.values())  # each method once
    if not method_objs:
        raise ValueError("no methods specified")
    methods = tuple(m.name for m in method_objs)
    settings_map = select_settings(method_objs, settings)
    backends = {
        name: _worker_backend(device, backend, settings_map[name]) for name in methods
    }
    for m in method_objs:  # an unusable backend fails here, not once per light curve
        m.resolve_backend_for(backends[m.name], settings_map[m.name])

    sink_kind, sink_dir, sink_file = _classify_sink(sink)
    if store_raw and sink_kind == "file" and sink_file.suffix.lower() == ".csv":
        raise ValueError(
            "store_raw=True produces array-valued spectrum columns that a CSV sink "
            "cannot hold; use a .parquet sink or a directory sink."
        )

    items = resolve_inputs(
        inputs, columns=columns, domain=domain, band_column=band_column
    )
    summary_sink = str(sink) if sink is not None else None
    if not items:
        if sink is not None:
            warnings.warn(
                "batch_periodograms: no inputs to process; the sink was left untouched",
                UserWarning,
                stacklevel=2,
            )
        return BatchSummary(
            n_inputs=0, n_done=0, n_failed=0, n_skipped=0, methods=methods,
            device=device, sink=summary_sink,
            rows=[] if sink_kind == "memory" else None, n_workers=0,
        )

    multiband = (
        band_column is not None
        or (columns is not None and columns.band is not None)
        or any(isinstance(src, MultiBandLightCurve) for _, src in items)
    )
    cfg = _ChunkConfig(
        methods=methods,
        backends=backends,
        settings_map=settings_map,
        columns=columns,
        domain=domain,
        band_column=band_column,
        n_best=n_best,
        store_raw=store_raw,
        grid=grid,
        extra_keys=peak_extra_columns(methods, multiband=multiband),
    )

    # -- what is already done ------------------------------------------------------
    done: dict[tuple[str | None, ...], str] = {}
    next_index = 0
    csv_sink = sink_kind == "file" and sink_file.suffix.lower() == ".csv"
    if sink_kind == "dir":
        fingerprint = _run_fingerprint(
            methods, settings_map, n_best=n_best, store_raw=store_raw,
            columns=columns, domain=domain, band_column=band_column, grid=grid,
            extra_keys=cfg.extra_keys,
        )
        record = _dir_manifest_guard(sink_dir, None, resume, config=fingerprint)
        # Keep the directory's one schema (the recorded extras may be a superset).
        cfg = dataclasses.replace(
            cfg, extra_keys=tuple(record["config"]["extra_keys"])
        )
        if resume:
            done, next_index = _scan_parts(sink_dir, ("key", "method"))
    elif sink_kind == "file" and resume:
        done = _file_sink_done(sink_file, ("key", "method"))

    plan, n_skipped = _plan_items(items, methods, done, csv_sink=csv_sink)

    # -- workers and chunks ----------------------------------------------------------
    if device == "gpu":
        n_workers = _gpu_worker_count(methods, grid, workers)
    else:
        n_workers = cpu_worker_count(workers)
    size = (
        int(chunk_size) if chunk_size is not None
        else _auto_chunk_size(len(plan), n_workers)
    )
    chunks: dict[int, list[InputItem]] = {}
    configs: dict[int, _ChunkConfig] = {}
    groups: dict[tuple[str, ...], list[InputItem]] = {}
    for item, missing in plan:
        groups.setdefault(missing, []).append(item)
    idx = next_index
    for missing, group in groups.items():
        group_cfg = (
            cfg if missing == methods else dataclasses.replace(cfg, methods=missing)
        )
        for chunk in _chunked(group, size):
            chunks[idx] = chunk
            configs[idx] = group_cfg
            idx += 1
    pending = sorted(chunks)

    # -- run ---------------------------------------------------------------------------
    rows_by_chunk: dict[int, list[dict[str, Any]]] = {}
    errors_by_chunk: dict[int, list[tuple[str, str]]] = {}
    n_done = 0

    def absorb(
        idx: int, rows: list[dict[str, Any]], errs: list[tuple[str, str]]
    ) -> None:
        nonlocal n_done
        n_done += len(rows)
        errors_by_chunk[idx] = errs
        if sink_kind == "dir":
            if rows:
                write_rows(rows, _part_path(sink_dir, idx))
        else:
            rows_by_chunk[idx] = rows

    n_procs = min(n_workers, len(pending))
    if not pending:
        n_procs = 0
    elif n_workers <= 1:
        _run_inline(chunks, pending, configs, absorb, gpu=device == "gpu")
        n_procs = 1
    elif device == "gpu":
        _run_pool(
            chunks, pending, cfg, absorb, n_procs, configs=configs,
            initializer=_worker_init, initargs=(methods, backends, settings_map),
        )
    else:
        with pinned_worker_threads():
            _run_pool(
                chunks, pending, cfg, absorb, n_procs, configs=configs,
                initializer=init_worker_threads,
                initargs=(
                    numba_threads_per_worker(n_procs),
                    _worker_init,
                    (methods, backends, settings_map),
                ),
            )

    # Chunks finish in any order; report rows and errors in input order.
    all_rows = [row for i in sorted(rows_by_chunk) for row in rows_by_chunk[i]]
    errors = [err for i in sorted(errors_by_chunk) for err in errors_by_chunk[i]]
    if sink_kind == "file":
        _finalize_file(all_rows, sink_file, resume)

    return BatchSummary(
        n_inputs=len(items),
        n_done=n_done,
        n_failed=len(errors),
        n_skipped=n_skipped,
        methods=methods,
        device=device,
        sink=summary_sink,
        errors=errors,
        rows=None if sink_kind != "memory" else all_rows,
        n_workers=n_procs,
    )


def _plan_items(
    items: Sequence[InputItem],
    methods: tuple[str, ...],
    done: Mapping[tuple[str | None, ...], str],
    *,
    csv_sink: bool,
) -> tuple[list[tuple[InputItem, tuple[str, ...]]], int]:
    """Each input's methods still to compute, and how many inputs are complete.

    A key found in the sink with a different source file than this input is still
    taken as done (resuming after the data moved must not recompute, nor duplicate,
    a whole catalogue), but it is reported: two different light curves would
    otherwise share one row without a word.
    """
    plan: list[tuple[InputItem, tuple[str, ...]]] = []
    n_skipped = 0
    clashes: list[tuple[str, str, str]] = []
    for key, src in items:
        lookup = _sink_key(key, csv_sink=csv_sink)
        label = source_label(src)
        missing: list[str] = []
        for name in methods:
            prev = done.get((lookup, name))
            if prev is None:
                missing.append(name)
            elif _source_clash(prev, label, csv_sink=csv_sink):
                clashes.append((key, prev, label))
        if missing:
            plan.append(((key, src), tuple(missing)))
        else:
            n_skipped += 1
    _warn_source_clashes(clashes)
    return plan, n_skipped


def _run_inline(
    chunks: Mapping[int, list[InputItem]],
    pending: Sequence[int],
    configs: Mapping[int, _ChunkConfig],
    absorb: Callable[..., None],
    *,
    gpu: bool,
) -> None:
    """Run the chunks in this process, with engines built once for the whole run."""
    global _WORKER_ENGINES
    first = configs[pending[0]]
    names = tuple(dict.fromkeys(n for idx in pending for n in configs[idx].methods))
    _WORKER_ENGINES = _build_engines(names, first.backends, first.settings_map)
    try:
        for idx in pending:
            absorb(idx, *_process_chunk(chunks[idx], configs[idx]))
            if gpu:
                free_gpu_memory()
    finally:
        _WORKER_ENGINES = {}


def _run_pool(
    chunks: Sequence[Any] | Mapping[int, Any],
    pending: Sequence[int],
    cfg: Any,
    absorb: Any,
    max_workers: int,
    *,
    initializer: Any = None,
    initargs: tuple[Any, ...] = (),
    worker: Any = None,
    configs: Mapping[int, Any] | None = None,
) -> None:
    """Run ``worker(chunks[idx], cfg)`` for each pending chunk in a spawned pool.

    ``configs`` optionally gives a chunk its own config (``configs[idx]``) instead of
    ``cfg``. The pool size is capped at the platform's limit (61 on Windows).

    Raises
    ------
    RuntimeError
        When called while a spawned worker is still importing the main script (an
        unguarded script re-entering the batch).
    BrokenProcessPool
        When a worker dies; the message says how an unguarded script causes that.
    """
    # Always use "spawn". Linux's default "fork" copies the parent's already-built
    # native thread pools (numba / OpenBLAS / OpenMP, plus any CUDA context for the
    # GPU pool) into the child and deadlocks the workers. Spawn starts fresh,
    # thread-pinned workers (the Windows/macOS default) — see pinned_worker_threads.
    # ``worker`` lets a sibling batch driver (pre-whitening, period change) reuse this
    # pool with its own per-chunk function; it must be a module-level callable so spawn
    # can pickle it. Its return value is splatted into ``absorb`` rather than unpacked
    # here, so a driver emitting several tables per chunk needs no pool of its own.
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # A spawned worker re-imports __main__; an unguarded script reaches its batch
        # call again here. Fail with the fix instead of the stdlib's bootstrap error.
        raise RuntimeError(
            "cuperiod batch: a worker process started another batch while importing "
            f"the main script. {_GUARD_HINT}"
        )
    task = worker or _process_chunk
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=clamp_pool_workers(max(1, int(max_workers))), mp_context=ctx,
            initializer=initializer, initargs=initargs,
        ) as pool:
            futures = {
                pool.submit(
                    task, chunks[idx], configs[idx] if configs is not None else cfg
                ): idx
                for idx in pending
            }
            for future in as_completed(futures):
                absorb(futures[future], *future.result())
    except BrokenProcessPool as exc:
        raise BrokenProcessPool(f"{exc} {_POOL_HINT}") from exc


def _finalize_file(rows: list[dict[str, Any]], path: Path, resume: bool) -> None:
    """Write a file sink: with ``resume``, merge ``rows`` into the existing file.

    Existing rows are kept and an incoming row whose ``(key, method)`` already occurs
    is dropped, so a run adding a second method to the same file adds its rows. When
    nothing is new the file is not rewritten at all.

    Parameters
    ----------
    rows : list of dict
        This run's rows.
    path : Path
        The ``.parquet``/``.csv`` sink.
    resume : bool
        Merge into an existing file (``False`` overwrites it).
    """
    if not (resume and path.exists()):
        write_rows(rows, path)
        return
    if not rows:
        return  # a resume that found nothing new: do not even read the file
    existing = _read_existing_rows(path)
    if not existing:
        write_rows(rows, path)
        return
    csv_sink = path.suffix.lower() == ".csv"

    def ident(row: Mapping[str, Any]) -> tuple[str | None, ...]:
        fields = ("key", "method")
        return tuple(_sink_key(row.get(f), csv_sink=csv_sink) for f in fields)

    seen = {ident(r) for r in existing}
    new = [r for r in rows if ident(r) not in seen]
    if new:
        write_rows(existing + new, path)


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    """All rows of an existing file sink.

    A file that is not valid Parquet/CSV is moved aside to
    ``<name>.corrupt-<timestamp>`` (never overwritten) with a warning, and reads as
    empty. Any other failure -- an I/O error, ``MemoryError`` -- propagates: resuming
    must never replace earlier results because one read went wrong.
    """
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            with path.open("rb") as fh:
                rows: list[dict[str, Any]] = pq.read_table(fh).to_pylist()
            return rows
        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except FileNotFoundError:
        return []
    except _corrupt_errors() as exc:
        moved = _quarantine(path, hidden=False)
        warnings.warn(
            f"could not read existing sink {path} ({exc}); it was moved to {moved} "
            "and a new file is written",
            UserWarning,
            stacklevel=3,
        )
        return []


__all__ = ["DEFAULT_CHUNK_SIZE", "BatchSummary", "batch_periodograms"]
