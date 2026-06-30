"""Batch orchestration: :func:`batch_periodograms`.

Scales a method (or several) over many light curves with user-chosen workers:

* ``device="cpu"`` — a process pool, one chunk per worker, math threads pinned to 1.
* ``device="gpu"`` with one worker — a single process holding reusable plan/kernel
  engines, the throughput path for one device.
* ``device="gpu"`` with several workers — a process pool, each worker building its own
  engines once, parallelizing the CPU-side work over an otherwise-idle GPU.

Results are flattened to one row per light curve and either returned in memory
(``sink=None``) or written to Parquet/CSV. A directory sink writes one part file per
chunk and is resumable: a re-run skips chunks whose part already exists.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

from cuperiod.api import _multiband_grid
from cuperiod.batch.io import (
    InputItem,
    load_source,
    periodogram_to_row,
    resolve_inputs,
    write_rows,
)
from cuperiod.batch.sizing import cpu_worker_count, pin_worker_threads
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import get_method

#: Per-process engine cache (method name -> reusable GPU engine), set inline (single
#: GPU process) or by the pool initializer (GPU pool).
_WORKER_ENGINES: dict[str, object] = {}


@dataclass(frozen=True)
class _ChunkConfig:
    """Immutable, picklable settings shared by every chunk task."""

    methods: tuple[str, ...]
    backend: str
    settings_map: Mapping[str, BaseSettings]
    columns: ColumnMap | None
    domain: Domain | None
    n_best: int
    store_raw: bool


@dataclass
class BatchSummary:
    """Outcome of a :func:`batch_periodograms` run.

    Attributes
    ----------
    n_inputs, n_done, n_failed, n_skipped : int
        Input count and per-light-curve outcome tallies.
    methods : tuple of str
        Methods that were run.
    device : str
        ``"cpu"``/``"gpu"``/``"hybrid"``.
    sink : str or None
        Output location, or ``None`` for in-memory.
    errors : list of (str, str)
        ``(key, message)`` for light curves that failed.
    rows : list of dict or None
        Result rows when ``sink is None``.
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


def _compute_one(
    method_name: str,
    lc: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
    backend_request: str,
    engine: object | None,
) -> Periodogram:
    method = get_method(method_name)
    backend = method.resolve_backend(backend_request)
    if isinstance(lc, MultiBandLightCurve):
        if not method.supports_multiband:
            raise ValueError(f"{method.name} does not support multi-band input")
        grid = _multiband_grid(method, lc, settings)
        return method.multiband_power(grid, lc, settings, backend)
    single = lc.in_domain(method.natural_domain) if method.natural_domain else lc
    grid = method.default_grid(single, settings)
    return method.power(grid, single, settings, backend, engine=engine)


def _process_chunk(
    items: Sequence[InputItem], cfg: _ChunkConfig
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Run every method on every light curve in a chunk; collect rows and errors."""
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    for key, source in items:
        try:
            lc = load_source(source, columns=cfg.columns, domain=cfg.domain)
            for method_name in cfg.methods:
                method = get_method(method_name)
                engine = _WORKER_ENGINES.get(method.name)
                pg = _compute_one(
                    method.name,
                    lc,
                    cfg.settings_map[method.name],
                    cfg.backend,
                    engine,
                )
                rows.append(
                    periodogram_to_row(
                        key, pg, n_best=cfg.n_best, store_raw=cfg.store_raw
                    )
                )
        except Exception as exc:  # one bad light curve must not kill the batch
            errors.append((key, f"{type(exc).__name__}: {exc}"))
    return rows, errors


def _build_engines(
    method_names: Sequence[str], settings_map: Mapping[str, BaseSettings]
) -> dict[str, object]:
    engines: dict[str, object] = {}
    for name in method_names:
        method = get_method(name)
        backend = method.resolve_backend("gpu")
        engine = method.make_engine(backend, settings_map[method.name])
        if engine is not None:
            engines[method.name] = engine
    return engines


def _gpu_worker_init(
    method_names: tuple[str, ...], settings_map: Mapping[str, BaseSettings]
) -> None:
    """Pool initializer: build this worker's reusable GPU engines once."""
    global _WORKER_ENGINES
    _WORKER_ENGINES = _build_engines(method_names, settings_map)


def _chunked(items: list[InputItem], size: int) -> list[list[InputItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _part_path(sink_dir: Path, idx: int) -> Path:
    return sink_dir / f"part-{idx:05d}.parquet"


def _dir_manifest_guard(sink_dir: Path, chunk_size: int, resume: bool) -> None:
    """Pin a directory sink's chunk_size so a resume cannot realign part indices.

    Part files are named purely by chunk index, so resuming with a different chunk_size
    would silently drop or duplicate light curves. Refuse the mismatch and record the
    size for the next run.
    """
    manifest = sink_dir / "_manifest.json"
    if resume and manifest.exists():
        try:
            prev = int(json.loads(manifest.read_text(encoding="utf-8"))["chunk_size"])
        except Exception:  # noqa: BLE001 - a corrupt manifest must not abort the run
            prev = chunk_size
        if prev != chunk_size:
            raise ValueError(
                f"directory sink {sink_dir} was written with chunk_size={prev}; "
                f"resuming requires the same chunk_size (got {chunk_size}). Use the "
                "same chunk_size, a fresh directory, or resume=False."
            )
    sink_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"chunk_size": chunk_size}), encoding="utf-8")


def batch_periodograms(
    inputs: Any,
    method: str | Sequence[str] = "GLS",
    *,
    backend: str = "auto",
    device: str = "cpu",
    workers: int | None = None,
    grid: Any = None,
    settings: BaseSettings | Mapping[str, BaseSettings] | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
    band_column: str | None = None,
    sink: str | Path | None = None,
    n_best: int = 10,
    store_raw: bool = False,
    resume: bool = True,
    chunk_size: int = 256,
) -> BatchSummary:
    """Compute periodograms for many light curves.

    Parameters
    ----------
    inputs : various
        Iterable of light curves / ``(key, lc)`` pairs / paths, a glob string, a
        directory, or a ``(DataFrame, group_column)`` tuple.
    method : str or sequence of str, default "GLS"
        Method(s) to run on each light curve.
    backend : str, default "auto"
        Backend selector (``"auto"``/``"cpu"``/``"gpu"``/concrete).
    device : {"cpu", "gpu", "hybrid"}, default "cpu"
        Where to run workers. ``"hybrid"`` is not yet implemented.
    workers : int, optional
        Worker count. ``None`` → all-but-one core (CPU) or
        :func:`~cuperiod.suggest_gpu_workers` (GPU).
    grid : GridSpec, optional
        Custom grid applied to every light curve (rare; per-LC defaults are typical).
    settings : settings model or mapping, optional
        Per-method settings.
    columns, domain : optional
        Column / domain handling for file and table inputs.
    band_column : str, optional
        Multi-band split column for ``(DataFrame, group_column)`` inputs.
    sink : str or Path, optional
        ``.parquet``/``.csv`` file, or a directory (resumable, one part per chunk).
        ``None`` returns rows in memory.
    n_best : int, default 10
        Peaks stored per light curve.
    store_raw : bool, default False
        Also store the peak-preserving downsampled spectrum.
    resume : bool, default True
        Skip chunks already written (directory sink).
    chunk_size : int, default 256
        Light curves per chunk/task.

    Returns
    -------
    BatchSummary
    """
    if device not in {"cpu", "gpu", "hybrid"}:
        raise ValueError("device must be 'cpu', 'gpu', or 'hybrid'")
    if device == "hybrid":
        raise NotImplementedError(
            "hybrid device is planned; use device='cpu' or device='gpu'"
        )

    method_names = (method,) if isinstance(method, str) else tuple(method)
    methods = tuple(get_method(m).name for m in method_names)
    if not methods:
        raise ValueError("no methods specified")
    settings_map = {name: _settings_for(name, settings) for name in methods}
    cfg = _ChunkConfig(
        methods=methods,
        backend=("gpu" if device == "gpu" else backend),
        settings_map=settings_map,
        columns=columns,
        domain=domain,
        n_best=n_best,
        store_raw=store_raw,
    )

    sink_kind, sink_dir, sink_file = _classify_sink(sink)
    if store_raw and sink_kind == "file" and sink_file.suffix.lower() == ".csv":
        raise ValueError(
            "store_raw=True produces array-valued spectrum columns that a CSV sink "
            "cannot hold; use a .parquet sink or a directory sink."
        )

    items = resolve_inputs(
        inputs, columns=columns, domain=domain, band_column=band_column
    )
    chunks = _chunked(items, max(1, chunk_size))
    if sink_kind == "dir":
        _dir_manifest_guard(sink_dir, max(1, chunk_size), resume)

    pending = _pending_chunks(chunks, sink_kind, sink_dir, resume)
    n_skipped = len(items) - sum(len(chunks[i]) for i in pending)

    all_rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    n_done = 0

    def absorb(
        idx: int, rows: list[dict[str, Any]], errs: list[tuple[str, str]]
    ) -> None:
        nonlocal n_done
        n_done += len(rows)
        errors.extend(errs)
        if sink_kind == "dir":
            if rows:
                write_rows(rows, _part_path(sink_dir, idx))
        else:
            all_rows.extend(rows)

    if device == "gpu" and (workers is None or workers <= 1) and workers != 0:
        workers = workers or suggest_gpu_workers(methods[0])
    if device == "gpu" and workers is not None and workers <= 1:
        _run_gpu_single(chunks, pending, cfg, absorb)
    elif device == "gpu":
        _run_pool(
            chunks, pending, cfg, absorb, int(workers or 1),
            initializer=_gpu_worker_init, initargs=(methods, settings_map),
        )
    else:  # cpu
        n_workers = cpu_worker_count(workers)
        if n_workers <= 1:
            for idx in pending:
                absorb(idx, *_process_chunk(chunks[idx], cfg))
        else:
            pin_worker_threads()
            _run_pool(chunks, pending, cfg, absorb, n_workers)

    if sink_kind == "file":
        _finalize_file(all_rows, sink_file, resume)

    return BatchSummary(
        n_inputs=len(items),
        n_done=n_done,
        n_failed=len(errors),
        n_skipped=n_skipped,
        methods=methods,
        device=device,
        sink=str(sink) if sink is not None else None,
        errors=errors,
        rows=None if sink_kind != "memory" else all_rows,
    )


def _settings_for(
    name: str, settings: BaseSettings | Mapping[str, BaseSettings] | None
) -> BaseSettings:
    method = get_method(name)
    if settings is None:
        return method.coerce_settings(None)
    if isinstance(settings, BaseSettings):
        return settings if isinstance(settings, method.settings_cls) else (
            method.coerce_settings(None)
        )
    for key, value in settings.items():
        if key.upper() == method.name:
            return method.coerce_settings(value)
    return method.coerce_settings(None)


def _classify_sink(sink: str | Path | None) -> tuple[str, Path, Path]:
    if sink is None:
        return "memory", Path(), Path()
    path = Path(sink)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq", ".csv"}:
        return "file", path.parent, path
    if suffix and not path.is_dir():
        raise ValueError(
            f"unsupported sink {str(sink)!r}: a file sink must end in .parquet or "
            ".csv; pass a directory (no extension) for a resumable multi-part sink."
        )
    return "dir", path, path


def _pending_chunks(
    chunks: list[list[InputItem]], sink_kind: str, sink_dir: Path, resume: bool
) -> list[int]:
    if sink_kind == "dir" and resume:
        return [i for i in range(len(chunks)) if not _part_path(sink_dir, i).exists()]
    return list(range(len(chunks)))


def _run_gpu_single(
    chunks: list[list[InputItem]],
    pending: list[int],
    cfg: _ChunkConfig,
    absorb: Any,
) -> None:
    global _WORKER_ENGINES
    _WORKER_ENGINES = _build_engines(cfg.methods, cfg.settings_map)
    try:
        for idx in pending:
            absorb(idx, *_process_chunk(chunks[idx], cfg))
            free_gpu_memory()
    finally:
        _WORKER_ENGINES = {}


def _run_pool(
    chunks: list[list[InputItem]],
    pending: list[int],
    cfg: _ChunkConfig,
    absorb: Any,
    max_workers: int,
    *,
    initializer: Any = None,
    initargs: tuple[Any, ...] = (),
) -> None:
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=initializer, initargs=initargs
    ) as pool:
        futures = {
            pool.submit(_process_chunk, chunks[idx], cfg): idx for idx in pending
        }
        for future in as_completed(futures):
            idx = futures[future]
            rows, errs = future.result()
            absorb(idx, rows, errs)


def _finalize_file(rows: list[dict[str, Any]], path: Path, resume: bool) -> None:
    if resume and path.exists():
        existing = _read_existing_rows(path)
        # Dedup on (key, method): a later run adding a different method to the same file
        # must not be discarded as an already-seen key.
        seen = {(r.get("key"), r.get("method")) for r in existing}
        merged = existing + [
            r for r in rows if (r.get("key"), r.get("method")) not in seen
        ]
        write_rows(merged, path)
    else:
        write_rows(rows, path)


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
            return rows
        import csv

        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception:  # noqa: BLE001 - a corrupt prior file should not abort resume
        warnings.warn(f"could not read existing sink {path}; overwriting", stacklevel=2)
        return []


__all__ = ["BatchSummary", "batch_periodograms"]
