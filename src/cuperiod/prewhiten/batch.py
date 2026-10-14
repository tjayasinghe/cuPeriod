"""Batch pre-whitening: :func:`batch_prewhiten`.

Runs the whole extraction loop over many light curves with the same worker machinery as
:func:`cuperiod.batch_periodograms` — a spawned CPU process pool, or GPU workers — and
flattens the result to **one row per extracted component**, which is the shape a
frequency catalogue wants: filter on ``snr``, group by ``key``, join on ``label``. A
light curve that yields no component still contributes one row, so nothing silently
disappears from the catalogue.

Directory sinks are resumable exactly as in the periodogram batch. A file sink is
resumable per light curve too: a re-run skips every key the file already holds and
appends only the new ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuperiod.batch.io import (
    InputItem,
    csv_safe,
    load_source,
    resolve_inputs,
    write_rows,
)
from cuperiod.batch.runner import (
    BatchSummary,
    _chunked,
    _classify_sink,
    _dir_manifest_guard,
    _part_path,
    _pending_chunks,
    _read_existing_rows,
    _run_pool,
)
from cuperiod.batch.sizing import (
    cpu_worker_count,
    init_worker_threads,
    numba_threads_per_worker,
    pinned_worker_threads,
)
from cuperiod.core.backend import cpu_pool_backend
from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.prewhiten.engine import prewhiten
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid

#: Tasks per worker the CPU pool aims for when chunks are free to shrink (no
#: directory sink pins their layout): enough that one slow light curve does not leave
#: the other workers idle at the end, few enough that per-task overhead stays small.
_TASKS_PER_WORKER = 4


@dataclass(frozen=True)
class _ChunkConfig:
    """Immutable, picklable settings shared by every chunk task."""

    settings: PreWhitenSettings
    backend: str
    columns: ColumnMap | None
    domain: Domain | None
    max_components: int | None


def prewhiten_to_rows(
    key: str, result: PreWhitenResult, *, max_components: int | None = None
) -> list[dict[str, Any]]:
    """Flatten a solution to catalogue rows — one per component (or one empty row).

    Every row repeats the per-star summary (sample count, baseline, stop reason, fit
    statistics) so a single table is self-describing without a join.

    Parameters
    ----------
    key : str
        The light curve's batch key.
    result : PreWhitenResult
        The solution to flatten.
    max_components : int, optional
        Keep only the first ``max_components`` components.

    Returns
    -------
    list of dict
    """
    summary = {
        "key": key,
        "n_components": result.n_components,
        "n_pruned": result.n_pruned,
        "n_samples": result.n_samples,
        "baseline": result.baseline,
        "rayleigh": result.rayleigh,
        "t_ref": result.t_ref,
        "offset": result.offset,
        "stop_reason": result.stop_reason,
        "rms": result.rms,
        "reduced_chi2": result.reduced_chi2,
        "bic": result.bic,
        "correlation_factor": result.correlation_factor,
        "backend": result.backend,
    }
    components = result.components
    if max_components is not None:
        components = components[:max_components]
    if not components:
        return [{**summary, **dict(_EMPTY_COMPONENT)}]
    return [{**summary, **_component_cells(component)} for component in components]


#: Numeric component fields, in row order. Kept float (``rank`` included) and NaN-filled
#: rather than ``None`` so every chunk of a directory sink infers the *same* Arrow type:
#: a part in which no star had a combination would otherwise type that column null,
#: making the whole dataset unreadable.
#: ``blended`` is deliberately absent: a boolean has no missing value to fill for a
#: light curve that yielded nothing, and ``amplitude_ratio`` carries the same
#: information with a threshold the catalogue's consumer picks.
_NUMERIC_FIELDS: tuple[str, ...] = (
    "rank", "frequency", "frequency_error", "period", "period_error",
    "amplitude", "amplitude_error", "spectrum_amplitude", "amplitude_ratio",
    "phase", "phase_error", "snr", "fap", "delta_bic",
)

#: String component fields, empty-string-filled for the same reason.
_TEXT_FIELDS: tuple[str, ...] = ("label", "combination")

#: The all-missing component block used for a light curve that yielded nothing.
_EMPTY_COMPONENT: dict[str, Any] = {
    **dict.fromkeys(_NUMERIC_FIELDS, float("nan")),
    **dict.fromkeys(_TEXT_FIELDS, ""),
}


def _component_cells(component: Sinusoid) -> dict[str, Any]:
    """One component as stably-typed row cells (see :data:`_NUMERIC_FIELDS`)."""
    values = component.to_dict()
    cells: dict[str, Any] = {
        name: float(values[name]) for name in _NUMERIC_FIELDS
    }
    cells.update({name: values[name] or "" for name in _TEXT_FIELDS})
    return cells


def _process_chunk(
    items: Sequence[InputItem], cfg: _ChunkConfig
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Pre-whiten every light curve in a chunk; collect rows and errors."""
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    for key, source in items:
        try:
            lc = load_source(source, columns=cfg.columns, domain=cfg.domain)
            if isinstance(lc, MultiBandLightCurve):
                raise ValueError("pre-whitening needs single-band light curves")
            result = prewhiten(lc, settings=cfg.settings, backend=cfg.backend)
            rows.extend(
                prewhiten_to_rows(key, result, max_components=cfg.max_components)
            )
        except Exception as exc:  # one bad light curve must not kill the batch
            errors.append((key, f"{type(exc).__name__}: {exc}"))
    return rows, errors


def _pool_chunk_size(n_items: int, chunk_size: int, n_workers: int) -> int:
    """Light curves per task for a CPU pool whose chunk layout is not pinned.

    One task per chunk means ``ceil(n / chunk_size)`` tasks: a batch smaller than
    ``chunk_size`` ran on a single worker however many were asked for. Aim for
    :data:`_TASKS_PER_WORKER` tasks per worker instead, never above ``chunk_size``.
    """
    per_task = math.ceil(n_items / (_TASKS_PER_WORKER * max(1, n_workers)))
    return max(1, min(chunk_size, per_task))


def batch_prewhiten(
    inputs: Any,
    *,
    settings: PreWhitenSettings | None = None,
    backend: str | None = None,
    device: str = "cpu",
    workers: int | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
    sink: str | Path | None = None,
    max_components: int | None = None,
    resume: bool = True,
    chunk_size: int = 64,
) -> BatchSummary:
    """Pre-whiten many light curves.

    Parameters
    ----------
    inputs : various
        Anything :func:`cuperiod.batch_periodograms` accepts: an iterable of light
        curves or ``(key, lc)`` pairs, a glob string, a directory, or a
        ``(DataFrame, group_column)`` tuple.
    settings : PreWhitenSettings, optional
        Extraction settings applied to every light curve. ``store_spectra`` is forced
        off — catalogue rows never carry spectra, so keeping them would only waste
        worker memory.
    backend : str, optional
        Amplitude-spectrum backend. ``None`` (the default) takes
        ``settings.backend`` (also settable as ``CUPERIOD_PREWHITEN_BACKEND``), as
        :func:`cuperiod.prewhiten` does. With ``device="cpu"``, ``"auto"`` means the
        best *CPU* backend — a CPU pool never opens a CUDA context per worker — and
        a GPU-only backend (``"gpu"``, ``"cufinufft"``) raises ``ValueError``.
        ``device="gpu"`` always requests the GPU.
    device : {"cpu", "gpu"}, default "cpu"
        Where to run the workers.
    workers : int, optional
        Worker count; ``None`` picks all-but-one core (CPU) or a GPU-sized default.
    columns, domain : optional
        Column mapping and brightness-domain handling for file and table inputs.
    sink : str or Path, optional
        ``.parquet``/``.csv`` file, or a directory (resumable, one part per chunk).
        ``None`` returns the rows in memory.
    max_components : int, optional
        Store only this many components per light curve.
    resume : bool, default True
        Skip light curves already written: those a directory sink's part files hold,
        or whose key a file sink already holds (rows of a key are never replaced, so
        re-run with ``resume=False`` after changing settings). A light curve that
        failed wrote no row and is retried.
    chunk_size : int, default 64
        Light curves per chunk/task. A directory sink uses exactly this (its part
        files are numbered by chunk); a resumed one may use a different size, since
        it skips the keys already written whatever part holds them. Otherwise a CPU
        pool treats it as an upper bound and shrinks chunks so every worker gets
        several tasks.

    Returns
    -------
    BatchSummary
        ``methods`` is reported as ``("PREWHITEN",)``. ``n_done`` counts the
        component rows computed in this run (for a file sink, the rows appended),
        ``n_skipped`` the light curves resume left alone. Rows and errors are in
        input order.

    Raises
    ------
    ValueError
        For an unknown ``device``, or a GPU-only backend with ``device="cpu"``.

    Examples
    --------
    >>> cup.batch_prewhiten("lightcurves/*.csv", sink="modes.parquet")  # doctest: +SKIP
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    # A misspelt domain is never read for in-memory inputs, so it would otherwise
    # surface only on the first file or table input, deep in a worker.
    domain = coerce_domain(domain)
    base = settings if settings is not None else PreWhitenSettings()
    # The backend handed to every worker. An explicit request wins, else the settings'
    # pin (prewhiten() itself does `backend or settings.backend`, so forwarding the old
    # truthy default 'auto' silently discarded it). On a CPU pool 'auto' must become
    # 'cpu': resolved inside each worker it picks cufinufft on a CUDA box, one GPU
    # context per core.
    worker_backend = "gpu" if device == "gpu" else cpu_pool_backend(backend, base)
    # Catalogue rows never carry spectra, so keeping them would only make each worker
    # hold (and ship nothing from) megabytes of grid arrays per star: force them off.
    cfg = _ChunkConfig(
        settings=base.model_copy(update={"store_spectra": False}),
        backend=worker_backend,
        columns=columns,
        domain=domain,
        max_components=max_components,
    )

    sink_kind, sink_dir, sink_file = _classify_sink(sink)
    items = resolve_inputs(inputs, columns=columns, domain=domain)
    n_inputs = len(items)

    # A file sink is merged at the end, keeping the rows already in it, so computing
    # a light curve the file already holds only threw the result away (while n_done
    # still counted it). Drop those keys before any work is scheduled.
    n_skipped = 0
    existing_rows: list[dict[str, Any]] = []
    if sink_kind == "file" and resume and sink_file.exists():
        existing_rows = _read_existing_rows(sink_file)
        done = {str(row.get("key")) for row in existing_rows}
        # A CSV sink stores text cells formula-escaped; compare keys as stored.
        csv_sink = sink_file.suffix.lower() == ".csv"
        items = [
            item
            for item in items
            if (str(csv_safe(item[0])) if csv_sink else item[0]) not in done
        ]
        n_skipped = n_inputs - len(items)

    if device == "gpu":
        n_workers = workers if workers is not None else suggest_gpu_workers("GLS")
    else:
        n_workers = cpu_worker_count(workers)
    size = max(1, chunk_size)
    if device == "cpu" and n_workers > 1 and sink_kind != "dir":
        size = _pool_chunk_size(len(items), size, n_workers)
    chunks = _chunked(items, size)
    if sink_kind == "dir":
        # Resume is by key (_pending_chunks skips written keys and re-chunks the
        # rest past the last part), so a different chunk_size is safe and the
        # manifest no longer pins it, as in batch_periodograms.
        _dir_manifest_guard(sink_dir, None, resume)
    pending = _pending_chunks(chunks, sink_kind, sink_dir, resume)
    n_skipped += len(items) - sum(len(chunks[i]) for i in pending)

    # Results are buffered per chunk and flattened in chunk order at the end: a pool
    # finishes chunks in completion order, and rows/errors should follow the inputs.
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
            if rows:  # written at once, so an interrupted run keeps finished chunks
                write_rows(rows, _part_path(sink_dir, idx))
        else:
            rows_by_chunk[idx] = rows

    if n_workers <= 1 or not pending:
        for idx in pending:
            absorb(idx, *_process_chunk(chunks[idx], cfg))
            if device == "gpu":
                free_gpu_memory()
    elif device == "gpu":
        _run_pool(chunks, pending, cfg, absorb, n_workers, worker=_process_chunk)
    else:
        pool_size = min(n_workers, len(pending))
        # BLAS/OpenMP pinned to one thread only while the pool exists (the parent
        # session keeps its threads afterwards); numba gets its share per worker.
        with pinned_worker_threads():
            _run_pool(
                chunks,
                pending,
                cfg,
                absorb,
                pool_size,
                worker=_process_chunk,
                initializer=init_worker_threads,
                initargs=(numba_threads_per_worker(pool_size),),
            )

    all_rows = [row for idx in sorted(rows_by_chunk) for row in rows_by_chunk[idx]]
    errors = [err for idx in sorted(errors_by_chunk) for err in errors_by_chunk[idx]]

    # The kept rows first, then this run's (their keys never overlap: resume dropped
    # the known ones). An up-to-date file is left untouched.
    if sink_kind == "file" and (all_rows or not sink_file.exists()):
        write_rows([*existing_rows, *all_rows], sink_file)

    return BatchSummary(
        n_inputs=n_inputs,
        n_done=n_done,
        n_failed=len(errors),
        n_skipped=n_skipped,
        methods=("PREWHITEN",),
        device=device,
        sink=str(sink) if sink is not None else None,
        errors=errors,
        rows=None if sink_kind != "memory" else all_rows,
    )


__all__ = ["batch_prewhiten", "prewhiten_to_rows"]
