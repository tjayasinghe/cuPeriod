"""Batch pre-whitening: :func:`batch_prewhiten`.

Runs the whole extraction loop over many light curves with the same worker machinery as
:func:`cuperiod.batch_periodograms` — a spawned CPU process pool, or GPU workers — and
flattens the result to **one row per extracted component**, which is the shape a
frequency catalogue wants: filter on ``snr``, group by ``key``, join on ``label``. A
light curve that yields no component still contributes one row, so nothing silently
disappears from the catalogue.

Directory sinks are resumable exactly as in the periodogram batch: a re-run skips chunks
whose part file already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuperiod.batch.io import InputItem, load_source, resolve_inputs, write_rows
from cuperiod.batch.runner import (
    BatchSummary,
    _chunked,
    _classify_sink,
    _dir_manifest_guard,
    _finalize_file,
    _part_path,
    _pending_chunks,
    _run_pool,
)
from cuperiod.batch.sizing import cpu_worker_count, pin_worker_threads
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.device import free_gpu_memory, suggest_gpu_workers
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.prewhiten.engine import prewhiten
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid


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


def batch_prewhiten(
    inputs: Any,
    *,
    settings: PreWhitenSettings | None = None,
    backend: str = "auto",
    device: str = "cpu",
    workers: int | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
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
    backend : str, default "auto"
        Amplitude-spectrum backend.
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
        Skip chunks already written (directory sink).
    chunk_size : int, default 64
        Light curves per chunk/task.

    Returns
    -------
    BatchSummary
        ``methods`` is reported as ``("PREWHITEN",)``.

    Examples
    --------
    >>> cup.batch_prewhiten("lightcurves/*.csv", sink="modes.parquet")  # doctest: +SKIP
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    # Catalogue rows never carry spectra, so keeping them would only make each worker
    # hold (and ship nothing from) megabytes of grid arrays per star: force them off.
    cfg = _ChunkConfig(
        settings=(settings or PreWhitenSettings()).model_copy(
            update={"store_spectra": False}
        ),
        backend="gpu" if device == "gpu" else backend,
        columns=columns,
        domain=domain,
        max_components=max_components,
    )

    sink_kind, sink_dir, sink_file = _classify_sink(sink)
    items = resolve_inputs(inputs, columns=columns, domain=domain)
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

    if device == "gpu":
        n_workers = workers if workers is not None else suggest_gpu_workers("GLS")
        if n_workers <= 1:
            for idx in pending:
                absorb(idx, *_process_chunk(chunks[idx], cfg))
                free_gpu_memory()
        else:
            _run_pool(chunks, pending, cfg, absorb, n_workers, worker=_process_chunk)
    else:
        n_workers = cpu_worker_count(workers)
        if n_workers <= 1:
            for idx in pending:
                absorb(idx, *_process_chunk(chunks[idx], cfg))
        else:
            pin_worker_threads()
            _run_pool(chunks, pending, cfg, absorb, n_workers, worker=_process_chunk)

    if sink_kind == "file":
        _finalize_file(all_rows, sink_file, resume)

    return BatchSummary(
        n_inputs=len(items),
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
