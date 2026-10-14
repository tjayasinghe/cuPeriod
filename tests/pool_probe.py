"""Chunk functions that report a pool worker's thread configuration.

Importable by spawned workers (the test directory is on their ``sys.path``), so a
test can stand them in for a batch driver's chunk function and read back, as
ordinary result rows, what each worker process was configured with.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

#: The math-library variables the batch drivers pin for CPU pools.
BLAS_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def thread_state() -> dict[str, Any]:
    """This process's thread caps: numba's pool size and the BLAS variables."""
    import numba

    state: dict[str, Any] = {
        "pid": os.getpid(),
        "numba_threads": int(numba.get_num_threads()),
    }
    state.update({var: os.environ.get(var) for var in BLAS_VARS})
    return state


def rows_chunk(
    items: Sequence[tuple[str, Any]], cfg: Any
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The ``(rows, errors)`` shape of batch_periodograms / batch_prewhiten."""
    return [{"key": key, **thread_state()} for key, _ in items], []


def tables_chunk(
    items: Sequence[tuple[str, Any]], cfg: Any
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, str]]]:
    """The ``({table: rows}, errors)`` shape of batch_pdot."""
    summary = [{"key": key, "status": "ok", **thread_state()} for key, _ in items]
    return {"summary": summary, "oc": [], "profile": []}, []


def failing_chunk(items: Sequence[tuple[str, Any]], cfg: Any) -> Any:
    """A chunk whose worker raises: the pool fails mid-run."""
    raise RuntimeError("pool probe failure")
