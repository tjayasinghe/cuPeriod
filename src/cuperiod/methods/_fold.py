"""Plumbing shared by the phase-fold statistics (PDM, CE, String-Length) and MHAOV.

Private to those four methods, which is why it lives here rather than in
:mod:`cuperiod.core`:

* :func:`fold_batch` -- the trial-period chunk length of a vectorized kernel, capped by
  a transient-memory budget so the ``(chunk, N)`` work arrays stay bounded however long
  the light curve is (a fixed 2048-period chunk needed ~4 GiB at 65k points).
* :func:`foldable_periods` / :func:`finite_angles` / :func:`evaluate_trials` -- which
  trial values can be evaluated at all (a period of 0 or NaN has no fold), and running
  a kernel on those only, NaN at the rest. A NaN that reached the compiled kernels
  either crashed them, leaked uninitialised memory (numba MHAOV) or read out of bounds
  (the CUDA string-length sort).
* :func:`fold_phase` -- ``(tau / p) mod 1`` over a chunk, in its cheapest exact form.
* :func:`usable_curve` -- the finite curve, or an error that names the real reason
  nothing is usable instead of the "no usable time baseline" an emptied curve gives.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from cuperiod.core._typing import FloatArray
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import LightCurve, raise_if_unusable

#: Transient byte budgets of one vectorized chunk: host (numpy, torch on the CPU) and
#: device (torch GPU devices, cupy). The same budgets MHAOV and SuperSmoother use.
FOLD_CHUNK_BYTES: Final = 1 << 27
FOLD_DEVICE_CHUNK_BYTES: Final = 1 << 29


def fold_batch(
    batch: int, n_points: int, *, workspaces: int, on_device: bool, default: int
) -> int:
    """Trial periods per vectorized chunk: ``batch`` capped by a memory budget.

    Each chunk of ``P`` periods holds about ``workspaces`` arrays of ``P x n_points``
    8-byte elements at once, so the chunk is capped at ``budget // (n_points * 8 *
    workspaces)`` (128 MiB on the host, 512 MiB on a device). Short curves keep the
    requested chunk (the cap only bites above ~1-3k points); ``batch <= 0`` means
    ``default``. The chunk length never changes a result, only peak memory.

    Parameters
    ----------
    batch : int
        Requested trial periods per chunk (the settings' ``batch_periods``).
    n_points : int
        Light-curve length.
    workspaces : int
        ``(chunk, N)`` arrays the method's kernel keeps alive at its peak.
    on_device : bool
        Whether the kernel runs on a GPU device (larger budget).
    default : int
        Chunk used when ``batch <= 0``.

    Returns
    -------
    int
        The effective chunk length, at least 1.
    """
    requested = int(batch) if batch > 0 else int(default)
    budget = FOLD_DEVICE_CHUNK_BYTES if on_device else FOLD_CHUNK_BYTES
    cap = budget // max(1, int(n_points) * 8 * int(workspaces))
    return max(1, min(requested, cap))


def check_backend(backend: str) -> None:
    """Raise ``ValueError`` for a backend name none of the fold kernels knows.

    Checked up front, so the error does not depend on whether any trial is valid.
    """
    if backend in ("numpy", "numba", "cupy", "torch") or backend.startswith("torch:"):
        return
    raise ValueError(f"unknown backend {backend!r}")


def foldable_periods(tau_max: float, periods: FloatArray) -> NDArray[np.bool_]:
    """Mask of trial periods whose fold ``tau / p`` is finite for every point.

    ``tau`` runs from 0 to ``tau_max``, so ``tau / p`` is finite for all points
    exactly when ``tau_max / p`` is: a period of 0, a NaN, or one so small that the
    quotient overflows fails; an infinite period (frequency 0) folds every point to
    phase 0 and passes.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.asarray(np.isfinite(float(tau_max) / periods))


def finite_angles(tau_max: float, frequencies: FloatArray) -> NDArray[np.bool_]:
    """Mask of trial frequencies whose angles ``2 pi f tau`` are all finite."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.asarray(np.isfinite((2.0 * np.pi * float(tau_max)) * frequencies))


def evaluate_trials(
    compute: Callable[[FloatArray], FloatArray],
    trials: FloatArray,
    ok: NDArray[np.bool_],
) -> FloatArray:
    """``compute(trials)`` with the entries where ``ok`` is False set to NaN.

    The usual all-valid case calls ``compute`` on ``trials`` itself (no copy). The
    kernels evaluate every trial independently, so the values at the valid entries
    are those of a clean grid.
    """
    if bool(ok.all()):
        return compute(trials)
    out = np.full(trials.shape[0], np.nan, dtype=np.float64)
    if bool(ok.any()):
        out[ok] = compute(np.ascontiguousarray(trials[ok]))
    return out


def fold_phase(xp: ModuleType, tau: Any, periods: Any) -> Any:
    """Folded phase ``(tau / p) mod 1`` of every (period, point) pair, ``(P, N)``.

    On numpy the fold is ``q - floor(q)``, done in place: bit-identical to
    ``remainder(q, 1)`` (including NaN/inf and negative ``q``) and to the numba
    kernels, which use the same formula, at a third of the cost per element and with
    no temporary beyond the floor. Device namespaces keep ``remainder``, one fused
    kernel there (on torch CPU the floor form is not reliably faster).
    """
    q = tau[None, :] / periods[:, None]
    if isinstance(q, np.ndarray):
        whole = np.floor(q)
        np.subtract(q, whole, out=q)
        return q
    return xp.remainder(q, 1.0)


def row_offsets(n_rows: int, n_cols: int) -> NDArray[np.int64]:
    """``(n_rows, 1)`` offsets that turn per-row bin indices into one flat index."""
    return (np.arange(n_rows, dtype=np.int64) * int(n_cols))[:, None]


def usable_curve(lc: LightCurve, label: str) -> LightCurve:
    """``lc.finite()``, or an :class:`InsufficientDataError` naming why it is empty.

    The fold methods do not use the errors, but :meth:`LightCurve.finite` drops every
    point whose error is zero, negative or non-finite, so a placeholder error column
    used to surface as "no usable time baseline". This raises that real cause (via
    :func:`~cuperiod.core.lightcurve.raise_if_unusable`), or that no point has a
    finite time and value; otherwise it returns the finite curve.
    """
    raise_if_unusable(lc, label)
    finite = lc.finite()
    if finite.n == 0 and lc.n > 0:
        raise InsufficientDataError(
            f"{label}: none of the {lc.n} points has a finite time and value"
        )
    return finite


__all__ = [
    "FOLD_CHUNK_BYTES",
    "FOLD_DEVICE_CHUNK_BYTES",
    "check_backend",
    "evaluate_trials",
    "finite_angles",
    "fold_batch",
    "fold_phase",
    "foldable_periods",
    "row_offsets",
    "usable_curve",
]
