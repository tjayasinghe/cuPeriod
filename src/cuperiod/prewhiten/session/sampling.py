"""Which points (and weights and subset codes) a fit of a session state uses.

The session never removes a point from its data: a state names deleted points (its
mask) and excluded subsets, and a partition may supply relative weights. This module
turns a state into the arrays one fit and one spectrum engine see.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import IntArray
from cuperiod.prewhiten.session.state import SessionData, SessionState, UsedData
from cuperiod.prewhiten.subsets import Partition


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def used_mask(data: SessionData, state: SessionState) -> npt.NDArray[np.bool_]:
    """``(N,)`` booleans: not deleted and not in an excluded subset."""
    used = ~state.mask.deleted
    for name, subsets in state.excluded:
        if subsets:
            partition = state.partition(name)
            used = used & ~np.isin(partition.codes, np.asarray(subsets, dtype=np.int32))
    return np.asarray(used, dtype=bool)


def compact_codes(
    partition: Partition, index: IntArray
) -> tuple[npt.NDArray[np.int32], tuple[str, ...], IntArray]:
    """Codes of the points ``index`` renumbered ``0..S'-1`` over the non-empty subsets.

    Returns
    -------
    codes : numpy.ndarray
        ``int32`` compacted code per point in use.
    labels : tuple of str
        Label of each compacted subset.
    subsets : numpy.ndarray
        Index of each compacted subset in ``partition``.
    """
    codes = partition.codes[index]
    present = np.unique(codes).astype(np.int64)
    lookup = np.full(partition.n_subsets, -1, dtype=np.int32)
    lookup[present] = np.arange(present.size, dtype=np.int32)
    compacted = _readonly(lookup[codes].astype(np.int32))
    labels = tuple(partition.labels[int(i)] for i in present)
    return compacted, labels, _readonly(present)


def used_data(data: SessionData, state: SessionState) -> UsedData:
    """The points, weights and compacted group codes a fit of ``state`` uses.

    With every point in use and no subset weights, ``time`` / ``value`` / ``error``
    are the :class:`SessionData` arrays themselves (only group codes are added), so
    a fit of such a state passes the very arrays :func:`cuperiod.prewhiten` would.

    Subset weights reach the fit as ``point_weights = omega`` (normalised to mean 1
    over the points in use) next to the unchanged errors, and the spectrum engine and
    the false-alarm probability as ``error_eff = error / sqrt(omega)`` (or
    ``1 / sqrt(omega)`` without error bars): both see ``w_i = omega_i / sigma_i^2``.
    """
    used = used_mask(data, state)
    full = bool(used.all())
    if full:
        index = np.arange(data.n, dtype=np.int64)
        time, value, error = data.time, data.value, data.error
    else:
        index = np.flatnonzero(used).astype(np.int64)
        time = data.time[index]
        value = data.value[index]
        error = None if data.error is None else data.error[index]
    _readonly(index)

    point_weights = None
    error_eff = error
    if state.weights_by is not None:
        partition = state.partition(state.weights_by)
        omega = np.asarray(partition.weights, dtype=np.float64)[partition.codes[index]]
        omega = omega / float(np.mean(omega)) if omega.size else omega
        point_weights = _readonly(omega)
        scale = 1.0 / np.sqrt(omega)
        error_eff = _readonly(scale if error is None else error * scale)

    offset_codes = variation_codes = None
    offset_labels: tuple[str, ...] = ()
    variation_labels: tuple[str, ...] = ()
    offset_subsets = variation_subsets = None
    if state.offsets_by is not None:
        offset_codes, offset_labels, offset_subsets = compact_codes(
            state.partition(state.offsets_by), index
        )
    if state.variation_by is not None:
        variation_codes, variation_labels, variation_subsets = compact_codes(
            state.partition(state.variation_by), index
        )
    return UsedData(
        time=time,
        value=value,
        error=error,
        error_eff=error_eff,
        point_weights=point_weights,
        index=index,
        used=_readonly(np.array(used, dtype=bool)),
        offset_codes=offset_codes,
        offset_labels=offset_labels,
        variation_codes=variation_codes,
        variation_labels=variation_labels,
        offset_subsets=offset_subsets,
        variation_subsets=variation_subsets,
    )


__all__ = ["compact_codes", "used_data", "used_mask"]
