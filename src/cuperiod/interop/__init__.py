"""Interoperability with the LINCC Frameworks stack (nested-pandas / lsdb).

LINCC's survey tables keep a star's light curve *inside* its object row: one row per
object, the per-epoch arrays in a nested column (nested-pandas ``NestedFrame``), and
``lsdb`` spreads that frame over the dask partitions of a HATS catalog. This package
runs cuPeriod directly on that layout — no flattening, no ``groupby``, no per-object
DataFrames — in two tiers that share their column handling and output columns:

Tier 1, :func:`~cuperiod.interop.lincc.nested_periodogram` (row-wise)
    Wraps ``map_rows``: one light curve per row, one periodogram per row, results
    appended as ordinary base columns. The simple, composable default — right for a
    CPU backend, a quick look, or a small catalog.

Tier 2, :func:`~cuperiod.interop.lincc.partition_periodogram` (partition-wise)
    Reads a whole partition's nested column once through its Arrow buffers, slices
    each object out of the flat arrays, and evaluates the entire partition against
    **one** method engine built per partition. GPU plan/kernel setup is amortized over
    thousands of stars instead of paid per star — the throughput tier.

Both accept an in-memory ``NestedFrame`` or a lazy lsdb ``Catalog`` (which stays lazy),
resolve their columns from the nest's schema alone (never by computing), and turn a
failed object into NaN result columns rather than an aborted run.

This package is optional: ``pip install 'cuperiod[nested]'`` (add lsdb with
``'cuperiod[lsdb]'``). Importing :mod:`cuperiod` never pulls in nested-pandas, and
importing this package does not either — the dependency is checked on first use, so
even :data:`COLUMN_PRESETS` can be inspected without it.

Examples
--------
>>> from cuperiod.interop import nested_periodogram    # doctest: +SKIP
>>> out = nested_periodogram(frame, "lc", preset="ztf_dr22")   # doctest: +SKIP
"""

from __future__ import annotations

from cuperiod.interop.lincc import COLUMN_PRESETS as _COLUMN_PRESETS
from cuperiod.interop.lincc import (
    PRESET_KEYS,
    NestedColumns,
    nested_periodogram,
    partition_periodogram,
    require_nested_pandas,
    resolve_nested_columns,
)

#: Column presets for common survey layouts, keyed by preset name — currently
#: ``"ztf_dr22"``, ``"ztf_alerts"``, ``"rubin_dp1_object"``, and
#: ``"rubin_dp1_dia"``. Each value is the keyword mapping that
#: :func:`~cuperiod.interop.lincc.resolve_nested_columns` expands into a
#: :class:`~cuperiod.interop.lincc.NestedColumns`; pass the key as ``preset=``.
COLUMN_PRESETS = _COLUMN_PRESETS

__all__ = [
    "COLUMN_PRESETS",
    "PRESET_KEYS",
    "NestedColumns",
    "nested_periodogram",
    "partition_periodogram",
    "require_nested_pandas",
    "resolve_nested_columns",
]
