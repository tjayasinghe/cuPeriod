"""Partitions of a light curve's points into subsets: nights, sectors, seasons, bands.

Period04 attaches "attributes" to every point — the night, the observatory, the
observer — and fits one zero point per value, weights the points per value, or lets an
amplitude change from one value to the next. A :class:`Partition` is one such
attribute: a name, an integer code per point, one label and one relative weight per
subset, and the recipe (:data:`PartitionOrigin`) that produced it.

The codes are the fact; the origin is how they were derived. A partition built by a
recipe (:meth:`Partition.from_gaps`, :meth:`~Partition.from_chunks`,
:meth:`~Partition.from_labels`, :meth:`~Partition.from_ranges`,
:meth:`~Partition.crossed`) carries that recipe so a script can rebuild it with the
same call; every hand edit (:meth:`~Partition.merged`, :meth:`~Partition.split_at`,
...) yields a :class:`ManualOrigin` partition whose runs (:meth:`~Partition.runs`,
:meth:`~Partition.from_runs`) are the only faithful record.

Pure numpy; nothing here knows about fits or sessions. Points may come in any time
order: every derivation works on the time-sorted view and maps back.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from cuperiod.core._typing import FloatArray, IntArray

#: ``(N,)`` subset codes.
CodeArray = npt.NDArray[np.int32]

#: A point mask.
BoolArray = npt.NDArray[np.bool_]

#: Longest subset label.
MAX_LABEL_LENGTH: Final = 64

#: Most edges a ranges recipe may carry.
MAX_RANGE_EDGES: Final = 1000

#: Gap above which consecutive points belong to different nights (days), and the
#: longest span a "night" may have for :func:`suggest_partition` to call the sampling
#: nightly.
_NIGHT_GAP: Final = 0.3
_NIGHT_MAX_SPAN: Final = 0.75

#: Gap that separates observing seasons (days).
_SEASON_GAP: Final = 60.0

#: Sector recipe: TESS sectors are ~27 d with a ~1 d mid-sector downlink gap.
_SECTOR_GAP: Final = 0.5
_SECTOR_MAX_SPAN: Final = 28.5


# -- origins -------------------------------------------------------------------------


class _Origin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GapsOrigin(_Origin):
    """A new subset at every time gap longer than ``min_gap`` (see ``from_gaps``)."""

    kind: Literal["gaps"] = "gaps"
    min_gap: float = Field(default=0.5, gt=0.0, allow_inf_nan=False)
    max_span: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    min_points: int = Field(default=10, ge=1)
    prefix: str = Field(default="N", max_length=8)


class ChunksOrigin(_Origin):
    """Fixed-length chunks of time (see ``from_chunks``)."""

    kind: Literal["chunks"] = "chunks"
    length: float = Field(gt=0.0, allow_inf_nan=False)
    origin: float | None = Field(default=None, allow_inf_nan=False)
    min_points: int = Field(default=10, ge=1)
    prefix: str = Field(default="C", max_length=8)


class LabelsOrigin(_Origin):
    """One subset per distinct value of a per-point label column."""

    kind: Literal["labels"] = "labels"
    column: str = Field(min_length=1, max_length=64)


class RangesOrigin(_Origin):
    """Time ranges split at ``edges`` (see ``from_ranges``)."""

    kind: Literal["ranges"] = "ranges"
    edges: list[float] = Field(max_length=MAX_RANGE_EDGES)
    labels: list[str] | None = None

    @field_validator("edges")
    @classmethod
    def _sorted(cls, edges: list[float]) -> list[float]:
        values = np.asarray(edges, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("range edges must be finite")
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError("range edges must be strictly increasing")
        return edges


class CrossedOrigin(_Origin):
    """Every non-empty combination of two partitions' subsets (``a.crossed(b)``)."""

    kind: Literal["crossed"] = "crossed"
    a: str = Field(min_length=1, max_length=64)
    b: str = Field(min_length=1, max_length=64)


class ManualOrigin(_Origin):
    """Edited by hand: the partition's runs are its only record."""

    kind: Literal["manual"] = "manual"


#: How a partition was derived; one discriminated union on ``kind``.
PartitionOrigin = Annotated[
    GapsOrigin | ChunksOrigin | LabelsOrigin | RangesOrigin | CrossedOrigin
    | ManualOrigin,
    Field(discriminator="kind"),
]

_ORIGIN_ADAPTER: TypeAdapter[PartitionOrigin] = TypeAdapter(PartitionOrigin)


def parse_origin(raw: Mapping[str, Any]) -> PartitionOrigin:
    """Validate a plain mapping (e.g. from a project file) into an origin.

    Raises
    ------
    pydantic.ValidationError
        For an unknown ``kind``, an unknown key or an out-of-range value.
    """
    return _ORIGIN_ADAPTER.validate_python(dict(raw))


# -- helpers -------------------------------------------------------------------------


def _as_time(time: Any) -> FloatArray:
    t = np.ascontiguousarray(np.asarray(time, dtype=np.float64)).ravel()
    if not np.all(np.isfinite(t)):
        raise ValueError("partition times must be finite")
    return t


def _check_label(label: Any) -> str:
    if not isinstance(label, str) or not label or len(label) > MAX_LABEL_LENGTH:
        raise ValueError(
            f"subset labels must be non-empty strings of at most {MAX_LABEL_LENGTH} "
            f"characters, not {label!r}"
        )
    return label


def _groups_to_codes(groups: Sequence[Sequence[int]], codes: IntArray) -> CodeArray:
    """Map old codes to the index of the group holding them."""
    lookup = np.empty(int(codes.max()) + 1 if codes.size else 0, dtype=np.int32)
    for new, members in enumerate(groups):
        for old in members:
            lookup[old] = new
    out: CodeArray = lookup[codes].astype(np.int32)
    return out


def _merge_small(
    codes: IntArray, time: FloatArray, n_subsets: int, min_points: int
) -> list[list[int]]:
    """Groups of codes after merging every small subset into a time neighbour.

    Subsets are ordered by their first time; a subset with fewer than ``min_points``
    points (smallest first) is merged into the neighbour across the smaller gap
    between their spans, until every subset is large enough or one remains. Empty
    subsets are dropped. Returns the groups in time order.
    """
    counts = np.bincount(codes, minlength=n_subsets)
    present = [int(s) for s in np.flatnonzero(counts)]
    first = np.full(n_subsets, np.inf)
    last = np.full(n_subsets, -np.inf)
    np.minimum.at(first, codes, time)
    np.maximum.at(last, codes, time)
    present.sort(key=lambda s: (first[s], s))
    groups = [[s] for s in present]
    sizes = [int(counts[s]) for s in present]
    starts = [float(first[s]) for s in present]
    ends = [float(last[s]) for s in present]
    while len(groups) > 1:
        small = [i for i, size in enumerate(sizes) if size < min_points]
        if not small:
            break
        i = min(small, key=lambda j: (sizes[j], j))
        left_gap = starts[i] - ends[i - 1] if i > 0 else np.inf
        right_gap = starts[i + 1] - ends[i] if i + 1 < len(groups) else np.inf
        j = i - 1 if left_gap <= right_gap else i + 1
        lo, hi = min(i, j), max(i, j)
        groups[lo] = groups[lo] + groups[hi]
        sizes[lo] += sizes[hi]
        starts[lo] = min(starts[lo], starts[hi])
        ends[lo] = max(ends[lo], ends[hi])
        del groups[hi], sizes[hi], starts[hi], ends[hi]
    return groups


def _sequential(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{i + 1}" for i in range(count))


# -- the partition ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Partition:
    """A named split of the points into disjoint subsets (Period04's "attribute").

    Compares by identity; compare :attr:`codes`, :attr:`labels` and :attr:`weights`
    for equal content.

    Attributes
    ----------
    name : str
        ``"night"``, ``"sector"``, ``"band"``, ``"site"``, ...
    codes : numpy.ndarray
        ``(N,)`` ``int32`` codes in ``[0, S)``, aligned with the points (read-only).
    labels : tuple of str
        One unique label per subset.
    weights : tuple of float
        Relative weight per subset (default 1.0).
    origin : PartitionOrigin
        How the partition was derived; the codes stay authoritative.
    """

    name: str
    codes: CodeArray
    labels: tuple[str, ...]
    weights: tuple[float, ...]
    origin: PartitionOrigin

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ValueError("a partition needs a name of 1-64 characters")
        labels = tuple(_check_label(label) for label in self.labels)
        if len(set(labels)) != len(labels):
            raise ValueError(f"partition {self.name!r} has duplicate subset labels")
        raw = np.asarray(self.codes)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError("partition codes must be a 1-D integer array")
        if raw.size and (int(raw.min()) < 0 or int(raw.max()) >= len(labels)):
            raise ValueError(
                f"partition codes must lie in [0, {len(labels)}) — one label per subset"
            )
        codes = np.array(raw, dtype=np.int32, copy=True)
        codes.setflags(write=False)
        weights = tuple(float(w) for w in self.weights)
        if len(weights) != len(labels):
            raise ValueError("a partition needs one weight per subset")
        if not all(np.isfinite(w) and w > 0.0 for w in weights):
            raise ValueError("subset weights must be finite and positive")
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "weights", weights)

    # -- shape -----------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.codes.size)

    @property
    def n_subsets(self) -> int:
        return len(self.labels)

    def counts(self, used: BoolArray | None = None) -> IntArray:
        """Points per subset (only the ``used`` ones when a mask is given)."""
        codes = self.codes if used is None else self.codes[self._mask(used)]
        out: IntArray = np.bincount(codes, minlength=self.n_subsets).astype(np.int64)
        return out

    def spans(self, time: Any, used: BoolArray | None = None) -> FloatArray:
        """``(S, 2)`` first and last time per subset; NaN rows for empty subsets."""
        t = self._time(time)
        codes = self.codes
        if used is not None:
            mask = self._mask(used)
            t, codes = t[mask], codes[mask]
        first = np.full(self.n_subsets, np.inf)
        last = np.full(self.n_subsets, -np.inf)
        np.minimum.at(first, codes, t)
        np.maximum.at(last, codes, t)
        out: FloatArray = np.column_stack([first, last])
        out[~np.isfinite(first)] = np.nan
        return out

    def runs(self) -> list[tuple[int, int, int]]:
        """Maximal runs of equal codes in point order: ``(start, stop, code)``."""
        if self.codes.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(self.codes) != 0) + 1
        starts = np.concatenate([[0], breaks])
        stops = np.concatenate([breaks, [self.codes.size]])
        return [
            (int(a), int(b), int(self.codes[a]))
            for a, b in zip(starts, stops, strict=True)
        ]

    # -- builders --------------------------------------------------------------------
    @classmethod
    def from_runs(
        cls,
        n: int,
        runs: Sequence[Sequence[int]],
        *,
        name: str,
        labels: Sequence[str],
        weights: Sequence[float] | None = None,
        origin: PartitionOrigin | Mapping[str, Any] | None = None,
    ) -> Partition:
        """Rebuild a partition from :meth:`runs` (the faithful record of any edit).

        ``origin`` may be given as an origin model or as its plain mapping (what
        ``origin.model_dump(mode="json")`` and a project file hold).

        Raises
        ------
        ValueError
            Unless the runs are contiguous, in order and cover ``[0, n)`` exactly,
            with every code in ``[0, len(labels))``.
        pydantic.ValidationError
            For an ``origin`` mapping that is not a known origin.
        """
        if origin is None:
            recipe: PartitionOrigin = ManualOrigin()
        elif isinstance(origin, Mapping):
            recipe = parse_origin(origin)
        else:
            recipe = origin
        codes = np.empty(int(n), dtype=np.int32)
        position = 0
        for run in runs:
            if len(run) != 3:
                raise ValueError("a run is (start, stop, code)")
            start, stop, code = (int(v) for v in run)
            if start != position or stop <= start or stop > n:
                raise ValueError(
                    f"runs must tile [0, {n}) in order; got ({start}, {stop}) at "
                    f"{position}"
                )
            # Checked before the int32 store, which would overflow on a huge code.
            if not 0 <= code < len(labels):
                raise ValueError(
                    f"partition codes must lie in [0, {len(labels)}) — one label per "
                    f"subset; got {code}"
                )
            codes[start:stop] = code
            position = stop
        if position != n:
            raise ValueError(f"runs cover [0, {position}), not [0, {n})")
        return cls(
            name=name,
            codes=codes,
            labels=tuple(labels),
            weights=tuple(weights) if weights is not None else (1.0,) * len(labels),
            origin=recipe,
        )

    @classmethod
    def from_gaps(
        cls,
        time: Any,
        *,
        min_gap: float = 0.5,
        max_span: float | None = None,
        min_points: int = 10,
        name: str = "night",
        prefix: str = "N",
    ) -> Partition:
        """A new subset at every gap longer than ``min_gap`` days.

        With ``max_span``, consecutive gap-delimited runs are merged greedily while the
        merged span stays ``<= max_span`` (TESS sectors: ``min_gap=0.5,
        max_span=28.5``, no mission calendar needed). Subsets with fewer than
        ``min_points`` points are then merged into the neighbour across the smaller
        gap. Labels are ``prefix`` + 1, 2, ... in time order.
        """
        origin = GapsOrigin(
            min_gap=min_gap, max_span=max_span, min_points=min_points, prefix=prefix
        )
        t = _as_time(time)
        if t.size == 0:
            raise ValueError("cannot partition an empty light curve")
        order = np.argsort(t, kind="stable")
        ts = t[order]
        breaks = np.flatnonzero(np.diff(ts) > min_gap) + 1
        starts = np.concatenate([[0], breaks]).astype(np.int64)
        stops = np.concatenate([breaks, [ts.size]]).astype(np.int64)
        if max_span is not None:
            merged_starts = [int(starts[0])]
            merged_stops = [int(stops[0])]
            for a, b in zip(starts[1:], stops[1:], strict=True):
                if ts[b - 1] - ts[merged_starts[-1]] <= max_span:
                    merged_stops[-1] = int(b)
                else:
                    merged_starts.append(int(a))
                    merged_stops.append(int(b))
            starts = np.asarray(merged_starts, dtype=np.int64)
            stops = np.asarray(merged_stops, dtype=np.int64)
        sorted_codes = np.repeat(np.arange(starts.size), stops - starts)
        codes = np.empty(t.size, dtype=np.int64)
        codes[order] = sorted_codes
        groups = _merge_small(codes, t, int(starts.size), min_points)
        return cls(
            name=name,
            codes=_groups_to_codes(groups, codes),
            labels=_sequential(prefix, len(groups)),
            weights=(1.0,) * len(groups),
            origin=origin,
        )

    @classmethod
    def from_chunks(
        cls,
        time: Any,
        *,
        length: float,
        origin: float | None = None,
        min_points: int = 10,
        name: str = "chunk",
        prefix: str = "C",
    ) -> Partition:
        """Consecutive chunks of ``length`` days starting at ``origin`` (first time).

        Empty chunks are skipped and chunks with fewer than ``min_points`` points
        merged into the neighbour across the smaller gap.
        """
        recipe = ChunksOrigin(
            length=length, origin=origin, min_points=min_points, prefix=prefix
        )
        t = _as_time(time)
        if t.size == 0:
            raise ValueError("cannot partition an empty light curve")
        start = float(t.min()) if origin is None else float(origin)
        index = np.floor((t - start) / length).astype(np.int64)
        _, raw = np.unique(index, return_inverse=True)
        raw = raw.astype(np.int64).ravel()
        groups = _merge_small(raw, t, int(raw.max()) + 1, min_points)
        return cls(
            name=name,
            codes=_groups_to_codes(groups, raw),
            labels=_sequential(prefix, len(groups)),
            weights=(1.0,) * len(groups),
            origin=recipe,
        )

    @classmethod
    def from_labels(
        cls, values: Any, *, name: str, column: str | None = None
    ) -> Partition:
        """One subset per distinct value (sorted; labels are ``str(value)``).

        ``column`` names the label column for the recorded origin (defaults to
        ``name``).
        """
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("labels must be a 1-D array, one per point")
        if array.size == 0:
            raise ValueError("cannot partition an empty light curve")
        unique, inverse = np.unique(array, return_inverse=True)
        labels = tuple(str(v) for v in unique.tolist())
        if len(set(labels)) != len(labels):
            raise ValueError("distinct label values render to the same text")
        return cls(
            name=name,
            codes=inverse.astype(np.int32).ravel(),
            labels=labels,
            weights=(1.0,) * len(labels),
            origin=LabelsOrigin(column=column or name),
        )

    @classmethod
    def from_ranges(
        cls,
        time: Any,
        edges: Sequence[float],
        *,
        name: str = "range",
        labels: Sequence[str] | None = None,
    ) -> Partition:
        """Split the time axis at ``edges`` (``t < e0``, ``e0 <= t < e1``, ..., ``t >=
        e_last``).

        ``labels`` names the ``len(edges) + 1`` ranges (default ``R1``, ``R2``, ...);
        ranges holding no point are dropped with their labels.
        """
        recipe = RangesOrigin(
            edges=[float(e) for e in edges],
            labels=list(labels) if labels is not None else None,
        )
        t = _as_time(time)
        if t.size == 0:
            raise ValueError("cannot partition an empty light curve")
        names = (
            tuple(labels) if labels is not None else _sequential("R", len(edges) + 1)
        )
        if len(names) != len(edges) + 1:
            raise ValueError(
                f"{len(edges)} edges make {len(edges) + 1} ranges, not {len(names)}"
            )
        index = np.searchsorted(np.asarray(recipe.edges), t, side="right")
        used = np.unique(index)
        lookup = np.full(len(names), -1, dtype=np.int32)
        lookup[used] = np.arange(used.size, dtype=np.int32)
        return cls(
            name=name,
            codes=lookup[index],
            labels=tuple(names[i] for i in used.tolist()),
            weights=(1.0,) * int(used.size),
            origin=recipe,
        )

    @classmethod
    def from_origin(
        cls,
        origin: PartitionOrigin,
        time: Any,
        *,
        name: str,
        labels: Mapping[str, Any] | None = None,
        partitions: Mapping[str, Partition] | None = None,
    ) -> Partition:
        """Run a recipe: the partition ``origin`` describes, on these points.

        ``labels`` supplies the per-point columns a :class:`LabelsOrigin` names and
        ``partitions`` the partitions a :class:`CrossedOrigin` names.

        Raises
        ------
        ValueError
            For a :class:`ManualOrigin` (it has no recipe; use :meth:`from_runs`) or a
            missing column or partition.
        """
        if isinstance(origin, GapsOrigin):
            return cls.from_gaps(
                time, min_gap=origin.min_gap, max_span=origin.max_span,
                min_points=origin.min_points, name=name, prefix=origin.prefix,
            )
        if isinstance(origin, ChunksOrigin):
            return cls.from_chunks(
                time, length=origin.length, origin=origin.origin,
                min_points=origin.min_points, name=name, prefix=origin.prefix,
            )
        if isinstance(origin, LabelsOrigin):
            if labels is None or origin.column not in labels:
                raise ValueError(f"no label column {origin.column!r}")
            return cls.from_labels(labels[origin.column], name=name,
                                   column=origin.column)
        if isinstance(origin, RangesOrigin):
            return cls.from_ranges(time, origin.edges, name=name, labels=origin.labels)
        if isinstance(origin, CrossedOrigin):
            if partitions is None or origin.a not in partitions or (
                origin.b not in partitions
            ):
                raise ValueError(
                    f"crossing needs the partitions {origin.a!r} and {origin.b!r}"
                )
            return partitions[origin.a].crossed(partitions[origin.b], name=name)
        raise ValueError(
            "a manual partition has no recipe; rebuild it with Partition.from_runs"
        )

    # -- edits -----------------------------------------------------------------------
    def _edited(
        self, codes: Any, labels: Sequence[str], weights: Sequence[float]
    ) -> Partition:
        return Partition(
            name=self.name, codes=codes, labels=tuple(labels),
            weights=tuple(weights), origin=ManualOrigin(),
        )

    def crossed(self, other: Partition, *, name: str | None = None) -> Partition:
        """Every non-empty ``(self, other)`` combination (``site x night``).

        Subsets are ordered by ``(self code, other code)``, labelled ``"a/b"`` and
        weighted by the product of the two weights.
        """
        if other.n_points != self.n_points:
            raise ValueError("crossed partitions must cover the same points")
        pair = self.codes.astype(np.int64) * max(other.n_subsets, 1) + other.codes
        unique, inverse = np.unique(pair, return_inverse=True)
        width = max(other.n_subsets, 1)
        labels = []
        weights = []
        for value in unique.tolist():
            a, b = divmod(int(value), width)
            labels.append(f"{self.labels[a]}/{other.labels[b]}")
            weights.append(self.weights[a] * other.weights[b])
        return Partition(
            name=name or f"{self.name}x{other.name}",
            codes=inverse.astype(np.int32).ravel(),
            labels=tuple(labels),
            weights=tuple(weights),
            origin=CrossedOrigin(a=self.name, b=other.name),
        )

    def merged(self, a: int, b: int) -> Partition:
        """Subsets ``a`` and ``b`` as one, at the lower index (label ``"la+lb"``,
        weight of the lower index)."""
        self._check_index(a)
        self._check_index(b)
        if a == b:
            raise ValueError("cannot merge a subset with itself")
        lo, hi = min(a, b), max(a, b)
        groups = [[s] for s in range(self.n_subsets)]
        groups[lo] = [lo, hi]
        del groups[hi]
        return self._regrouped(groups)

    def merged_small(self, time: Any, *, min_points: int) -> Partition:
        """Merge every subset below ``min_points`` into its time neighbour across the
        smaller gap (the remedy :class:`~cuperiod.prewhiten.constraints.FitConstraints`
        names for undersized variation subsets). Empty subsets are dropped."""
        t = self._time(time)
        groups = _merge_small(
            self.codes.astype(np.int64), t, self.n_subsets, int(min_points)
        )
        # Keep the surviving subsets in their original index order.
        groups.sort(key=min)
        return self._regrouped(groups)

    def _regrouped(self, groups: Sequence[Sequence[int]]) -> Partition:
        lookup = np.zeros(self.n_subsets, dtype=np.int32)
        labels = []
        weights = []
        for new, members in enumerate(groups):
            ordered = sorted(members)
            lookup[ordered] = new
            labels.append("+".join(self.labels[m] for m in ordered))
            weights.append(self.weights[ordered[0]])
        return self._edited(lookup[self.codes], labels, weights)

    def split_at(self, time_value: float, time: Any) -> Partition:
        """Split every subset with points on both sides of ``time_value``.

        Points at or after ``time_value`` form a new subset right after the original
        (labels ``"<label>a"`` / ``"<label>b"``, weight copied).

        Raises
        ------
        ValueError
            When no subset straddles ``time_value``.
        """
        t = self._time(time)
        cut = float(time_value)
        before = np.bincount(self.codes[t < cut], minlength=self.n_subsets)
        after = np.bincount(self.codes[t >= cut], minlength=self.n_subsets)
        straddling = set(np.flatnonzero((before > 0) & (after > 0)).tolist())
        if not straddling:
            raise ValueError(f"no subset has points on both sides of {cut!r}")
        new_index = np.zeros(self.n_subsets, dtype=np.int32)
        late_index = np.zeros(self.n_subsets, dtype=np.int32)
        labels: list[str] = []
        weights: list[float] = []
        for s in range(self.n_subsets):
            new_index[s] = len(labels)
            if s in straddling:
                labels.append(f"{self.labels[s]}a")
                weights.append(self.weights[s])
                late_index[s] = len(labels)
                labels.append(f"{self.labels[s]}b")
            else:
                labels.append(self.labels[s])
                late_index[s] = new_index[s]
            weights.append(self.weights[s])
        codes = np.where(t >= cut, late_index[self.codes], new_index[self.codes])
        return self._edited(codes, labels, weights)

    def with_weights(self, weights: Sequence[float]) -> Partition:
        """The same subsets with new relative weights (one per subset)."""
        if len(weights) != self.n_subsets:
            raise ValueError(f"{len(weights)} weights for {self.n_subsets} subsets")
        return self._edited(self.codes, self.labels, weights)

    def renamed(self, index: int, label: str) -> Partition:
        """Subset ``index`` relabelled (labels stay unique)."""
        self._check_index(index)
        labels = list(self.labels)
        labels[index] = _check_label(label)
        return self._edited(self.codes, labels, self.weights)

    # -- internals -------------------------------------------------------------------
    def _check_index(self, index: int) -> None:
        if not 0 <= int(index) < self.n_subsets:
            raise ValueError(f"no subset {index} in partition {self.name!r}")

    def _time(self, time: Any) -> FloatArray:
        t = _as_time(time)
        if t.size != self.n_points:
            raise ValueError("time must have one value per partitioned point")
        return t

    def _mask(self, used: BoolArray) -> BoolArray:
        mask = np.asarray(used, dtype=bool)
        if mask.shape != self.codes.shape:
            raise ValueError("the point mask must have one entry per point")
        return mask


def suggest_partition(time: Any) -> PartitionOrigin | None:
    """The origin of the obvious split of this sampling, or ``None``.

    In order: ground-based nights (several gap-separated runs, each shorter than
    ~18 h), observing seasons (gaps over 60 d), space-mission sectors (runs merged up
    to 28.5 d). A single continuous run has no obvious split.
    """
    t = _as_time(time)
    if t.size < 2:
        return None
    nights = GapsOrigin(min_gap=_NIGHT_GAP)
    night_split = Partition.from_origin(nights, t, name="night")
    if night_split.n_subsets >= 2:
        spans = night_split.spans(t)
        if float(np.median(spans[:, 1] - spans[:, 0])) < _NIGHT_MAX_SPAN:
            return nights
    seasons = GapsOrigin(min_gap=_SEASON_GAP, prefix="S")
    if Partition.from_origin(seasons, t, name="season").n_subsets >= 2:
        return seasons
    sectors = GapsOrigin(min_gap=_SECTOR_GAP, max_span=_SECTOR_MAX_SPAN, prefix="S")
    if Partition.from_origin(sectors, t, name="sector").n_subsets >= 2:
        return sectors
    return None


__all__ = [
    "ChunksOrigin",
    "CrossedOrigin",
    "GapsOrigin",
    "LabelsOrigin",
    "ManualOrigin",
    "Partition",
    "PartitionOrigin",
    "RangesOrigin",
    "parse_origin",
    "suggest_partition",
]
