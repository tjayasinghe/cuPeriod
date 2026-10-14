"""Plain data models for the GUI: cache keys, the result cache, and source items.

None of these touch Qt — they are pure Python so they unit-test headlessly. A
:class:`ResultKey` identifies a result by *(source, analysis, settings, backend, data)*;
the :class:`ResultCache` (a small LRU, bounded by entry count and by bytes) returns a
previously computed :class:`~cuperiod.core.result.Periodogram` — or a
:class:`~cuperiod.prewhiten.PreWhitenResult`, since the cache is generic over its value
type — instantly when the user revisits an unchanged setup. A :class:`SourceItem` is one
entry in single/batch mode: a label and a lazy ``loader`` so a folder of thousands of
light curves is not all read up front.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Generic, TypeVar

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve

#: A loaded source is either a single- or multi-band light curve.
LoadedCurve = LightCurve | MultiBandLightCurve

#: Band-picker tokens for the two synthetic "all bands" entries. They travel from the
#: controls to the controller unchanged, so a real band that happens to be *named*
#: ``combined`` or ``stacked`` can never be mistaken for the all-bands choice. The
#: double underscores keep them out of any plausible filter name.
COMBINED_BAND: Final = "__combined__"
STACKED_BAND: Final = "__stacked__"

_ResultT = TypeVar("_ResultT")


def settings_hash(settings: BaseSettings, *, exclude: Iterable[str] = ()) -> str:
    """A short, order-independent hash of a settings object.

    Uses the JSON-mode model dump with sorted keys, so two settings that differ only in
    field order hash identically and any value change produces a different digest.
    ``exclude`` drops fields that do not change the computed result (the controller
    leaves out the peak-selection fields, which are applied after the compute), so
    editing one of those is a cache hit rather than a full recompute.
    """
    dumped = settings.model_dump(mode="json", exclude=set(exclude) or None)
    blob = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _hash_array(digest: Any, values: np.ndarray | None) -> None:
    """Feed one optional float array into ``digest`` (shape first, then bytes)."""
    if values is None:
        digest.update(b"\x00none")
        return
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(np.int64(array.size).tobytes())
    digest.update(memoryview(array).cast("B"))


def data_fingerprint(lc: LoadedCurve | None) -> str:
    """A content hash of a light curve: equal data, equal fingerprint.

    Covers every band's name, domain, time, value and error arrays (in band order),
    so it changes whenever the analysed numbers change — another star that happens to
    share a file stem, or the same file edited on disk — and stays put when the very
    same data is loaded again as a new object (batch mode re-runs the loader on every
    visit, which must stay an instant cache hit). Metadata such as the source path is
    deliberately left out: it identifies where the data came from, not what it is.

    The cost is one pass over the arrays (about 2 ms per 10^5 points), paid once per
    load. Returns ``""`` for no curve.
    """
    if lc is None:
        return ""
    digest = hashlib.blake2b(digest_size=8)
    if isinstance(lc, MultiBandLightCurve):
        digest.update(b"multiband")
        bands: Mapping[str, LightCurve] = lc.bands
    else:
        digest.update(b"single")
        bands = {"": lc}
    for name, band in bands.items():
        digest.update(b"\x00band\x00" + name.encode("utf-8", "surrogatepass"))
        digest.update(band.domain.value.encode("ascii"))
        _hash_array(digest, band.time)
        _hash_array(digest, band.value)
        _hash_array(digest, band.error)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResultKey:
    """Identity of a computed result, used as the cache key.

    ``method`` is the periodogram method for a spectrum run and ``"PREWHITEN"`` for a
    frequency-solution run, so the two analyses can never collide. ``backend`` is the
    *requested* backend (``"auto"``/``"cpu"``/``"gpu"``/...), kept distinct so a CPU and
    a GPU run of the same configuration cache separately. ``data_id`` is the
    :func:`data_fingerprint` of the analysed curve: ``source_id`` alone (a file stem, a
    path) says where a curve came from, not what it holds, so two different stars with
    the same name, or a file edited on disk, must not share a cached result.
    """

    source_id: str
    method: str
    settings_hash: str
    backend: str
    data_id: str = ""


def result_nbytes(value: object) -> int:
    """Bytes of numpy buffers reachable from ``value`` (each buffer counted once).

    Walks dataclass fields, mappings and sequences, so a
    :class:`~cuperiod.core.result.Periodogram` (frequency, period, power, extras) and a
    pre-whitening result (several spectra sharing one frequency grid) are both sized by
    what they actually keep alive. Views are charged for their whole base buffer, the
    memory a cached view really retains.
    """
    total = 0
    seen_buffers: set[int] = set()
    seen_objects: set[int] = set()
    stack: list[object] = [value]
    while stack:
        obj = stack.pop()
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in seen_buffers:
                seen_buffers.add(id(base))
                total += int(base.nbytes)
            continue
        if obj is None or isinstance(
            obj, (str, bytes, int, float, complex, np.generic)
        ):
            continue
        if id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            stack.extend(getattr(obj, f.name) for f in dataclasses.fields(obj))
        elif isinstance(obj, Mapping):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
    return total


class ResultCache(Generic[_ResultT]):
    """A bounded LRU cache of results keyed by :class:`ResultKey`.

    Generic over the value type: the app keeps one instance for periodograms and one for
    pre-whitening solutions, so switching analysis mode back and forth is instant.

    Two limits apply. ``maxsize`` caps the number of entries; ``max_bytes`` (optional)
    caps the numpy memory they hold, measured once per entry by :func:`result_nbytes`.
    A count alone says little: an automatic grid scales with baseline times maximum
    frequency, so one multi-year ground-based periodogram is tens of MB and a
    pre-whitening result (three spectra) about a hundred. Least-recently-used entries
    are evicted until both limits hold, but the newest entry is always kept, even when
    it alone exceeds the byte budget.
    """

    def __init__(self, maxsize: int = 64, *, max_bytes: int | None = None) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be >= 0 or None")
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._items: OrderedDict[ResultKey, tuple[_ResultT, int]] = OrderedDict()
        self._nbytes = 0

    @property
    def nbytes(self) -> int:
        """Total numpy bytes held by the cached entries (as measured on insert)."""
        return self._nbytes

    def get(self, key: ResultKey) -> _ResultT | None:
        """Cached result for ``key`` (marks it recently used), else None."""
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: ResultKey, value: _ResultT) -> None:
        """Insert/refresh ``key``; evict LRU entries while over either limit."""
        size = result_nbytes(value) if self._max_bytes is not None else 0
        old = self._items.pop(key, None)
        if old is not None:
            self._nbytes -= old[1]
        self._items[key] = (value, size)
        self._nbytes += size
        while len(self._items) > 1 and (
            len(self._items) > self._maxsize
            or (self._max_bytes is not None and self._nbytes > self._max_bytes)
        ):
            _, (_, evicted) = self._items.popitem(last=False)
            self._nbytes -= evicted

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """Drop all cached results."""
        self._items.clear()
        self._nbytes = 0


@dataclass(frozen=True)
class SourceItem:
    """One selectable light curve: a stable key, a display label, and a lazy loader.

    ``loader`` is called only when the source is selected, so enumerating a large folder
    stays cheap. ``meta`` carries demo annotations (e.g. a catalogue period) used in
    labels and sanity checks.
    """

    key: str
    label: str
    loader: Callable[[], LoadedCurve]
    meta: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "COMBINED_BAND",
    "STACKED_BAND",
    "LoadedCurve",
    "ResultCache",
    "ResultKey",
    "SourceItem",
    "data_fingerprint",
    "result_nbytes",
    "settings_hash",
]
