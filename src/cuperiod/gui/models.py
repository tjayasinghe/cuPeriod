"""Plain data models for the GUI: cache keys, the result cache, and source items.

None of these touch Qt — they are pure Python so they unit-test headlessly. A
:class:`ResultKey` identifies a result by *(source, analysis, settings, backend)*; the
:class:`ResultCache` (a small LRU) returns a previously computed
:class:`~cuperiod.core.result.Periodogram` — or a
:class:`~cuperiod.prewhiten.PreWhitenResult`, since the cache is generic over its value
type — instantly when the user revisits an unchanged setup. A :class:`SourceItem` is one
entry in single/batch mode: a label and a lazy ``loader`` so a folder of thousands of
light curves is not all read up front.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic_settings import BaseSettings

from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve

#: A loaded source is either a single- or multi-band light curve.
LoadedCurve = LightCurve | MultiBandLightCurve

_ResultT = TypeVar("_ResultT")


def settings_hash(settings: BaseSettings) -> str:
    """A short, order-independent hash of a settings object.

    Uses the JSON-mode model dump with sorted keys, so two settings that differ only in
    field order hash identically and any value change produces a different digest.
    """
    blob = json.dumps(
        settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ResultKey:
    """Identity of a computed result, used as the cache key.

    ``method`` is the periodogram method for a spectrum run and ``"PREWHITEN"`` for a
    frequency-solution run, so the two analyses can never collide. ``backend`` is the
    *requested* backend (``"auto"``/``"cpu"``/``"gpu"``/...), kept distinct so a CPU and
    a GPU run of the same configuration cache separately.
    """

    source_id: str
    method: str
    settings_hash: str
    backend: str


class ResultCache(Generic[_ResultT]):
    """A bounded LRU cache of results keyed by :class:`ResultKey`.

    Generic over the value type: the app keeps one instance for periodograms and one for
    pre-whitening solutions, so switching analysis mode back and forth is instant.
    """

    def __init__(self, maxsize: int = 64) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._items: OrderedDict[ResultKey, _ResultT] = OrderedDict()

    def get(self, key: ResultKey) -> _ResultT | None:
        """Cached result for ``key`` (marks it recently used), else None."""
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: ResultKey, value: _ResultT) -> None:
        """Insert/refresh ``key``; evict the LRU entry if over capacity."""
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._maxsize:
            self._items.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """Drop all cached results."""
        self._items.clear()


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
    "LoadedCurve",
    "ResultCache",
    "ResultKey",
    "SourceItem",
    "settings_hash",
]
