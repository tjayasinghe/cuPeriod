"""Pure data of an interactive pre-whitening session.

A session is a sequence of immutable :class:`SessionState` snapshots joined by logged
actions. A state holds *parameters only* — component values and flags, the point
mask, the partitions and their roles, the requested settings and the loop's
bookkeeping — so it is kilobytes, compares by content, serializes to canonical JSON
and is what undo moves between. Everything heavy (the fit's residuals, the residual
spectrum) lives in a :class:`Derived` that the kernel can always rebuild from a state.

Nothing in this module computes a fit; it only defines the records the kernel,
the reducers, the history, the project file and the GUI exchange.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import PreWhitenSettings, SessionSettings
from cuperiod.core.errors import CuPeriodError
from cuperiod.prewhiten.constraints import Variation
from cuperiod.prewhiten.subsets import Partition

if TYPE_CHECKING:
    from cuperiod.prewhiten.fit import MultiSineFit
    from cuperiod.prewhiten.spectrum import AmplitudeSpectrum, _SpectrumReadout

BoolArray = npt.NDArray[np.bool_]

#: What a result reports as ``stop_reason`` for a session that is still searching.
INTERACTIVE_STOP_REASON = "interactive session (not run to a stopping rule)"


# --- errors ---------------------------------------------------------------------

#: The remedy a refusal names, as a code (:attr:`StepReport.hint`). The refusal's
#: message spells the remedy in the library's own keyword arguments ("pass
#: allow_close=True ..."), which a front end cannot offer as such; the code lets it
#: phrase its own advice without parsing the message.
#:
#: ``"allow_close"``
#:     The frequency is within the collision radius of another component;
#:     ``add(..., allow_close=True)`` or ``tie(..., allow_close=True)`` keeps it.
#: ``"cascade"``
#:     The component is a tie parent; ``remove(..., cascade=True)`` dissolves the
#:     ties of its children.
#: ``"exact"``
#:     The frequency cannot be refined there; ``add(..., refine="none")`` adds it
#:     exactly where it is.
#: ``"fit_mean"``
#:     Zero points per subset need a floating mean: ``configure(fit_mean=True)``.
RefusalHint = Literal["allow_close", "cascade", "exact", "fit_mean"]

#: Every :data:`RefusalHint` code.
REFUSAL_HINTS: frozenset[str] = frozenset(get_args(RefusalHint))


class SessionError(CuPeriodError):
    """An action a session cannot honour (bad argument, unknown id, wrong state).

    :meth:`~cuperiod.PreWhitenSession.do` turns it into a ``"refused"`` report; it
    propagates from the read-only helpers and from the thread guard.

    Parameters
    ----------
    *args
        The message, as for any exception.
    hint : str, optional
        The remedy as a :data:`RefusalHint` code; the refused report carries it as
        :attr:`StepReport.hint`.
    """

    def __init__(self, *args: object, hint: RefusalHint | None = None) -> None:
        super().__init__(*args)
        self.hint: str | None = hint


class ProjectError(SessionError):
    """A project file, snapshot or log that cannot be read, verified or replayed."""


class SessionWarning(UserWarning):
    """Something a session did that deserves attention but is not an error."""


# --- canonical values ---------------------------------------------------------------


def _number(value: float) -> float | str:
    """A float as canonical JSON: itself when finite, its name otherwise.

    ``json.dumps`` writes a finite float with ``repr``, which round-trips exactly, so
    two states compare equal through this form exactly when their values are
    bitwise equal (NaN equal to NaN).
    """
    x = float(value)
    if math.isfinite(x):
        return x
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def _numbers(values: Sequence[float] | None) -> list[float | str] | None:
    return None if values is None else [_number(v) for v in values]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text (sorted keys, no spaces) of plain data."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def settings_dump(settings: PreWhitenSettings) -> dict[str, Any]:
    """The requested settings as JSON-safe plain data."""
    out: dict[str, Any] = settings.model_dump(mode="json")
    return out


# --- components -------------------------------------------------------------------


@dataclass(frozen=True)
class Tie:
    """A component's frequency tied to an integer combination of others.

    Attributes
    ----------
    parents : tuple of int
        Component ids.
    coefficients : tuple of int
        Non-zero integers aligned with ``parents``.
    """

    parents: tuple[int, ...]
    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.parents) != len(self.coefficients) or not self.parents:
            raise SessionError("a tie needs one coefficient per parent, at least one")

    def text(self) -> str:
        """The tie as the grammar writes it (``"F1 + F2"``, ``"2F1"``)."""
        from cuperiod.prewhiten.constraints import format_tie

        return format_tie(self.parents, self.coefficients)


@dataclass(frozen=True)
class Component:
    """One row of the frequency table.

    Attributes
    ----------
    id : int
        Stable identity; the label is ``F{id}``. Compacted only by ``prune`` and
        ``renumber``.
    frequency, amplitude, phase : float
        The values (phase in radians at the session's ``t_ref``). The amplitude is
        signed: negative only under a fixed phase. Under phase variation it is the
        coherent headline ``A |mean exp(i phi_s)|`` the fit reports, except for a
        fixed amplitude: that stays the held amplitude ``A`` of every subset, and
        only the fit's result reports the (smaller) headline.
    active : bool
        Inactive rows keep their values but leave the model and the peak search.
    fix_frequency, fix_amplitude, fix_phase : bool
        Held at the stored value.
    tie : Tie or None
        The frequency follows its parents.
    variation : {"none", "amplitude", "phase", "both"}
        Which quantities take one value per subset of the variation partition.
    subset_amplitude, subset_phase : tuple of float or None
        Per-subset values (NaN for a subset with no point in use).
    origin : {"auto", "manual"}
        How the row entered the table.
    fap, delta_bic : float
        Of the peak it was drawn from / the BIC change when it was added.
    note : str
        Free text.
    """

    id: int
    frequency: float
    amplitude: float
    phase: float
    active: bool = True
    fix_frequency: bool = False
    fix_amplitude: bool = False
    fix_phase: bool = False
    tie: Tie | None = None
    variation: Variation = "none"
    subset_amplitude: tuple[float, ...] | None = None
    subset_phase: tuple[float, ...] | None = None
    origin: Literal["auto", "manual"] = "auto"
    fap: float = float("nan")
    delta_bic: float = float("nan")
    note: str = ""

    @property
    def label(self) -> str:
        """``F{id}``."""
        return f"F{self.id}"

    @property
    def is_constrained(self) -> bool:
        """Inactive, fixed, tied or varying — anything beyond a free sinusoid."""
        return (
            not self.active
            or self.fix_frequency
            or self.fix_amplitude
            or self.fix_phase
            or self.tie is not None
            or self.variation != "none"
        )

    def canonical(self) -> dict[str, Any]:
        """Plain JSON-safe data, exact for every float."""
        return {
            "id": self.id,
            "frequency": _number(self.frequency),
            "amplitude": _number(self.amplitude),
            "phase": _number(self.phase),
            "active": self.active,
            "fix_frequency": self.fix_frequency,
            "fix_amplitude": self.fix_amplitude,
            "fix_phase": self.fix_phase,
            "tie": None
            if self.tie is None
            else {
                "parents": list(self.tie.parents),
                "coefficients": list(self.tie.coefficients),
            },
            "variation": self.variation,
            "subset_amplitude": _numbers(self.subset_amplitude),
            "subset_phase": _numbers(self.subset_phase),
            "origin": self.origin,
            "fap": _number(self.fap),
            "delta_bic": _number(self.delta_bic),
            "note": self.note,
        }


@dataclass(frozen=True)
class FitStats:
    """The statistics of the fit a state was committed with."""

    rss: float
    rms: float
    bic: float
    n_samples: int
    n_parameters: int
    weighted: bool

    @classmethod
    def of(cls, fit: MultiSineFit) -> FitStats:
        """The statistics of ``fit``."""
        return cls(
            rss=float(fit.rss),
            rms=float(fit.rms),
            bic=float(fit.bic),
            n_samples=int(fit.n_samples),
            n_parameters=int(fit.n_parameters),
            weighted=bool(fit.weighted),
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "rss": _number(self.rss),
            "rms": _number(self.rms),
            "bic": _number(self.bic),
            "n_samples": self.n_samples,
            "n_parameters": self.n_parameters,
            "weighted": self.weighted,
        }


# --- point mask -------------------------------------------------------------------


def _mask_digest(deleted: BoolArray) -> str:
    h = hashlib.sha256(str(int(deleted.size)).encode())
    h.update(np.packbits(deleted).tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class PointMask:
    """Which points of the session data are deleted.

    Nothing is ever removed from :class:`SessionData`: restoring is exact and
    undoable. Compare masks by :attr:`digest`.

    Attributes
    ----------
    deleted : numpy.ndarray
        ``(N,)`` read-only booleans, ``True`` = left out of the fit.
    digest : str
        SHA-256 of the mask (and its length).
    """

    deleted: BoolArray
    digest: str

    @classmethod
    def _of(cls, deleted: BoolArray) -> PointMask:
        out = np.array(deleted, dtype=bool, copy=True)
        out.setflags(write=False)
        return cls(deleted=out, digest=_mask_digest(out))

    @classmethod
    def none(cls, n: int) -> PointMask:
        """No point deleted."""
        return cls._of(np.zeros(int(n), dtype=bool))

    @classmethod
    def from_runs(cls, n: int, runs: Sequence[Sequence[int]]) -> PointMask:
        """Deleted points given as half-open ``(start, stop)`` index runs.

        Raises
        ------
        SessionError
            For a run outside ``[0, n)`` or with ``stop <= start``.
        """
        deleted = np.zeros(int(n), dtype=bool)
        for run in runs:
            start, stop = _check_run(run, int(n))
            deleted[start:stop] = True
        return cls._of(deleted)

    @property
    def n(self) -> int:
        return int(self.deleted.size)

    @property
    def n_deleted(self) -> int:
        return int(np.count_nonzero(self.deleted))

    def without(self, indices: IntArray) -> PointMask:
        """This mask with ``indices`` deleted as well."""
        deleted = np.array(self.deleted, copy=True)
        deleted[_check_indices(indices, self.n)] = True
        return PointMask._of(deleted)

    def restored(self, indices: IntArray | None = None) -> PointMask:
        """This mask with ``indices`` (``None``: every point) restored."""
        if indices is None:
            return PointMask.none(self.n)
        deleted = np.array(self.deleted, copy=True)
        deleted[_check_indices(indices, self.n)] = False
        return PointMask._of(deleted)

    def runs(self) -> list[tuple[int, int]]:
        """Maximal runs of deleted points, half-open ``(start, stop)``."""
        if not self.deleted.any():
            return []
        padded = np.concatenate([[False], self.deleted, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        return [
            (int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)
        ]


def _check_run(run: Sequence[int], n: int) -> tuple[int, int]:
    if len(run) != 2:
        raise SessionError(f"a run is (start, stop), got {tuple(run)!r}")
    start, stop = int(run[0]), int(run[1])
    if not (0 <= start < stop <= n):
        raise SessionError(f"run ({start}, {stop}) is outside [0, {n})")
    return start, stop


def _check_indices(indices: Any, n: int) -> IntArray:
    array = np.asarray(indices)
    if array.dtype == bool:
        if array.shape != (n,):
            raise SessionError(f"a boolean selection needs {n} entries")
        return np.flatnonzero(array).astype(np.int64)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise SessionError("point indices must be integers")
    out = array.astype(np.int64).ravel()
    if out.size and (int(out.min()) < 0 or int(out.max()) >= n):
        raise SessionError(f"point indices must lie in [0, {n})")
    return out


# --- data -------------------------------------------------------------------------


def content_sha256(
    time: FloatArray, value: FloatArray, error: FloatArray | None
) -> str:
    """SHA-256 of the float64 little-endian bytes of ``time | value | error``."""
    h = hashlib.sha256()
    for array in (time, value) if error is None else (time, value, error):
        h.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return h.hexdigest()


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True, eq=False)
class SessionData:
    """The light curve a session works on, fixed for its whole life.

    The arrays are the coerced, finite arrays exactly as :func:`cuperiod.prewhiten`
    holds them (often the caller's own arrays). The session never writes them and
    never changes their ``writeable`` flag; :meth:`readonly` hands out read-only
    views.

    Attributes
    ----------
    time, value : numpy.ndarray
        Finite points, in input order.
    error : numpy.ndarray or None
        1-sigma errors.
    domain : Domain
        Brightness domain.
    source_index : numpy.ndarray
        Row of each point in the original input (before the finite filter).
    labels : Mapping[str, numpy.ndarray]
        Extra per-point columns (``"band"``, ``"site"``), filtered alike.
    meta : Mapping
        Metadata carried from the light curve.
    content_sha256 : str
        See :func:`content_sha256`.
    """

    time: FloatArray
    value: FloatArray
    error: FloatArray | None
    domain: Domain
    source_index: IntArray
    labels: Mapping[str, np.ndarray]
    meta: Mapping[str, Any]
    content_sha256: str

    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.time.size)

    def readonly(self) -> SessionData:
        """A copy of this record whose arrays are read-only views."""
        return SessionData(
            time=_readonly_view(self.time),
            value=_readonly_view(self.value),
            error=None if self.error is None else _readonly_view(self.error),
            domain=self.domain,
            source_index=_readonly_view(self.source_index),
            labels={k: _readonly_view(v) for k, v in self.labels.items()},
            meta=dict(self.meta),
            content_sha256=self.content_sha256,
        )


@dataclass(frozen=True, eq=False)
class UsedData:
    """The points (and weights and group codes) a fit of one state uses.

    Attributes
    ----------
    time, value, error : numpy.ndarray
        The points in use. The :class:`SessionData` arrays themselves (same objects)
        when every point is in use and no subset weights apply.
    error_eff : numpy.ndarray or None
        What the spectrum engine and the false-alarm probability see:
        ``error / sqrt(omega)`` with subset weights (``1 / sqrt(omega)`` without error
        bars), else ``error``.
    point_weights : numpy.ndarray or None
        ``omega`` per point in use, normalised to mean 1; ``None`` without weights.
    index : numpy.ndarray
        Position of each point in use within :class:`SessionData`.
    used : numpy.ndarray
        ``(N,)`` read-only booleans over :class:`SessionData`.
    offset_codes, variation_codes : numpy.ndarray or None
        Compacted ``0..S-1`` codes per point in use (empty subsets dropped).
    offset_labels, variation_labels : tuple of str
        Labels of the subsets the compacted codes name.
    offset_subsets, variation_subsets : numpy.ndarray or None
        Index of each compacted subset in its partition.
    """

    time: FloatArray
    value: FloatArray
    error: FloatArray | None
    error_eff: FloatArray | None
    point_weights: FloatArray | None
    index: IntArray
    used: BoolArray
    offset_codes: npt.NDArray[np.int32] | None
    offset_labels: tuple[str, ...]
    variation_codes: npt.NDArray[np.int32] | None
    variation_labels: tuple[str, ...]
    offset_subsets: IntArray | None = None
    variation_subsets: IntArray | None = None

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def is_full(self) -> bool:
        """Every point in use and no weights."""
        return self.point_weights is None and bool(self.used.all())

    @classmethod
    def full(cls, data: SessionData) -> UsedData:
        """Every point, no weights, no groups: the :class:`SessionData` arrays."""
        used = np.ones(data.n, dtype=bool)
        used.setflags(write=False)
        index = np.arange(data.n, dtype=np.int64)
        index.setflags(write=False)
        return cls(
            time=data.time,
            value=data.value,
            error=data.error,
            error_eff=data.error,
            point_weights=None,
            index=index,
            used=used,
            offset_codes=None,
            offset_labels=(),
            variation_codes=None,
            variation_labels=(),
        )


# --- state ------------------------------------------------------------------------

Status = Literal["searching", "stopped"]


@dataclass(frozen=True, eq=False)
class SessionState:
    """One immutable snapshot of a session: parameters only.

    Compares by identity; :meth:`same_content` compares everything but
    :attr:`serial`.

    Attributes
    ----------
    serial : int
        Unique, increasing within a session; the cache key of its :class:`Derived`.
    components : tuple of Component
        The frequency table in order of addition.
    offset : float
        The fitted constant (with ``fit_mean=False``: the fixed weighted mean).
    subset_offsets : tuple of float or None
        One zero point per subset of the ``offsets_by`` partition (NaN when empty).
    stats : FitStats
        Of the committed fit.
    mask : PointMask
        Deleted points.
    partitions : tuple of Partition
        Defined partitions (root ones first).
    excluded : tuple of (str, tuple of int)
        Per partition name, subset indices left out of the fit.
    offsets_by, variation_by, weights_by : str or None
        The partition supplying zero points / per-subset amplitudes and phases /
        subset weights.
    settings : PreWhitenSettings
        The requested settings exactly as given (``None`` grid fields stay ``None``).
    grid : tuple of (float, float, int)
        ``(f0, df, nf)`` as the spectrum engine uses them.
    blocked : tuple of float
        Peak seeds whose refinement merged into an existing component.
    n_iterations : int
        Extraction attempts, as :attr:`~cuperiod.PreWhitenResult.n_iterations`.
    budget_origin : int
        ``n_iterations`` at the last manual edit (the attempt budget restarts there).
    n_pruned : int
        Components removed by ``prune``.
    status : {"searching", "stopped"}
    stop_reason : str
        Why the automatic search stopped (``""`` while searching).
    next_id : int
        Id of the next new component.
    """

    serial: int
    components: tuple[Component, ...]
    offset: float
    subset_offsets: tuple[float, ...] | None
    stats: FitStats
    mask: PointMask
    partitions: tuple[Partition, ...]
    excluded: tuple[tuple[str, tuple[int, ...]], ...]
    offsets_by: str | None
    variation_by: str | None
    weights_by: str | None
    settings: PreWhitenSettings
    grid: tuple[float, float, int]
    blocked: tuple[float, ...]
    n_iterations: int
    budget_origin: int
    n_pruned: int
    status: Status
    stop_reason: str
    next_id: int

    # -- views ---------------------------------------------------------------------
    @property
    def active(self) -> tuple[Component, ...]:
        """The components in the model, in table order."""
        return tuple(c for c in self.components if c.active)

    @property
    def active_frequencies(self) -> FloatArray:
        return np.asarray([c.frequency for c in self.active], dtype=np.float64)

    @property
    def inactive_frequencies(self) -> FloatArray:
        return np.asarray(
            [c.frequency for c in self.components if not c.active], dtype=np.float64
        )

    def component(self, ref: int | str) -> Component:
        """The component with id ``ref`` (``3`` or ``"F3"``).

        Raises
        ------
        SessionError
            For an unknown id.
        """
        wanted = parse_ref(ref)
        for c in self.components:
            if c.id == wanted:
                return c
        raise SessionError(f"no component F{wanted}")

    def partition(self, name: str) -> Partition:
        """The partition called ``name`` (:class:`SessionError` when undefined)."""
        for p in self.partitions:
            if p.name == name:
                return p
        raise SessionError(f"no partition named {name!r}")

    def excluded_subsets(self, name: str) -> tuple[int, ...]:
        for partition, subsets in self.excluded:
            if partition == name:
                return subsets
        return ()

    # -- classification --------------------------------------------------------------
    def has_model_constraints(self) -> bool:
        """Any inactive, fixed, tied or varying component, or fitted zero points."""
        return self.offsets_by is not None or any(
            c.is_constrained for c in self.components
        )

    def has_data_edits(self) -> bool:
        """Any deleted point, excluded subset, or subset weights."""
        return (
            bool(self.mask.deleted.any())
            or any(subsets for _, subsets in self.excluded)
            or self.weights_by is not None
        )

    def is_plain(self) -> bool:
        """Neither constraints nor data edits: the automatic loop's code applies."""
        return not (self.has_model_constraints() or self.has_data_edits())

    # -- content -------------------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        """Every field but :attr:`serial` as plain, exact, JSON-safe data."""
        return {
            "components": [c.canonical() for c in self.components],
            "offset": _number(self.offset),
            "subset_offsets": _numbers(self.subset_offsets),
            "stats": self.stats.canonical(),
            "mask": {"n": self.mask.n, "runs": [list(r) for r in self.mask.runs()]},
            "partitions": [_partition_canonical(p) for p in self.partitions],
            "excluded": [[name, list(subsets)] for name, subsets in self.excluded],
            "offsets_by": self.offsets_by,
            "variation_by": self.variation_by,
            "weights_by": self.weights_by,
            "settings": settings_dump(self.settings),
            "grid": [_number(self.grid[0]), _number(self.grid[1]), int(self.grid[2])],
            "blocked": _numbers(self.blocked),
            "n_iterations": self.n_iterations,
            "budget_origin": self.budget_origin,
            "n_pruned": self.n_pruned,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "next_id": self.next_id,
        }

    def same_content(self, other: SessionState) -> bool:
        """Equal in every field but :attr:`serial` (arrays and settings by value)."""
        if self is other:
            return True
        return self.canonical() == other.canonical()

    def content_digest(self) -> str:
        """SHA-256 of the canonical JSON (serial excluded)."""
        return hashlib.sha256(canonical_json(self.canonical()).encode()).hexdigest()


def _partition_canonical(partition: Partition) -> dict[str, Any]:
    return {
        "name": partition.name,
        "n": partition.n_points,
        "runs": [list(r) for r in partition.runs()],
        "labels": list(partition.labels),
        "weights": _numbers(partition.weights),
        "origin": partition.origin.model_dump(mode="json"),
    }


def parse_ref(ref: int | str) -> int:
    """A component id from ``3``, ``"3"`` or ``"F3"`` (:class:`SessionError` else)."""
    if isinstance(ref, bool):
        raise SessionError(f"not a component reference: {ref!r}")
    if isinstance(ref, (int, np.integer)):
        return int(ref)
    if isinstance(ref, str):
        text = ref.strip()
        if text[:1] in ("F", "f"):
            text = text[1:]
        if text.isdigit():
            return int(text)
    raise SessionError(f"not a component reference: {ref!r}")


# --- derived ------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Derived:
    """The heavy, recomputable companion of a state (never serialized).

    Attributes
    ----------
    fit : MultiSineFit
        Of the active components, in table order.
    scan : spectrum readout
        The residual spectrum as the extraction loop reads it (amplitude-only until a
        full :attr:`residual_spectrum` is asked for).
    sampling_key : str
        Identifies the points, weights and grid the spectrum was evaluated on.
    drift : float
        Largest relative difference between the state and a rebuilt fit (0 when the
        fit was not rebuilt).
    """

    fit: MultiSineFit
    scan: _SpectrumReadout
    sampling_key: str
    drift: float = 0.0

    @cached_property
    def residual_spectrum(self) -> AmplitudeSpectrum:
        """The full residual :class:`AmplitudeSpectrum` (materialised once)."""
        from cuperiod.prewhiten.spectrum import AmplitudeSpectrum

        if isinstance(self.scan, AmplitudeSpectrum):
            return self.scan
        spectrum: AmplitudeSpectrum = self.scan.to_spectrum()  # type: ignore[attr-defined]
        return spectrum


# --- reports and the log --------------------------------------------------------

ReportKind = Literal[
    "accepted", "blocked", "rejected", "stopped", "edited", "refused", "noop"
]

#: Report kinds that are committed (and logged).
COMMITTED_KINDS: frozenset[str] = frozenset(
    {"accepted", "blocked", "rejected", "stopped", "edited"}
)


@dataclass(frozen=True)
class StepReport:
    """What one action did.

    Attributes
    ----------
    kind : str
        ``"accepted"``, ``"blocked"``, ``"rejected"``, ``"stopped"``, ``"edited"``
        (committed), or ``"refused"`` / ``"noop"`` (nothing changed).
    message : str
        One human-readable line.
    component_id : int or None
        The component the action concerned.
    frequency : float or None
        The frequency it concerned.
    statistics : Mapping[str, float]
        ``snr``, ``fap``, ``delta_bic``, ``amplitude``, ``rms`` where they apply.
    changed : frozenset of str
        Of ``components, fit, data, partitions, settings, status, ids``.
    warning : str
        E.g. a logged seed far from this run's tallest peak.
    resolved : Mapping or None
        The arguments the reducer actually used, when it resolved some (the seed and
        grid index of a step, a snapped frequency): what the log stores instead of the
        arguments as given. Not part of the log entry's report.
    hint : str or None
        For a ``"refused"`` report whose message names a remedy, that remedy as a
        :data:`RefusalHint` code (``"allow_close"``, ``"cascade"``, ``"exact"``,
        ``"fit_mean"``); ``None`` otherwise. The message is unchanged.
    """

    kind: ReportKind
    message: str
    component_id: int | None = None
    frequency: float | None = None
    statistics: Mapping[str, float] = field(default_factory=dict)
    changed: frozenset[str] = frozenset()
    warning: str = ""
    resolved: Mapping[str, Any] | None = field(default=None, compare=False)
    hint: str | None = None

    @property
    def committed(self) -> bool:
        return self.kind in COMMITTED_KINDS

    def to_dict(self) -> dict[str, Any]:
        """The log-entry form (NaN statistics become ``None``).

        ``hint`` is written only when set: only refusals carry one, and they are
        never logged, so a log entry's form is what it was before the field existed.
        """
        out: dict[str, Any] = {"kind": self.kind, "message": self.message[:500]}
        out["component_id"] = self.component_id
        out["frequency"] = (
            None
            if self.frequency is None or not math.isfinite(self.frequency)
            else float(self.frequency)
        )
        out["statistics"] = {
            k: (float(v) if v is not None and math.isfinite(float(v)) else None)
            for k, v in sorted(self.statistics.items())
        }
        if self.warning:
            out["warning"] = self.warning[:500]
        if self.hint is not None:
            out["hint"] = self.hint
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StepReport:
        stats = {
            str(k): (float("nan") if v is None else float(v))
            for k, v in dict(data.get("statistics") or {}).items()
        }
        frequency = data.get("frequency")
        component = data.get("component_id")
        hint = data.get("hint")
        return cls(
            kind=data["kind"],
            message=str(data["message"]),
            component_id=None if component is None else int(component),
            frequency=None if frequency is None else float(frequency),
            statistics=stats,
            warning=str(data.get("warning", "")),
            hint=None if hint is None else str(hint),
        )


@dataclass(frozen=True)
class Candidate:
    """What :meth:`~cuperiod.PreWhitenSession.step` would try next (a preview)."""

    frequency: float
    index: int
    spectrum_amplitude: float
    snr_estimate: float
    fap: float


@dataclass(frozen=True)
class LogEntry:
    """One committed action of the effective log."""

    seq: int
    group: int
    op: str
    args: Mapping[str, Any]
    report: StepReport
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "group": self.group,
            "op": self.op,
            "args": dict(self.args),
            "report": self.report.to_dict(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogEntry:
        return cls(
            seq=int(data["seq"]),
            group=int(data["group"]),
            op=str(data["op"]),
            args=dict(data["args"]),
            report=StepReport.from_dict(data["report"]),
            timestamp=str(data["timestamp"]),
        )


@dataclass(frozen=True)
class CreationArgs:
    """Everything that determines a session's root state besides the data.

    Attributes
    ----------
    backend : str
        The concrete spectrum backend that ran.
    t_ref : float
        The frozen phase epoch.
    grid : tuple of (float, float, int)
        ``(f0, df, nf)`` of the frozen frequency grid.
    settings : PreWhitenSettings
        The requested settings, as given to the constructor.
    session_settings : SessionSettings
    partitions : tuple of Partition
        Root partitions (e.g. ``band`` of :meth:`PreWhitenSession.from_bands`).
    label_names : tuple of str
        The extra per-point columns the session was given.
    content_sha256 : str
        :func:`content_sha256` of the data (``""`` when unknown); a restore onto
        different data is refused.
    """

    backend: str
    t_ref: float
    grid: tuple[float, float, int]
    settings: PreWhitenSettings
    session_settings: SessionSettings
    partitions: tuple[Partition, ...] = ()
    label_names: tuple[str, ...] = ()
    content_sha256: str = ""


@dataclass(frozen=True)
class SessionSnapshot:
    """A session without its data: the unit of persistence and of GUI caching.

    Attributes
    ----------
    creation : CreationArgs
    head : SessionState
    log : tuple of LogEntry
        The effective log, root to head.
    states : tuple or None
        In memory only (never written): the retained states aligned with
        ``(root, *log)``, ``None`` where a state was evicted.
    saved_serial : int or None
        Serial of the head when the session was last saved.
    """

    creation: CreationArgs
    head: SessionState
    log: tuple[LogEntry, ...]
    states: tuple[SessionState | None, ...] | None = None
    saved_serial: int | None = None


# --- variations (computed by session.variation, defined here) ---------------------


@dataclass(frozen=True, eq=False)
class VariationResult:
    """Per-subset amplitudes and phases of selected components, with constancy tests.

    A query result: computing it changes nothing in the session.
    """

    component_ids: tuple[int, ...]
    mode: Variation
    by: str
    subset_labels: tuple[str, ...]
    mid_time: FloatArray
    n_points: IntArray
    amplitude: FloatArray
    amplitude_error: FloatArray
    phase: FloatArray
    phase_error: FloatArray
    common_amplitude: FloatArray
    common_phase: FloatArray
    amplitude_statistic: FloatArray
    amplitude_p_value: FloatArray
    phase_statistic: FloatArray
    phase_p_value: FloatArray
    delta_bic: float
    frequencies_improved: bool

    def to_table(self) -> list[dict[str, Any]]:
        """One row per (component, subset)."""
        rows: list[dict[str, Any]] = []
        for k, cid in enumerate(self.component_ids):
            for s, label in enumerate(self.subset_labels):
                rows.append(
                    {
                        "id": cid,
                        "label": f"F{cid}",
                        "subset": label,
                        "mid_time": float(self.mid_time[s]),
                        "n_points": int(self.n_points[s]),
                        "amplitude": float(self.amplitude[k, s]),
                        "amplitude_error": float(self.amplitude_error[k, s]),
                        "phase": float(self.phase[k, s]),
                        "phase_error": float(self.phase_error[k, s]),
                        "amplitude_p_value": float(self.amplitude_p_value[k]),
                        "phase_p_value": float(self.phase_p_value[k]),
                    }
                )
        return rows


__all__ = [
    "COMMITTED_KINDS",
    "INTERACTIVE_STOP_REASON",
    "BoolArray",
    "Candidate",
    "Component",
    "CreationArgs",
    "Derived",
    "FitStats",
    "LogEntry",
    "PointMask",
    "ProjectError",
    "REFUSAL_HINTS",
    "RefusalHint",
    "ReportKind",
    "SessionData",
    "SessionError",
    "SessionSnapshot",
    "SessionState",
    "SessionWarning",
    "StepReport",
    "Tie",
    "UsedData",
    "VariationResult",
    "canonical_json",
    "content_sha256",
    "parse_ref",
    "settings_dump",
]
