"""Reducers of the data operations: points, subsets, roles, weights and settings.

Nothing is ever removed from the session's data. Deleting points produces a new
:class:`~cuperiod.prewhiten.session.state.PointMask`; leaving a whole subset out is a
separate record (``state.excluded``), so re-including a subset never resurrects points
deleted inside it and restoring points never re-includes an excluded subset. ``t_ref``
and the frequency grid stay where the session froze them; the baseline, the resolution
guards and the refinement box follow the points in use.

A change of the points in use or of their weights changes the state's *sampling key*,
so the kernel builds (or finds in its LRU, after an undo) the spectrum engine of the
new sampling; the implicit Calculate of the manual edit then re-solves the model on
it. Partitions themselves (``define_partition`` / ``drop_partition``) are neutral: a
partition no role uses does not touch the fit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.lightcurve import LightCurve
from cuperiod.prewhiten.session.actions import (
    ConfigureArgs,
    DefinePartitionArgs,
    DeletePointsArgs,
    ExcludeSubsetsArgs,
    NameArgs,
    RestorePointsArgs,
    SetSubsetWeightsArgs,
    UseArgs,
    register,
)
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.session.ops_components import labels, validate_model
from cuperiod.prewhiten.session.state import (
    Derived,
    PointMask,
    SessionError,
    SessionState,
    StepReport,
    settings_dump,
)
from cuperiod.prewhiten.subsets import Partition, parse_origin

Outcome = tuple[SessionState, Derived | None, StepReport]

#: Settings whose change recomputes the frequency grid from the points in use.
GRID_FIELDS: tuple[str, ...] = (
    "minimum_frequency",
    "maximum_frequency",
    "nyquist_factor",
    "samples_per_peak",
)

#: Settings fixed for the life of a session (a new session, ``load(..., backend=)``
#: or ``restore(..., backend=)`` changes them).
FIXED_FIELDS: tuple[str, ...] = ("backend", "device")

_ROLES: tuple[str, ...] = ("offsets_by", "variation_by", "weights_by")


def _edited(message: str, *changed: str) -> StepReport:
    return StepReport(kind="edited", message=message, changed=frozenset(changed))


def _noop(state: SessionState, message: str) -> Outcome:
    return state, None, StepReport(kind="noop", message=message)


def check_points(kernel: SessionKernel, state: SessionState) -> None:
    """Refuse a state whose points in use cannot carry a pre-whitening.

    Fewer than ``settings.min_detections`` points, or no time baseline, is the same
    refusal the constructor makes for a light curve that small.
    """
    used = kernel.used(state)
    minimum = int(state.settings.min_detections)
    if used.n < minimum:
        raise SessionError(
            f"only {used.n} points would remain in use (min_detections={minimum})"
        )
    if used.n < 3 or not float(np.ptp(used.time)) > 0.0:
        raise SessionError("the points in use would have no time baseline")


def _roles_of(state: SessionState, name: str) -> list[str]:
    return [role for role in _ROLES if getattr(state, role) == name]


def _same_partition(a: Partition, b: Partition) -> bool:
    return (
        a.labels == b.labels
        and a.weights == b.weights
        and np.array_equal(a.codes, b.codes)
        and a.origin.model_dump(mode="json") == b.origin.model_dump(mode="json")
    )


def _replaced(state: SessionState, partition: Partition) -> SessionState:
    """``state`` with ``partition`` replacing the one of the same name, else added."""
    names = [p.name for p in state.partitions]
    if partition.name in names:
        partitions = tuple(
            partition if p.name == partition.name else p for p in state.partitions
        )
    else:
        partitions = (*state.partitions, partition)
    return replace(state, partitions=partitions)


# --- points -------------------------------------------------------------------------


def _selection(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """Indices named by half-open runs (:class:`SessionError` outside ``[0, n)``)."""
    return np.flatnonzero(PointMask.from_runs(n, runs).deleted)


@register("delete_points")
def delete_points(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: DeletePointsArgs,
) -> Outcome:
    """Leave points out of every fit and spectrum (exactly restorable)."""
    mask = state.mask.without(_selection(kernel.data.n, args.runs))
    if mask.digest == state.mask.digest:
        return _noop(state, "those points are already deleted")
    new = replace(state, mask=mask)
    check_points(kernel, new)
    validate_model(kernel, new)
    count = mask.n_deleted - state.mask.n_deleted
    return new, None, _edited(
        f"deleted {count} point{'' if count == 1 else 's'} "
        f"({mask.n_deleted} deleted in all)",
        "data",
        "fit",
    )


@register("restore_points")
def restore_points(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: RestorePointsArgs,
) -> Outcome:
    """Bring deleted points back (``runs=None``: all of them)."""
    if args.runs is None:
        mask = state.mask.restored()
    else:
        mask = state.mask.restored(_selection(kernel.data.n, args.runs))
    if mask.digest == state.mask.digest:
        return _noop(state, "no deleted point to restore")
    new = replace(state, mask=mask)
    check_points(kernel, new)
    validate_model(kernel, new)
    count = state.mask.n_deleted - mask.n_deleted
    return new, None, _edited(
        f"restored {count} point{'' if count == 1 else 's'}", "data", "fit"
    )


# --- subsets ----------------------------------------------------------------------


@register("exclude_subsets")
def exclude_subsets(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: ExcludeSubsetsArgs,
) -> Outcome:
    """Leave whole subsets out of (``excluded=False``: back into) the fit.

    Independent of the point mask: points deleted inside a subset stay deleted when
    it is included again.
    """
    partition = state.partition(args.name)
    for index in args.subsets:
        if not 0 <= index < partition.n_subsets:
            raise SessionError(f"no subset {index} in partition {args.name!r}")
    current = set(state.excluded_subsets(args.name))
    chosen = set(args.subsets)
    target = current | chosen if args.excluded else current - chosen
    if target == current:
        return _noop(
            state,
            "those subsets are already "
            + ("excluded" if args.excluded else "included"),
        )
    entries = [(name, subsets) for name, subsets in state.excluded]
    names = [name for name, _ in entries]
    record = (args.name, tuple(sorted(target)))
    if args.name in names:
        entries[names.index(args.name)] = record
    else:
        entries.append(record)
    new = replace(
        state, excluded=tuple((name, subs) for name, subs in entries if subs)
    )
    check_points(kernel, new)
    validate_model(kernel, new)
    which = ", ".join(partition.labels[i] for i in sorted(chosen))
    verb = "excluded" if args.excluded else "included"
    return new, None, _edited(f"{verb} {which} of {args.name!r}", "data", "fit")


@register("define_partition")
def define_partition(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: DefinePartitionArgs,
) -> Outcome:
    """Add a partition, or replace one no role uses and none of whose subsets is
    excluded (neutral: the fit does not change)."""
    partition = Partition.from_runs(
        kernel.data.n,
        args.runs,
        name=args.name,
        labels=args.labels,
        weights=args.weights or None,
        origin=parse_origin(args.origin),
    )
    names = [p.name for p in state.partitions]
    if args.name in names:
        roles = _roles_of(state, args.name)
        if roles:
            raise SessionError(
                f"partition {args.name!r} is in use ({', '.join(roles)}); stop "
                "using it before redefining it"
            )
        if state.excluded_subsets(args.name):
            raise SessionError(
                f"partition {args.name!r} has excluded subsets; include them before "
                "redefining it"
            )
        if _same_partition(state.partition(args.name), partition):
            return _noop(state, f"partition {args.name!r} is unchanged")
        verb = "redefined"
    else:
        verb = "defined"
    new = _replaced(state, partition)
    return new, None, _edited(
        f"{verb} partition {args.name!r} ({partition.n_subsets} subsets)",
        "partitions",
    )


@register("drop_partition")
def drop_partition(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: NameArgs
) -> Outcome:
    """Forget a partition no role uses and none of whose subsets is excluded."""
    state.partition(args.name)
    roles = _roles_of(state, args.name)
    if roles:
        raise SessionError(
            f"partition {args.name!r} is in use ({', '.join(roles)}); stop using it "
            "before dropping it"
        )
    if state.excluded_subsets(args.name):
        raise SessionError(
            f"partition {args.name!r} has excluded subsets; include them before "
            "dropping it"
        )
    new = replace(
        state, partitions=tuple(p for p in state.partitions if p.name != args.name)
    )
    return new, None, _edited(f"dropped partition {args.name!r}", "partitions")


@register("set_subset_weights")
def set_subset_weights(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: SetSubsetWeightsArgs,
) -> Outcome:
    """Relative weights, one per subset (``w_i = omega[g(i)] / sigma_i^2``).

    They take effect while the partition is the weights role (:func:`use_weights`).
    """
    partition = state.partition(args.name)
    weights = tuple(float(w) for w in args.weights)
    if weights == partition.weights:
        return _noop(state, f"subset weights of {args.name!r} are unchanged")
    new = _replaced(state, partition.with_weights(weights))
    validate_model(kernel, new)
    changed = ("partitions", "data", "fit") if state.weights_by == args.name else (
        "partitions",
    )
    return new, None, _edited(f"set subset weights of {args.name!r}", *changed)


# --- roles --------------------------------------------------------------------------


@register("use_offsets")
def use_offsets(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: UseArgs
) -> Outcome:
    """One fitted zero point per subset of a partition (``None``: one constant).

    The zero points are re-solved jointly with the sinusoids on every fit; they need
    a free mean (``settings.fit_mean``).
    """
    if args.name == state.offsets_by:
        return _noop(state, "zero points are unchanged")
    if args.name is not None:
        state.partition(args.name)
        if not state.settings.fit_mean:
            raise SessionError(
                "zero points per subset need fit_mean=True (configure it first)"
            )
    new = replace(state, offsets_by=args.name, subset_offsets=None)
    validate_model(kernel, new)
    message = (
        "one zero point" if args.name is None else f"zero points per {args.name!r}"
    )
    return new, None, _edited(message, "fit")


@register("use_variation")
def use_variation(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: UseArgs
) -> Outcome:
    """The partition behind ``set_variation``.

    Switching it — to ``None`` or to another partition — resets every varying
    component to ``"none"`` in the same action: per-subset values never outlive the
    partition whose length they have.
    """
    if args.name == state.variation_by:
        return _noop(state, "the variation partition is unchanged")
    if args.name is not None:
        state.partition(args.name)
    reset = [c for c in state.components if c.variation != "none"]
    components = tuple(
        replace(c, variation="none", subset_amplitude=None, subset_phase=None)
        if c.variation != "none"
        else c
        for c in state.components
    )
    new = replace(state, variation_by=args.name, components=components)
    validate_model(kernel, new)
    message = (
        "no variation partition"
        if args.name is None
        else f"variation per {args.name!r}"
    )
    if reset:
        message += f"; {labels(c.id for c in reset)} reset to common values"
    changed = ("components", "fit") if reset else ("partitions",)
    return new, None, _edited(message, *changed)


@register("use_weights")
def use_weights(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: UseArgs
) -> Outcome:
    """Apply a partition's subset weights (``None``: every point by its error)."""
    if args.name == state.weights_by:
        return _noop(state, "the weights are unchanged")
    if args.name is not None:
        state.partition(args.name)
    new = replace(state, weights_by=args.name)
    check_points(kernel, new)
    validate_model(kernel, new)
    message = "no subset weights" if args.name is None else f"weights of {args.name!r}"
    return new, None, _edited(message, "data", "fit")


def residual_rms_weights(
    kernel: SessionKernel, state: SessionState, derived: Derived, name: str
) -> tuple[float, ...]:
    """Subset weights ``1 / rms_s^2`` of the current residuals, normalised to mean 1.

    How the GUI produces the "1/rms²" weights (the log stores the numbers this
    returns). A subset with no point in use, or residuals of zero, gets weight 1.

    Parameters
    ----------
    kernel, state, derived
        The session's kernel, the state and its Derived (whose residuals are used).
    name : str
        The partition to weight.
    """
    partition = state.partition(name)
    used = kernel.used(state)
    residuals = np.asarray(derived.fit.residuals, dtype=np.float64)
    codes = partition.codes[used.index]
    s = partition.n_subsets
    counts = np.bincount(codes, minlength=s)
    sums = np.bincount(codes, weights=residuals * residuals, minlength=s)
    weights = np.ones(s, dtype=np.float64)
    good = counts > 0
    mean_square = np.zeros(s, dtype=np.float64)
    mean_square[good] = sums[good] / counts[good]
    good &= mean_square > 0.0
    weights[good] = 1.0 / mean_square[good]
    if good.any():
        weights[good] /= float(np.mean(weights[good]))
    return tuple(float(w) for w in weights)


# --- settings -----------------------------------------------------------------------


@register("configure")
def configure(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: ConfigureArgs
) -> Outcome:
    """Replace the requested settings wholesale.

    A change of the grid settings recomputes the grid from the points in use (the
    default pre-whitening grid of those settings); a change of the spectrum settings
    makes the kernel build a new engine (the sampling key covers them). The backend
    and the device are fixed for the life of a session.
    """
    settings = PreWhitenSettings.model_validate(args.settings)
    old = state.settings
    after, before = settings_dump(settings), settings_dump(old)
    if after == before:
        return _noop(state, "the settings are unchanged")
    for field in FIXED_FIELDS:
        if after.get(field) != before.get(field):
            raise SessionError(
                f"the {field} is fixed for the life of a session (open a new session, "
                f"or load/restore it with backend=, to change it)"
            )
    grid = state.grid
    if any(after.get(field) != before.get(field) for field in GRID_FIELDS):
        from cuperiod.prewhiten.engine import default_prewhiten_grid

        used = kernel.used(state)
        lc = LightCurve(time=used.time, value=used.value, error=used.error)
        grid = kernel.register_grid(default_prewhiten_grid(lc, settings))
    new = replace(state, settings=settings.model_copy(), grid=grid)
    check_points(kernel, new)
    validate_model(kernel, new)
    changed = sorted(key for key in after if after[key] != before.get(key))
    return new, None, _edited(
        f"changed {', '.join(changed)}", "settings", "fit"
    )


__all__ = [
    "FIXED_FIELDS",
    "GRID_FIELDS",
    "check_points",
    "configure",
    "define_partition",
    "delete_points",
    "drop_partition",
    "exclude_subsets",
    "residual_rms_weights",
    "restore_points",
    "set_subset_weights",
    "use_offsets",
    "use_variation",
    "use_weights",
]
