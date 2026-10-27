"""Reducers of the frequency-table operations (the manual loop of a session).

Each reducer is pure: it receives the kernel, the current state and its
:class:`~cuperiod.prewhiten.session.state.Derived`, and returns a new state, its
Derived when the reducer fitted (``None`` otherwise) and a report. The facade's
:meth:`~cuperiod.PreWhitenSession.do` applies what every manual edit implies — the
implicit Calculate when the reducer did not fit, ``status="searching"``, a fresh
attempt budget, cleared blocked seeds — so the reducers here do only their own edit.

Bad input or an edit the state cannot take raises :class:`SessionError` (or a
:class:`~cuperiod.prewhiten.constraints.ConstraintError`); ``do()`` turns that into a
``"refused"`` report and commits nothing. Every edit that changes the model is checked
against the constraints the next fit will receive *before* it commits, so a session
with ``recalculate_on_edit=False`` cannot hold a state no fit could honour.

``step``, ``polish``, ``prune`` and ``renumber`` live in
:mod:`~cuperiod.prewhiten.session.stepper`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import replace

import numpy as np

from cuperiod.prewhiten.constraints import (
    ConstraintError,
    flatten_ties,
    parse_tie,
)
from cuperiod.prewhiten.session import stepper as _stepper
from cuperiod.prewhiten.session.actions import (
    AddArgs,
    FixArgs,
    ImproveArgs,
    NoArgs,
    RemoveArgs,
    SeedArgs,
    SetActiveArgs,
    SetNoteArgs,
    SetValuesArgs,
    SetVariationArgs,
    TieArgs,
    register,
)
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.session.state import (
    Component,
    Derived,
    SessionError,
    SessionState,
    StepReport,
    Tie,
)

Outcome = tuple[SessionState, Derived | None, StepReport]

_TWO_PI = 2.0 * np.pi


# --- shared helpers (ops_data and variation use them too) ---------------------------


def labels(ids: Iterable[int]) -> str:
    """``"F1, F3"``."""
    return ", ".join(f"F{i}" for i in ids)


def unique(ids: Iterable[int]) -> list[int]:
    """Ids in first-seen order without repeats."""
    return list(dict.fromkeys(int(i) for i in ids))


def require(state: SessionState, ids: Iterable[int]) -> list[Component]:
    """The components named by ``ids`` (:class:`SessionError` for an unknown one)."""
    return [state.component(i) for i in unique(ids)]


def with_components(
    state: SessionState, updated: Mapping[int, Component]
) -> SessionState:
    """``state`` with the components of ``updated`` (by id) replaced in place."""
    return replace(
        state,
        components=tuple(updated.get(c.id, c) for c in state.components),
    )


def validate_model(kernel: SessionKernel, state: SessionState) -> None:
    """Raise unless the next fit of ``state`` can honour its constraints.

    Builds the constraints the kernel would pass and runs their own validation
    (ties, cycles, duplicate frequencies, subset sizes, weights, fixed values), so a
    refusal happens at the edit that caused it, whether or not an implicit Calculate
    follows.
    """
    for c in state.components:
        if c.tie is not None:
            for parent in c.tie.parents:
                state.component(parent)
    tie_ids(state)  # a cycle through inactive components is refused too
    mm = kernel.model_map(state)
    if mm is None or mm.constraints is None:
        return
    used = kernel.used(state)
    mm.constraints.validate(
        mm.frequencies,
        used.n,
        fit_mean=bool(state.settings.fit_mean),
        t_ref=kernel.t_ref,
    )


def tie_ids(
    state: SessionState,
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...], float]]:
    """Every tie of the table flattened onto untied parents, keyed by child id.

    Raises
    ------
    ConstraintError
        ``"cyclic tie: F3 -> F5 -> F3"`` (component ids, as the user wrote them).
    """
    ties = {
        c.id: (c.tie.parents, c.tie.coefficients, 0.0)
        for c in state.components
        if c.tie is not None
    }
    return flatten_ties(ties) if ties else {}


def predicted_frequency(state: SessionState, component_id: int) -> float:
    """The frequency the tie of ``component_id`` puts it at (stored parent values)."""
    flat = tie_ids(state)
    if component_id not in flat:
        return float(state.component(component_id).frequency)
    parents, coefficients, constant = flat[component_id]
    total = float(constant)
    for parent, n in zip(parents, coefficients, strict=True):
        total += n * float(state.component(parent).frequency)
    return total


def band(kernel: SessionKernel, state: SessionState) -> tuple[float, float]:
    """The frequencies a refinement may reach: the band :meth:`SessionKernel.bounds`
    clips a component inside it to (half the grid's first frequency up to its last)."""
    engine = kernel.engine(state)
    f_min, f_max = float(engine.frequency[0]), float(engine.frequency[-1])
    return max(0.5 * f_min, 1e-12), f_max


def too_close(
    kernel: SessionKernel,
    state: SessionState,
    frequency: float,
    *,
    ignore: Collection[int] = (),
    radius: float | None = None,
) -> Component | None:
    """The active component within ``radius`` (default ``collision_rayleigh / T``)
    of ``frequency``, if any (the nearest one)."""
    if radius is None:
        radius = kernel.session_settings.collision_rayleigh / kernel.baseline(state)
    nearest: Component | None = None
    best = float("inf")
    for c in state.active:
        if c.id in ignore:
            continue
        distance = abs(c.frequency - frequency)
        if distance < radius and distance < best:
            nearest, best = c, distance
    return nearest


def _collision_message(
    kernel: SessionKernel, state: SessionState, frequency: float, other: Component
) -> str:
    rayleigh = kernel.session_settings.collision_rayleigh
    return (
        f"{frequency:.9g} c/d is within {rayleigh:g} Rayleigh widths of "
        f"{other.label} ({other.frequency:.9g} c/d); pass allow_close=True to "
        "add it anyway"
    )


def _edited(
    message: str,
    changed: Iterable[str],
    *,
    component_id: int | None = None,
    frequency: float | None = None,
    statistics: Mapping[str, float] | None = None,
) -> StepReport:
    return StepReport(
        kind="edited",
        message=message,
        component_id=component_id,
        frequency=frequency,
        statistics=dict(statistics or {}),
        changed=frozenset(changed),
    )


def _noop_outcome(state: SessionState, message: str) -> Outcome:
    """An edit that changes nothing: not committed, not logged."""
    return state, None, StepReport(kind="noop", message=message)


# --- add_peak / add ---------------------------------------------------------------


@register("add_peak")
def add_peak(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: SeedArgs
) -> Outcome:
    """The tallest admissible residual peak, added without the acceptance test.

    No pre-check applies (neither ``max_frequencies`` nor the attempt budget limits a
    manual add); the stopping criteria are computed and reported but never block. A
    refinement that merges into an existing component is committed as ``"blocked"``
    (its seed joins ``blocked``, status and budget untouched), so repeated calls walk
    down the spectrum. No admissible peak, or a failed fit, is refused.
    """
    return _stepper.attempt(
        kernel, state, derived, seed=args.seed, index=args.index, accept=False
    )


@register("add")
def add(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: AddArgs
) -> Outcome:
    """A manual component at ``args.frequency`` (the facade already snapped it).

    ``refine=None`` uses ``settings.refine`` (the trial fit of a step, boxed by the
    refinement bounds); ``"none"`` keeps the frequency exactly where it is.
    """
    frequency = float(args.frequency)
    refine = args.refine if args.refine is not None else state.settings.refine
    if refine != "none":
        low, high = band(kernel, state)
        if not low <= frequency <= high:
            raise SessionError(
                f"{frequency:.9g} c/d is outside the frequency grid "
                f"[{low:.9g}, {high:.9g}]; add it with refine='none'",
                hint="exact",
            )
    if not args.allow_close:
        other = too_close(kernel, state, frequency)
        if other is not None:
            raise SessionError(
                _collision_message(kernel, state, frequency, other),
                hint="allow_close",
            )
    new_id = state.next_id
    component = Component(
        id=new_id, frequency=frequency, amplitude=0.0, phase=0.0, origin="manual"
    )
    staged = replace(
        state, components=(*state.components, component), next_id=new_id + 1
    )
    mm = kernel.model_map(staged)
    frequencies = mm.frequencies if mm is not None else staged.active_frequencies
    fit = kernel.fit(staged, frequencies, refine=refine, covariance=False, mm=mm)
    position = mm.position(new_id) if mm is not None else len(staged.active) - 1
    landed = float(fit.frequency[position])
    if refine != "none" and not args.allow_close:
        # The step's merge guard: a refinement that slid onto an existing component
        # found that component's residual, not a new mode.
        other = too_close(
            kernel, state, landed, ignore=(new_id,), radius=kernel.separation(state)
        )
        if other is not None:
            # Both keywords would add it, but only an exact add keeps the frequency
            # the user chose rather than the one the refinement slid onto.
            raise SessionError(
                f"the refinement moved {frequency:.9g} c/d onto {other.label} "
                f"({other.frequency:.9g} c/d); add it with refine='none' or "
                "allow_close=True",
                hint="exact",
            )
    delta_bic = float(fit.bic - state.stats.bic)
    staged = with_components(staged, {new_id: replace(component, delta_bic=delta_bic)})
    new, new_derived = kernel.commit_fit(staged, fit, mm=mm)
    report = _edited(
        f"F{new_id} added at {landed:.9g} c/d",
        ("components", "fit", "ids"),
        component_id=new_id,
        frequency=landed,
        statistics={
            "amplitude": float(abs(fit.amplitude[position])),
            "delta_bic": delta_bic,
            "rms": float(fit.rms),
        },
    )
    return new, new_derived, report


# --- remove / set_active ------------------------------------------------------------


@register("remove")
def remove(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: RemoveArgs
) -> Outcome:
    """Delete rows. A tie parent is refused unless ``cascade``, which dissolves the
    ties of its children (each child becomes free at its current value)."""
    doomed = {c.id for c in require(state, args.ids)}
    children = [
        c
        for c in state.components
        if c.id not in doomed
        and c.tie is not None
        and doomed.intersection(c.tie.parents)
    ]
    if children and not args.cascade:
        child = children[0]
        assert child.tie is not None
        parent = next(p for p in child.tie.parents if p in doomed)
        raise SessionError(
            f"F{parent} is a tie parent of {child.label} ({child.label} = "
            f"{child.tie.text()}); remove with cascade=True to dissolve the tie",
            hint="cascade",
        )
    untied = {c.id: replace(c, tie=None) for c in children}
    new = replace(
        state,
        components=tuple(
            untied.get(c.id, c) for c in state.components if c.id not in doomed
        ),
    )
    validate_model(kernel, new)
    message = f"removed {labels(sorted(doomed))}"
    if untied:
        message += f"; untied {labels(untied)}"
    return new, None, _edited(message, ("components", "fit"))


@register("set_active")
def set_active(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: SetActiveArgs
) -> Outcome:
    """Inactive rows leave the model and the peak search but keep their values; a
    tie child holds an inactive parent as a constant."""
    targets = [c for c in require(state, args.ids) if c.active != args.active]
    if not targets:
        state_word = "active" if args.active else "inactive"
        return _noop_outcome(state, f"{labels(args.ids)} already {state_word}")
    new = with_components(
        state, {c.id: replace(c, active=bool(args.active)) for c in targets}
    )
    validate_model(kernel, new)
    verb = "activated" if args.active else "deactivated"
    return new, None, _edited(
        f"{verb} {labels(c.id for c in targets)}", ("components", "fit")
    )


# --- fix / set_values / tie -------------------------------------------------------


@register("fix")
def fix(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: FixArgs
) -> Outcome:
    """Hold (``True``) or free (``False``) frequency, amplitude and phase at their
    current values; ``None`` leaves a quantity as it is."""
    if args.frequency is None and args.amplitude is None and args.phase is None:
        raise SessionError("fix needs frequency=, amplitude= or phase=")
    updated: dict[int, Component] = {}
    for c in require(state, args.ids):
        if args.frequency and c.tie is not None:
            raise SessionError(
                f"{c.label} is tied ({c.label} = {c.tie.text()}); untie it before "
                "fixing its frequency"
            )
        if args.amplitude and c.variation in ("amplitude", "both"):
            raise SessionError(
                f"{c.label} has one amplitude per subset, which cannot be fixed; "
                "set its variation to 'none' or 'phase' first"
            )
        if args.phase and c.variation in ("phase", "both"):
            raise SessionError(
                f"{c.label} has one phase per subset, which cannot be fixed; set "
                "its variation to 'none' or 'amplitude' first"
            )
        edited = replace(
            c,
            fix_frequency=c.fix_frequency
            if args.frequency is None
            else bool(args.frequency),
            fix_amplitude=c.fix_amplitude
            if args.amplitude is None
            else bool(args.amplitude),
            fix_phase=c.fix_phase if args.phase is None else bool(args.phase),
        )
        if edited != c:
            updated[c.id] = edited
    if not updated:
        return _noop_outcome(state, "nothing to fix or free")
    new = with_components(state, updated)
    validate_model(kernel, new)
    return new, None, _edited(
        f"fixed/freed {labels(updated)}", ("components", "fit")
    )


@register("set_values")
def set_values(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: SetValuesArgs
) -> Outcome:
    """Write values. A frequency unties the row; nothing is fixed (a free value is
    re-solved by the implicit Calculate — fix it to keep it)."""
    c = state.component(args.id)
    changes: dict[str, object] = {}
    if args.frequency is not None:
        frequency = float(args.frequency)
        if c.active:
            other = too_close(kernel, state, frequency, ignore=(c.id,))
            if other is not None:
                raise SessionError(
                    f"{frequency:.9g} c/d is within "
                    f"{kernel.session_settings.collision_rayleigh:g} Rayleigh widths "
                    f"of {other.label} ({other.frequency:.9g} c/d)"
                )
        changes["frequency"] = frequency
        changes["tie"] = None
    if args.amplitude is not None:
        changes["amplitude"] = float(args.amplitude)
    if args.phase is not None:
        changes["phase"] = float(np.mod(float(args.phase), _TWO_PI))
    edited = replace(c, **changes)  # type: ignore[arg-type]
    if edited == c:
        return _noop_outcome(state, f"{c.label} already has those values")
    new = with_components(state, {c.id: edited})
    validate_model(kernel, new)
    return new, None, _edited(
        f"set values of {c.label}", ("components", "fit"), component_id=c.id
    )


@register("tie")
def tie(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: TieArgs
) -> Outcome:
    """Tie a frequency to an integer combination of others (``None`` unties).

    The expression is parsed (never evaluated), its parents must exist, the tie may
    not close a cycle, the predicted frequency must lie in the grid band and — unless
    ``allow_close`` — away from every other active component. The row's frequency
    becomes the predicted value; a fixed frequency gives way to the tie.
    """
    c = state.component(args.id)
    if args.expression is None:
        if c.tie is None:
            return _noop_outcome(state, f"{c.label} is not tied")
        new = with_components(state, {c.id: replace(c, tie=None)})
        validate_model(kernel, new)
        return new, None, _edited(
            f"untied {c.label}", ("components", "fit"), component_id=c.id
        )
    parents, coefficients = parse_tie(args.expression, child=c.id)
    for parent in parents:
        state.component(parent)
    link = Tie(parents=tuple(parents), coefficients=tuple(coefficients))
    if c.tie == link:
        return _noop_outcome(state, f"{c.label} is already {link.text()}")
    staged = with_components(
        state, {c.id: replace(c, tie=link, fix_frequency=False)}
    )
    predicted = predicted_frequency(staged, c.id)
    low, high = band(kernel, state)
    if not low <= predicted <= high:
        raise ConstraintError(
            f"tie puts {c.label} at {predicted:.9g} c/d, outside the frequency grid "
            f"[{low:.9g}, {high:.9g}]"
        )
    if c.active and not args.allow_close:
        other = too_close(kernel, staged, predicted, ignore=(c.id,))
        if other is not None:
            raise SessionError(
                f"tie puts {c.label} at {predicted:.9g} c/d, within "
                f"{kernel.session_settings.collision_rayleigh:g} Rayleigh widths of "
                f"{other.label}; pass allow_close=True to tie it anyway",
                hint="allow_close",
            )
    new = with_components(
        state,
        {c.id: replace(c, tie=link, fix_frequency=False, frequency=predicted)},
    )
    validate_model(kernel, new)
    return new, None, _edited(
        f"tied {c.label} = {link.text()} ({predicted:.9g} c/d)",
        ("components", "fit"),
        component_id=c.id,
        frequency=predicted,
    )


# --- variation --------------------------------------------------------------------


@register("set_variation")
def set_variation(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    args: SetVariationArgs,
) -> Outcome:
    """One amplitude and/or phase per subset of the variation partition.

    ``"phase"`` and ``"both"`` also fix the frequency when ``hold_frequency`` (a
    frequency free alongside per-subset phases is barely determined); a tied row
    keeps its tie. Stored per-subset values of a quantity that keeps varying are kept
    as starting values.
    """
    if state.variation_by is None:
        raise SessionError(
            "set_variation needs a variation partition: use_variation(name) first"
        )
    mode = args.mode
    varies_a = mode in ("amplitude", "both")
    varies_p = mode in ("phase", "both")
    updated: dict[int, Component] = {}
    for c in require(state, args.ids):
        if varies_a and c.fix_amplitude:
            raise SessionError(
                f"{c.label} has a fixed amplitude, which cannot vary per subset; "
                "free it first"
            )
        if varies_p and c.fix_phase:
            raise SessionError(
                f"{c.label} has a fixed phase, which cannot vary per subset; "
                "free it first"
            )
        hold = varies_p and args.hold_frequency and c.tie is None
        edited = replace(
            c,
            variation=mode,
            subset_amplitude=c.subset_amplitude if varies_a else None,
            subset_phase=c.subset_phase if varies_p else None,
            fix_frequency=True if hold else c.fix_frequency,
        )
        if edited != c:
            updated[c.id] = edited
    if not updated:
        return _noop_outcome(state, f"{labels(args.ids)} already vary as {mode!r}")
    new = with_components(state, updated)
    validate_model(kernel, new)
    return new, None, _edited(
        f"{labels(updated)}: variation {mode!r} per {state.variation_by!r}",
        ("components", "fit"),
    )


# --- calculate / improve ------------------------------------------------------------


@register("calculate")
def calculate(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: NoArgs
) -> Outcome:
    """Amplitudes, phases and zero points at the current frequencies (held).

    Unlike the implicit Calculate after an edit, this one solves with the covariance,
    so the table gets frequency, amplitude and phase errors without a re-fit (0 for
    a fixed quantity; a held but free frequency keeps its error).
    """
    new, new_derived = kernel.recalculate(state, covariance=True)
    return new, new_derived, _edited(
        "calculated amplitudes, phases and zero points",
        ("components", "fit"),
        statistics={"rms": float(new_derived.fit.rms)},
    )


@register("improve")
def improve(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: ImproveArgs
) -> Outcome:
    """Non-linear refinement of every (``ids=None``) or the selected frequencies.

    All: the final polish of the automatic path (``"simultaneous"`` up to
    ``max_simultaneous`` components, else ``"cyclic"``), boxed by the refinement
    bounds around the current values. Selected: one ``"simultaneous"`` fit in which
    every unselected free frequency is held for this fit only, then a Calculate under
    the persistent constraints — so the reported errors never pretend an unselected
    frequency was fixed.
    """
    active = state.active
    if not active:
        return _noop_outcome(state, "nothing to improve")
    if args.ids is None:
        mm = kernel.model_map(state)
        frequencies = np.ascontiguousarray(
            derived.fit.frequency if mm is None else mm.frequencies
        )
        fit = kernel.fit(
            state,
            frequencies,
            refine=kernel.polish_refine(state, len(active)),
            covariance=True,
            mm=mm,
        )
        new, new_derived = kernel.commit_fit(state, fit, mm=mm)
        message = f"improved all {len(active)} frequencies"
    else:
        selected = require(state, args.ids)
        for c in selected:
            if not c.active:
                raise SessionError(f"{c.label} is inactive")
        movable = [c for c in selected if c.tie is None and not c.fix_frequency]
        if not movable:
            raise SessionError(
                f"{labels(c.id for c in selected)} have no free frequency (fixed or "
                "tied: improve the parents of a tie)"
            )
        chosen = {c.id for c in selected}
        hold = frozenset(c.id for c in active if c.id not in chosen)
        mm = kernel.model_map(state, hold=hold)
        if mm is None:
            # Every active row selected on a plain state: nothing is held, but the
            # fit's values still go back by id, which needs the map.
            from cuperiod.prewhiten.session.modelmap import build

            mm = build(
                state, kernel.used(state), t_ref=kernel.t_ref,
                min_subset_points=kernel.session_settings.min_subset_points,
            )
        fit = kernel.fit(
            state, mm.frequencies, refine="simultaneous", covariance=False, mm=mm
        )
        from cuperiod.prewhiten.session.modelmap import apply

        update = apply(state, fit, mm)
        staged = replace(state, components=update.components)
        # The covariance comes from this solve, under the persistent constraints, not
        # from the fit above, which held the unselected frequencies.
        new, new_derived = kernel.recalculate(staged, covariance=True)
        message = f"improved {labels(c.id for c in movable)}"
    return new, new_derived, _edited(
        message,
        ("components", "fit"),
        statistics={"rms": float(new_derived.fit.rms)},
    )


# --- notes --------------------------------------------------------------------------


@register("set_note")
def set_note(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: SetNoteArgs
) -> Outcome:
    """Free text on a row (neutral: the model does not change)."""
    c = state.component(args.id)
    if c.note == args.note:
        return _noop_outcome(state, f"{c.label} already has that note")
    new = with_components(state, {c.id: replace(c, note=args.note)})
    return new, None, _edited(
        f"note on {c.label}", ("components",), component_id=c.id
    )


__all__ = [
    "add",
    "add_peak",
    "band",
    "calculate",
    "fix",
    "improve",
    "labels",
    "predicted_frequency",
    "remove",
    "require",
    "set_active",
    "set_note",
    "set_values",
    "set_variation",
    "tie",
    "tie_ids",
    "too_close",
    "unique",
    "validate_model",
    "with_components",
]
