"""From a session state to the constraints of one fit, and back.

The frequency table speaks in component *ids*; a fit speaks in *positions* of its
frequency vector. :func:`build` maps the active components of a state onto a
:class:`~cuperiod.prewhiten.constraints.FitConstraints` (fixed values, ties with
inactive parents folded into constants, per-subset variation, zero-point groups,
relative weights and starting phases), and :func:`apply` writes a fit's values back
onto the components it came from.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.prewhiten.constraints import (
    ComponentConstraint,
    FitConstraints,
    FitStart,
    FrequencyTie,
)
from cuperiod.prewhiten.fit import MultiSineFit
from cuperiod.prewhiten.session.state import (
    Component,
    SessionError,
    SessionState,
    UsedData,
)


@dataclass(frozen=True, eq=False)
class ModelMap:
    """How the active components of a state map onto one fit.

    Attributes
    ----------
    constraints : FitConstraints or None
        ``None`` when the mapped constraints are trivial (the automatic loop's plain
        fit applies).
    ids : tuple of int
        Component id at each fit position (the active components, table order).
    frequencies : numpy.ndarray
        Their current frequencies (a tie with only inactive parents: its constant).
    start : FitStart or None
        Starting values for the non-linear phases (the stored state).
    inactive_parent : frozenset of int
        Ids whose tie holds an inactive parent as a constant.
    offset_subsets, variation_subsets : numpy.ndarray or None
        Partition subset of each compacted group code.
    n_offset_subsets, n_variation_subsets : int
        Subset counts of the full partitions.
    """

    constraints: FitConstraints | None
    ids: tuple[int, ...]
    frequencies: FloatArray
    start: FitStart | None
    inactive_parent: frozenset[int] = frozenset()
    offset_subsets: IntArray | None = None
    variation_subsets: IntArray | None = None
    n_offset_subsets: int = 0
    n_variation_subsets: int = 0

    def position(self, component_id: int) -> int:
        """Fit position of a component id."""
        try:
            return self.ids.index(component_id)
        except ValueError:
            raise SessionError(f"F{component_id} is not in this fit") from None

    def extended(self, component_id: int) -> ModelMap:
        """This map with one more (appended, free) component id."""
        return replace(self, ids=(*self.ids, component_id))


@dataclass(frozen=True)
class StateUpdate:
    """The component values, offset and zero points a fit writes back."""

    components: tuple[Component, ...]
    offset: float
    subset_offsets: tuple[float, ...] | None


def _frequency_constraint(
    component: Component,
    state: SessionState,
    position: dict[int, int],
    hold: Collection[int],
) -> tuple[Literal["free", "fixed"] | FrequencyTie, float, bool]:
    """``(constraint, start frequency, holds an inactive parent)`` of a component."""
    if component.tie is not None:
        parents: list[int] = []
        coefficients: list[int] = []
        constant = 0.0
        inactive = False
        for parent_id, n in zip(
            component.tie.parents, component.tie.coefficients, strict=True
        ):
            if parent_id in position:
                parents.append(position[parent_id])
                coefficients.append(n)
            else:
                # An inactive parent keeps its value and holds the child through it.
                constant += n * state.component(parent_id).frequency
                inactive = True
        if not parents:
            return "fixed", constant, inactive
        tie = FrequencyTie(
            parents=tuple(parents), coefficients=tuple(coefficients), constant=constant
        )
        return tie, component.frequency, inactive
    if component.fix_frequency or component.id in hold:
        return "fixed", component.frequency, False
    return "free", component.frequency, False


def _subset_start(
    values: tuple[float, ...] | None, headline: float, subsets: IntArray
) -> FloatArray:
    """Per compacted subset: the stored value, else the headline value."""
    if values is None:
        return np.full(subsets.size, headline, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64)[subsets]
    return np.where(np.isfinite(out), out, headline)


def build(
    state: SessionState,
    used: UsedData,
    *,
    t_ref: float,
    extra_free: int = 0,
    hold: Collection[int] = frozenset(),
    min_subset_points: int = 10,
) -> ModelMap:
    """The fit constraints of ``state``'s active components on the points ``used``.

    Parameters
    ----------
    state : SessionState
    used : UsedData
        From :func:`~cuperiod.prewhiten.session.sampling.used_data` of ``state``.
    t_ref : float
        The session's frozen phase epoch (a fixed phase is meaningless without it).
    extra_free : int, default 0
        Free components appended after the active ones (a peak candidate).
    hold : collection of int
        Component ids whose frequency is held for this fit only.
    min_subset_points : int, default 10
        Fewest points a variation subset may have.

    Returns
    -------
    ModelMap
    """
    del t_ref  # the fit receives it; the constraints are frame-free
    active = state.active
    ids = tuple(c.id for c in active)
    position = {cid: i for i, cid in enumerate(ids)}
    constraints: list[ComponentConstraint] = []
    frequencies = np.empty(len(active), dtype=np.float64)
    inactive_parent: set[int] = set()
    for i, c in enumerate(active):
        frequency, initial, inactive = _frequency_constraint(c, state, position, hold)
        frequencies[i] = initial
        if inactive:
            inactive_parent.add(c.id)
        constraints.append(
            ComponentConstraint(
                frequency=frequency,
                amplitude=c.amplitude if c.fix_amplitude else None,
                phase=c.phase if c.fix_phase else None,
                variation=c.variation,
            )
        )
    constraints.extend(ComponentConstraint() for _ in range(extra_free))

    varying = any(c.variation != "none" for c in active)
    variation_groups = used.variation_codes if varying else None
    offset_groups = used.offset_codes if state.offsets_by is not None else None
    k = len(active) + extra_free
    start: FitStart | None = None
    if any(not c.is_free for c in constraints):
        amplitude = np.zeros(k, dtype=np.float64)
        phase = np.zeros(k, dtype=np.float64)
        amplitude[: len(active)] = [c.amplitude for c in active]
        phase[: len(active)] = [c.phase for c in active]
        subset_amplitude = subset_phase = None
        if variation_groups is not None and used.variation_subsets is not None:
            subsets = used.variation_subsets
            subset_amplitude = np.full((k, subsets.size), np.nan)
            subset_phase = np.full((k, subsets.size), np.nan)
            for i, c in enumerate(active):
                if c.variation in ("amplitude", "both"):
                    subset_amplitude[i] = _subset_start(
                        c.subset_amplitude, c.amplitude, subsets
                    )
                if c.variation in ("phase", "both"):
                    subset_phase[i] = _subset_start(c.subset_phase, c.phase, subsets)
        start = FitStart(
            amplitude=amplitude,
            phase=phase,
            subset_amplitude=subset_amplitude,
            subset_phase=subset_phase,
        )
    mapped = FitConstraints(
        components=tuple(constraints),
        offset_groups=offset_groups,
        variation_groups=variation_groups,
        point_weights=used.point_weights,
        min_subset_points=int(min_subset_points),
        start=start,
        # Refusals then name the rows the user sees, not fit positions (an appended
        # candidate is named by the id it would get).
        labels=tuple(f"F{cid}" for cid in ids)
        + tuple(f"F{state.next_id + j}" for j in range(extra_free)),
    )
    offsets = (
        state.partition(state.offsets_by).n_subsets if state.offsets_by else 0
    )
    variations = (
        state.partition(state.variation_by).n_subsets if state.variation_by else 0
    )
    return ModelMap(
        constraints=None if mapped.is_trivial() else mapped,
        ids=ids,
        frequencies=frequencies,
        start=start,
        inactive_parent=frozenset(inactive_parent),
        offset_subsets=used.offset_subsets,
        variation_subsets=used.variation_subsets,
        n_offset_subsets=offsets,
        n_variation_subsets=variations,
    )


def _expand(
    values: FloatArray, subsets: IntArray | None, total: int
) -> tuple[float, ...]:
    """Compacted per-subset values onto the full partition (NaN where empty)."""
    out = np.full(total, np.nan)
    if subsets is None:
        out[: values.size] = values
    else:
        out[subsets] = values
    return tuple(float(v) for v in out)


def _held_amplitude(mm: ModelMap, position: int) -> float | None:
    """The amplitude the fit held at ``position`` when it reports another, else None.

    A fixed amplitude on a component whose phase varies per subset is the amplitude
    ``A`` of every subset, while the fit's headline is their coherent mean
    ``A |mean exp(i phi_s)|``, smaller whenever the subset phases differ. Writing the
    headline back would fix a smaller amplitude at the next fit, and again at every
    fit after it, so the component keeps ``A`` (bit for bit) and the fit's result
    reports the headline.
    """
    if mm.constraints is None or position >= len(mm.constraints.components):
        return None
    constraint = mm.constraints.components[position]
    if constraint.amplitude is None or not constraint.varies_phase:
        return None
    return float(constraint.amplitude)


def apply(state: SessionState, fit: MultiSineFit, mm: ModelMap) -> StateUpdate:
    """Write ``fit`` (built from ``mm``) back onto the components of ``state``.

    Frequencies, signed amplitudes and phases go to the ids of ``mm``; per-subset
    amplitudes and phases are expanded onto the full variation partition (NaN for a
    subset with no point in use); components outside the fit are unchanged. A fixed
    amplitude under phase variation keeps the value the fit held, not the fit's
    smaller coherent headline (see :func:`_held_amplitude`).
    """
    detail = fit.detail
    position = {cid: i for i, cid in enumerate(mm.ids)}
    components: list[Component] = []
    for c in state.components:
        i = position.get(c.id)
        if i is None:
            components.append(c)
            continue
        subset_amplitude = subset_phase = None
        if detail is not None and c.variation != "none":
            subsets, total = mm.variation_subsets, mm.n_variation_subsets
            amplitudes, phases = detail.subset_amplitude, detail.subset_phase
            if c.variation in ("amplitude", "both") and amplitudes is not None:
                subset_amplitude = _expand(amplitudes[i], subsets, total)
            if c.variation in ("phase", "both") and phases is not None:
                subset_phase = _expand(phases[i], subsets, total)
        held = _held_amplitude(mm, i)
        components.append(
            replace(
                c,
                frequency=float(fit.frequency[i]),
                amplitude=float(fit.amplitude[i]) if held is None else held,
                phase=float(fit.phase[i]),
                subset_amplitude=subset_amplitude,
                subset_phase=subset_phase,
            )
        )
    subset_offsets = None
    if (
        state.offsets_by is not None
        and detail is not None
        and detail.subset_offset is not None
    ):
        subset_offsets = _expand(
            detail.subset_offset, mm.offset_subsets, mm.n_offset_subsets
        )
    return StateUpdate(
        components=tuple(components),
        offset=float(fit.offset),
        subset_offsets=subset_offsets,
    )


__all__ = ["ModelMap", "StateUpdate", "apply", "build"]
