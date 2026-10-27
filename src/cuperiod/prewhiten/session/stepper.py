"""The automatic path of a session: one legacy loop iteration per ``step``.

:func:`step` is exactly one iteration of the automatic extraction loop — the same
pre-checks, the same peak search, trial fit, collision guard, false-alarm
probability, trial spectrum, signal-to-noise and acceptance test, calling the same
helpers from :mod:`cuperiod.prewhiten.engine` — so a session stepped to its stopping
rule and finalized is :func:`cuperiod.prewhiten`, number for number. Every attempt
that chose a peak logs its seed (the sub-grid peak frequency) and grid index:
replaying them with ``step(seed=, index=)`` skips the peak search, and since the trial
fit depends only on the data and the candidate frequencies, repeats the attempt
bitwise on any spectrum backend.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten import engine as _engine
from cuperiod.prewhiten.fit import MultiSineFit
from cuperiod.prewhiten.session.actions import NoArgs, SeedArgs, register
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.session.state import (
    Component,
    Derived,
    SessionError,
    SessionState,
    StepReport,
    Tie,
)
from cuperiod.prewhiten.spectrum import _ILL_CONDITIONED, _SpectrumReadout

Outcome = tuple[SessionState, Derived | None, StepReport]

_NO_SEED: dict[str, Any] = {"seed": None, "index": None}


def _power_at(scan: _SpectrumReadout, index: int) -> float:
    """Normalized power at one grid index of a scan or a full spectrum."""
    reader = getattr(scan, "power_at", None)
    if reader is not None:
        return float(reader(index))
    return float(scan.power[index])  # type: ignore[attr-defined]


def _stopped(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    reason: str,
    *,
    iterations: int,
    resolved: dict[str, Any],
) -> Outcome:
    new = replace(
        state,
        serial=kernel.next_serial(),
        status="stopped",
        stop_reason=reason,
        n_iterations=iterations,
    )
    report = StepReport(
        kind="stopped",
        message=reason,
        changed=frozenset({"status"}),
        resolved=resolved,
    )
    return new, derived, report


def detectable(
    kernel: SessionKernel,
    state: SessionState,
    trial: MultiSineFit,
    previous: MultiSineFit,
) -> float:
    """The amplitude the data can vouch for at the newest trial frequency.

    Where the sampling can barely separate a sinusoid from the constant the
    least-squares amplitude is noise divided by a tiny eigenvalue. The drop in weighted
    mean-square residual it buys is bounded whatever the conditioning, and
    ``sqrt(2 * drop)`` is the amplitude of a sinusoid explaining that much; the smaller
    of the two is what the stopping criteria may believe.
    """
    amplitude = float(trial.amplitude[-1])
    frequency = float(trial.frequency[-1])
    if kernel.engine(state)._conditioning_at(frequency) >= _ILL_CONDITIONED:
        return amplitude
    drop = max(previous.rss - trial.rss, 0.0) / kernel.weight_sum(state)
    return min(amplitude, float(np.sqrt(2.0 * drop)))


def attempt(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    *,
    seed: float | None,
    index: int | None,
    accept: bool,
) -> tuple[SessionState, Derived | None, StepReport]:
    """One extraction attempt (shared by ``step`` and ``add_peak``).

    With ``accept`` (``step``) every outcome after the peak search commits
    ``n_iterations + 1``: no peak left or residuals at round-off (stopped), a failed
    trial fit (stopped), a candidate merged into an existing component (blocked), a
    failed stopping criterion (rejected) or a new component (accepted). Without
    ``accept`` (``add_peak``) the criteria inform but never block, there being no peak
    or a failed fit raise the refusal the facade reports, and an accepted component is
    a manual one.

    Raises
    ------
    SessionError
        (``add_peak`` only) when no admissible peak remains.
    """
    cfg = state.settings
    iterations = state.n_iterations + 1
    scan = derived.scan
    fit = derived.fit
    n = fit.n_components
    parts = [fit.frequency, np.asarray(state.blocked, dtype=np.float64)]
    inactive = state.inactive_frequencies
    if inactive.size:
        parts.append(inactive)
    avoid = np.concatenate(parts)
    separation = kernel.separation(state)
    warning = ""
    if seed is None:
        found = scan.peak_index(exclude=avoid, separation=separation)
        if found is None:
            reason = "no peak resolved from the existing components remains"
            if not accept:
                raise SessionError(reason)
            return _stopped(
                kernel, state, derived, reason, iterations=iterations,
                resolved=_NO_SEED,
            )
        if not float(scan.amplitude[found]) > kernel.roundoff(state):
            reason = "residuals are at numerical round-off"
            if not accept:
                raise SessionError(reason)
            return _stopped(
                kernel, state, derived, reason, iterations=iterations,
                resolved=_NO_SEED,
            )
        index = found
        f_guess, _ = scan.refine_peak(index)
    else:
        if index is None or not 0 <= index < scan.size:
            raise SessionError(f"grid index {index!r} is outside the grid")
        f_guess = float(seed)
        here = scan.peak_index(exclude=avoid, separation=separation)
        if here is None or abs(here - index) > 1:
            warning = f"logged seed {seed!r} is not this run's tallest admissible peak"
    resolved = {"seed": float(f_guess), "index": int(index)}

    mm = kernel.model_map(state, extra_free=1)
    base = fit.frequency if mm is None else mm.frequencies
    candidates = np.append(base, f_guess)
    try:
        trial = kernel.fit(
            state, candidates, refine=cfg.refine, covariance=False, mm=mm
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        if not accept:
            raise
        return _stopped(
            kernel, state, derived, f"fit failed: {exc}", iterations=iterations,
            resolved=resolved,
        )
    new_frequency = float(trial.frequency[-1])
    if n and float(np.min(np.abs(trial.frequency[:-1] - new_frequency))) < separation:
        # The refinement merged the candidate into an existing component: the peak
        # was that component's residual, not a new mode. Block it and look further
        # down the spectrum rather than abandoning the run.
        new = replace(
            state,
            serial=kernel.next_serial(),
            blocked=(*state.blocked, f_guess),
            n_iterations=iterations,
        )
        report = StepReport(
            kind="blocked",
            message=(
                f"peak at {f_guess:.9g} c/d merged into an existing component; "
                "blocked"
            ),
            frequency=float(f_guess),
            changed=frozenset({"status"}),
            warning=warning,
            resolved=resolved,
        )
        return new, derived, report

    # Deferred until the candidate survives the collision guard: a discarded
    # candidate never uses its false-alarm probability.
    used = kernel.used(state)
    fap = _engine._peak_fap(
        used.time, used.error_eff, _power_at(scan, index), f_max=kernel.f_max(state)
    )
    trial_scan = kernel.engine(state)._scan(trial.residuals)
    amplitude = detectable(kernel, state, trial, fit)
    noise = trial_scan.noise_at(
        new_frequency, window=cfg.snr_window, estimator=cfg.noise_estimator
    )
    snr = amplitude / noise if np.isfinite(noise) and noise > 0.0 else float("nan")
    delta_bic = trial.bic - fit.bic
    ok, reason = _engine._accept(
        snr=snr, fap=fap, delta_bic=delta_bic, amplitude=amplitude, settings=cfg
    )
    statistics = {
        "snr": float(snr),
        "fap": float(fap),
        "delta_bic": float(delta_bic),
        "amplitude": float(amplitude),
        "rms": float(trial.rms),
    }
    if accept and not ok:
        new, _, _ = _stopped(
            kernel, state, derived, reason, iterations=iterations, resolved=resolved
        )
        report = StepReport(
            kind="rejected",
            message=reason,
            frequency=new_frequency,
            statistics=statistics,
            changed=frozenset({"status"}),
            warning=warning,
            resolved=resolved,
        )
        return new, derived, report

    new_id = state.next_id
    component = Component(
        id=new_id,
        frequency=new_frequency,
        amplitude=float(trial.amplitude[-1]),
        phase=float(trial.phase[-1]),
        origin="auto" if accept else "manual",
        fap=float(fap),
        delta_bic=float(delta_bic),
    )
    changes: dict[str, Any] = {
        "components": (*state.components, component),
        "next_id": new_id + 1,
        "n_iterations": iterations,
    }
    if not accept:
        changes.update(status="searching", stop_reason="", budget_origin=iterations)
    new, new_derived = kernel.commit_fit(
        state,
        trial,
        mm=None if mm is None else mm.extended(new_id),
        spectrum=trial_scan,
        **changes,
    )
    message = f"F{new_id} at {new_frequency:.9g} c/d"
    if np.isfinite(snr):
        message += f" (S/N {snr:.2f})"
    if not accept and not ok:
        message += f"; would fail the stopping rule: {reason}"
    report = StepReport(
        kind="accepted",
        message=message,
        component_id=new_id,
        frequency=new_frequency,
        statistics=statistics,
        changed=frozenset({"components", "fit", "ids"}),
        warning=warning,
        resolved=resolved,
    )
    return new, new_derived, report


@register("step")
def step(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: SeedArgs
) -> Outcome:
    """One iteration of the automatic extraction loop (pre-checks, then an attempt)."""
    cfg = state.settings
    if state.status == "stopped":
        return state, None, StepReport(kind="noop", message="the search has stopped")
    n = len(state.active)
    max_attempts = cfg.max_frequencies + _engine._MAX_REJECTED_CANDIDATES
    attempts = state.n_iterations - state.budget_origin
    if not (n < cfg.max_frequencies and attempts < max_attempts):
        if cfg.max_frequencies == 0:
            reason = "max_frequencies is 0"
        elif attempts >= max_attempts:
            reason = "too many candidates merged with existing components"
        else:
            reason = f"reached max_frequencies ({cfg.max_frequencies})"
        return _stopped(
            kernel, state, derived, reason, iterations=state.n_iterations,
            resolved=_NO_SEED,
        )
    # Keep at least half the points as degrees of freedom: past that, every sinusoid
    # lowers the residual spectrum the next one is judged against and a sparse curve
    # runs away to an interpolating, meaningless solution.
    n_points = kernel.used(state).n
    m_next = 3 * (n + 1) + 1
    if 2 * m_next > n_points:
        reason = (
            f"too few points ({n_points}) to constrain another component "
            f"({m_next} parameters)"
        )
        return _stopped(
            kernel, state, derived, reason, iterations=state.n_iterations,
            resolved=_NO_SEED,
        )
    return attempt(
        kernel, state, derived, seed=args.seed, index=args.index, accept=True
    )


@register("polish")
def polish(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: NoArgs
) -> Outcome:
    """Re-fit every frequency jointly (cyclically past ``max_simultaneous``)."""
    k = len(state.active)
    if k == 0:
        return state, None, StepReport(kind="noop", message="nothing to polish")
    mm = kernel.model_map(state)
    frequencies = np.ascontiguousarray(
        derived.fit.frequency if mm is None else mm.frequencies
    )
    fit = kernel.fit(
        state, frequencies, refine=kernel.polish_refine(state, k), covariance=True,
        mm=mm,
    )
    new, new_derived = kernel.commit_fit(state, fit, mm=mm)
    report = StepReport(
        kind="edited",
        message=f"polished {k} frequenc{'y' if k == 1 else 'ies'}",
        statistics={"rms": float(fit.rms)},
        changed=frozenset({"components", "fit"}),
    )
    return new, new_derived, report


def protected_ids(state: SessionState, *, protect_manual: bool) -> frozenset[int]:
    """Components ``prune`` never removes.

    Varying components always (their headline amplitude is a coherent mean that
    collapses when phases wander); with ``protect_manual`` also manual, fixed and
    tied components and the parents of a tie.
    """
    out = {c.id for c in state.components if c.variation != "none"}
    if protect_manual:
        for c in state.components:
            if (
                c.origin == "manual"
                or c.fix_frequency
                or c.fix_amplitude
                or c.fix_phase
                or c.tie is not None
            ):
                out.add(c.id)
            if c.tie is not None:
                out.update(c.tie.parents)
    return frozenset(out)


def significance_amplitude(fit: MultiSineFit) -> FloatArray:
    """The amplitude each component's S/N is judged by: ``|A|``, or ``A_rms``."""
    amplitude = np.abs(fit.amplitude)
    detail = fit.detail
    if detail is not None and detail.constraints.has_variation:
        comps = detail.constraints.expanded(fit.n_components)
        rms = np.asarray(detail.rms_amplitude, dtype=np.float64)
        varying = np.array([c.variation != "none" for c in comps], dtype=bool)
        amplitude = np.where(varying, rms, amplitude)
    out: FloatArray = amplitude
    return out


def component_snr(
    scan: _SpectrumReadout, fit: MultiSineFit, settings: Any
) -> FloatArray:
    """Breger S/N of every component against a residual spectrum.

    For an unconstrained fit this is :func:`cuperiod.prewhiten.engine._component_snr`
    operation for operation (``|A| == A`` there).
    """
    if fit.detail is None:
        return _engine._component_snr(scan, fit, settings)
    amplitude = significance_amplitude(fit)
    out = np.empty(fit.n_components, dtype=np.float64)
    for k in range(fit.n_components):
        noise = scan.noise_at(
            float(fit.frequency[k]),
            window=settings.snr_window,
            estimator=settings.noise_estimator,
        )
        out[k] = (
            float(amplitude[k]) / noise
            if np.isfinite(noise) and noise > 0.0
            else float("nan")
        )
    return out


def renumbered(state: SessionState) -> SessionState:
    """Ids compacted to ``1..K`` in table order, ties rewritten (serial unchanged)."""
    mapping = {c.id: i + 1 for i, c in enumerate(state.components)}
    components = tuple(
        replace(
            c,
            id=mapping[c.id],
            tie=None
            if c.tie is None
            else Tie(
                parents=tuple(mapping[p] for p in c.tie.parents),
                coefficients=c.tie.coefficients,
            ),
        )
        for c in state.components
    )
    return replace(state, components=components, next_id=len(components) + 1)


@register("renumber")
def renumber(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: NoArgs
) -> Outcome:
    """Compact ids to ``1..K`` in table order; ties follow."""
    new = renumbered(state)
    if new.same_content(state):
        return state, None, StepReport(kind="noop", message="ids are already compact")
    new = replace(new, serial=kernel.next_serial())
    report = StepReport(
        kind="edited", message="renumbered components", changed=frozenset({"ids"})
    )
    return new, derived, report


@register("prune")
def prune(
    kernel: SessionKernel, state: SessionState, derived: Derived, args: NoArgs
) -> Outcome:
    """The automatic loop's significance re-check of the final solution.

    The joint polish redistributes power between close components, so a frequency
    that cleared the threshold when it was extracted can end up insignificant in the
    final solution. Drop those and re-fit until every surviving component passes the
    same test it was admitted by; each pass strictly shrinks the solution, so this
    ends. A component the polish shrank to rounding noise (a stale sidelobe of a
    noise-free mode) is dropped by the same pass. Protected components (see
    :func:`protected_ids`) always survive. Renumbers when it removed something.
    """
    cfg = state.settings
    if not state.active:
        return state, None, StepReport(kind="noop", message="nothing to prune")
    protected = protected_ids(
        state, protect_manual=kernel.session_settings.protect_manual
    )
    roundoff = kernel.roundoff(state)
    current, current_derived = state, derived
    n_pruned = 0
    untied: set[int] = set()
    for _ in range(_engine._MAX_PRUNE_PASSES):
        fit, scan = current_derived.fit, current_derived.scan
        ids = [c.id for c in current.active]
        measured = component_snr(scan, fit, cfg)
        amplitude = np.abs(fit.amplitude)
        survivors = [
            i
            for i in range(fit.n_components)
            if ids[i] in protected
            or (
                measured[i] >= cfg.snr_threshold
                and float(amplitude[i]) > roundoff
            )
        ]
        if len(survivors) == fit.n_components:
            break
        n_pruned += fit.n_components - len(survivors)
        keep = {ids[i] for i in survivors}
        dropped = {cid for cid in ids if cid not in keep}
        # Without protect_manual a tie parent may go: its children are untied (free
        # at their current value), as remove(cascade=True) does; a tie to a missing
        # id would make every later fit of the state impossible.
        orphans = {
            c.id
            for c in current.components
            if c.id not in dropped
            and c.tie is not None
            and dropped.intersection(c.tie.parents)
        }
        untied.update(orphans)
        reduced = replace(
            current,
            components=tuple(
                replace(c, tie=None) if c.id in orphans else c
                for c in current.components
                if c.id not in dropped
            ),
        )
        mm = kernel.model_map(reduced)
        if not survivors:
            fit = kernel.fit(
                reduced, np.zeros(0), refine="none", covariance=True, mm=mm
            )
            # Not the data spectrum: v1.3 re-evaluated the residuals here.
            current, current_derived = kernel.commit_fit(reduced, fit, mm=mm)
            break
        frequencies = fit.frequency[survivors] if mm is None else mm.frequencies
        fit = kernel.fit(
            reduced, frequencies, refine=kernel.polish_refine(reduced, len(survivors)),
            covariance=True, mm=mm,
        )
        current, current_derived = kernel.commit_fit(reduced, fit, mm=mm)
    if not n_pruned:
        return state, None, StepReport(kind="noop", message="every component passes")
    suffix = (
        f"{n_pruned} insignificant component{'' if n_pruned == 1 else 's'} pruned"
    )
    stop_reason = current.stop_reason
    if current.status == "stopped":
        stop_reason = f"{stop_reason}; {suffix}"
    new = replace(
        renumbered(current),
        serial=kernel.next_serial(),
        n_pruned=current.n_pruned + n_pruned,
        stop_reason=stop_reason,
    )
    kernel.remember(new, current_derived)
    message = suffix
    if untied:
        count = len(untied)
        message += (
            f"; {count} tie{'' if count == 1 else 's'} to a pruned parent dissolved"
        )
    report = StepReport(
        kind="edited",
        message=message,
        statistics={"n_pruned": float(n_pruned)},
        changed=frozenset({"components", "fit", "ids"}),
    )
    return new, current_derived, report


__all__ = [
    "attempt",
    "component_snr",
    "detectable",
    "polish",
    "protected_ids",
    "prune",
    "renumber",
    "renumbered",
    "significance_amplitude",
    "step",
]
