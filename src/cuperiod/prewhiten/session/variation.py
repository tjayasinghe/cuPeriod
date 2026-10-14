"""Amplitude and phase per subset, with a constancy test (a query on a session).

:func:`compute` answers "does this mode's amplitude (or phase) change from night to
night, season to season?" without touching the session: it fits one copy of the
head's constraints in which the selected components take one amplitude and/or phase
per subset of a partition, every frequency held (``refine="none"``; with
``mode="both"`` the fit is purely linear), and tests each component's per-subset
values for constancy with a Wald statistic

    W = (L x)^T (L C L^T)^{-1} (L x) ~ chi2(S - 1),

``L`` the successive-difference contrast, ``x`` the per-subset values and ``C`` their
block of the fit's parameter covariance (inflated by the residual correlation factor
when ``settings.correlation_correction``; phase differences wrapped to
``(-pi, pi]``). :meth:`~cuperiod.PreWhitenSession.set_variation` is the persistent
version of the same model.

The frequencies are held as *fixed* in the query fit, not merely left unrefined: a
covariance that kept them free would carry their uncertainty into every subset's
phase as a ramp growing with the distance from ``t_ref`` (a frequency shift and a
linear phase trend are the same model), and the phase test would lose one degree of
freedom (measured: mean ``W / (S - 1)`` of 0.69 on constant modes instead of 1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.prewhiten.constraints import Variation
from cuperiod.prewhiten.fit import Refinement
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.session.modelmap import apply
from cuperiod.prewhiten.session.ops_components import unique
from cuperiod.prewhiten.session.state import (
    Component,
    Derived,
    SessionError,
    SessionState,
    VariationResult,
)
from cuperiod.prewhiten.uncertainty import correlation_factor

_TWO_PI = 2.0 * np.pi


def contrast(size: int) -> FloatArray:
    """The ``(size - 1, size)`` successive-difference contrast ``x[s+1] - x[s]``."""
    out = np.zeros((max(size - 1, 0), size), dtype=np.float64)
    rows = np.arange(max(size - 1, 0))
    out[rows, rows] = -1.0
    out[rows, rows + 1] = 1.0
    return out


def wald(
    values: FloatArray, covariance: FloatArray, *, circular: bool = False
) -> tuple[float, float]:
    """Wald statistic and p-value of "every value equal" (``chi2`` with ``S - 1``
    degrees of freedom).

    Parameters
    ----------
    values : numpy.ndarray
        ``(S,)`` per-subset estimates.
    covariance : numpy.ndarray
        ``(S, S)`` their covariance.
    circular : bool, default False
        Phases: successive differences are wrapped to ``(-pi, pi]``.

    Returns
    -------
    tuple of float
        ``(W, p)``; NaN when fewer than two values or a non-finite or singular
        contrast covariance leaves the test undefined.
    """
    from scipy.stats import chi2

    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(covariance, dtype=np.float64)
    size = int(x.size)
    nan = float("nan")
    if size < 2 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(c))):
        return nan, nan
    contrast_matrix = contrast(size)
    difference = contrast_matrix @ x
    if circular:
        difference = -np.mod(-difference + np.pi, _TWO_PI) + np.pi
    middle = contrast_matrix @ c @ contrast_matrix.T
    try:
        solved = np.linalg.solve(middle, difference)
    except np.linalg.LinAlgError:
        return nan, nan
    statistic = float(difference @ solved)
    if not np.isfinite(statistic) or statistic < 0.0:
        return nan, nan
    return statistic, float(chi2.sf(statistic, size - 1))


def _query_state(
    state: SessionState,
    selected: Sequence[int],
    *,
    mode: Variation,
    by: str,
) -> SessionState:
    """The head with the selected components varying per ``by`` (a throwaway copy).

    A quantity the mode makes per-subset cannot stay fixed, so it is freed in the
    copy; switching to another partition returns every other varying component to
    common values (its per-subset values belong to the old partition).
    """
    varies_a = mode in ("amplitude", "both")
    varies_p = mode in ("phase", "both")
    switching = by != state.variation_by
    chosen = set(selected)
    components: list[Component] = []
    for c in state.components:
        if c.id in chosen:
            keep_a = varies_a and not switching and c.variation in ("amplitude", "both")
            keep_p = varies_p and not switching and c.variation in ("phase", "both")
            components.append(
                replace(
                    c,
                    variation=mode,
                    fix_amplitude=c.fix_amplitude and not varies_a,
                    fix_phase=c.fix_phase and not varies_p,
                    subset_amplitude=c.subset_amplitude if keep_a else None,
                    subset_phase=c.subset_phase if keep_p else None,
                )
            )
        elif switching and c.variation != "none":
            components.append(
                replace(c, variation="none", subset_amplitude=None, subset_phase=None)
            )
        else:
            components.append(c)
    return replace(state, variation_by=by, components=tuple(components))


def _block(names: Sequence[str], prefix: str, k: int, count: int) -> IntArray:
    """Covariance indices of ``prefix[k|0..count-1]`` (``A`` or ``phi``)."""
    position = {name: i for i, name in enumerate(names)}
    return np.asarray(
        [position[f"{prefix}[{k}|{g}]"] for g in range(count)], dtype=np.int64
    )


def compute(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    ids: Sequence[int] | None,
    *,
    mode: Variation,
    by: str | None,
    improve_frequencies: bool,
    should_stop: Callable[[], bool] | None,
) -> VariationResult:
    """Per-subset amplitudes/phases of selected components and their constancy tests.

    Parameters
    ----------
    kernel, state, derived
        The session's kernel, the head state and its Derived (nothing changes).
    ids : sequence of int, optional
        Component ids (active ones); ``None`` = every active component.
    mode : {"amplitude", "phase", "both"}
        What varies per subset in the query fit.
    by : str, optional
        The partition; defaults to the state's variation partition.
    improve_frequencies : bool
        Re-fit the frequencies (the polish policy) instead of holding them.
    should_stop : callable, optional
        Polled before the fit; a true answer abandons the query
        (:class:`SessionError`).

    Returns
    -------
    VariationResult
        Values and errors per (component, subset) over the full partition (NaN for a
        subset with no point in use), the head's common values, Wald statistics and
        p-values (NaN for a quantity the mode keeps common) and the BIC of the varying
        model minus the head's.

    Raises
    ------
    SessionError
        For no partition, ``mode="none"``, an unknown or inactive id, or a stop.
    ConstraintError
        When the partition cannot carry the model (a subset below
        ``min_subset_points``: merge small subsets first).
    """
    name = by if by is not None else state.variation_by
    if name is None:
        raise SessionError(
            "no variation partition: pass by= or choose one with use_variation()"
        )
    partition = state.partition(name)
    if mode not in ("amplitude", "phase", "both"):
        raise SessionError(f"mode must be 'amplitude', 'phase' or 'both', not {mode!r}")
    active_ids = [c.id for c in state.active]
    if not active_ids:
        raise SessionError("no active component to test")
    selected = active_ids if ids is None else unique(ids)
    if not selected:
        raise SessionError("no component selected")
    for cid in selected:
        if not state.component(cid).active:
            raise SessionError(f"F{cid} is inactive")

    query = _query_state(state, selected, mode=mode, by=name)
    if should_stop is not None and should_stop():
        raise SessionError("variation query stopped")
    free = kernel.model_map(query)
    if free is None:  # pragma: no cover - a varying component always maps
        raise SessionError("the query has nothing to vary")
    if improve_frequencies:
        refine: Refinement = kernel.polish_refine(query, len(free.ids))
        moved = kernel.fit(
            query, free.frequencies, refine=refine, covariance=False, mm=free
        )
        query = replace(query, components=apply(query, moved, free).components)
        free = kernel.model_map(query)
        assert free is not None
    n_free = (
        0
        if free.constraints is None
        else free.constraints.resolve(free.frequencies).n_free
    )
    # Held, not merely unrefined: see the module docstring.
    mm = kernel.model_map(query, hold=frozenset(c.id for c in query.active))
    assert mm is not None  # a hold always maps
    fit = kernel.fit(query, mm.frequencies, refine="none", covariance=True, mm=mm)
    detail = fit.detail
    used = kernel.used(query)
    if detail is None or used.variation_subsets is None:  # pragma: no cover
        raise SessionError("the query fit returned no per-subset values")

    cfg = state.settings
    factor = 1.0
    if cfg.correlation_correction:
        order = np.argsort(used.time, kind="stable")
        factor = float(correlation_factor(np.asarray(fit.residuals)[order]))
    scale = float(np.sqrt(factor))
    covariance = np.asarray(detail.parameter_covariance, dtype=np.float64) * factor
    names = detail.parameter_names
    subsets = used.variation_subsets
    count = int(subsets.size)
    total = partition.n_subsets

    k_sel = len(selected)
    amplitude = np.full((k_sel, total), np.nan)
    amplitude_error = np.full((k_sel, total), np.nan)
    phase = np.full((k_sel, total), np.nan)
    phase_error = np.full((k_sel, total), np.nan)
    a_stat = np.full(k_sel, np.nan)
    a_p = np.full(k_sel, np.nan)
    p_stat = np.full(k_sel, np.nan)
    p_p = np.full(k_sel, np.nan)
    varies_a = mode in ("amplitude", "both")
    varies_p = mode in ("phase", "both")
    for row, cid in enumerate(selected):
        k = mm.position(cid)
        if detail.subset_amplitude is not None:
            amplitude[row, subsets] = detail.subset_amplitude[k]
        if detail.subset_phase is not None:
            phase[row, subsets] = detail.subset_phase[k]
        if varies_a and detail.subset_amplitude_error is not None:
            amplitude_error[row, subsets] = detail.subset_amplitude_error[k] * scale
        if varies_p and detail.subset_phase_error is not None:
            phase_error[row, subsets] = detail.subset_phase_error[k] * scale
        if covariance.size:
            if varies_a and detail.subset_amplitude is not None:
                index = _block(names, "A", k, count)
                a_stat[row], a_p[row] = wald(
                    detail.subset_amplitude[k], covariance[np.ix_(index, index)]
                )
            if varies_p and detail.subset_phase is not None:
                index = _block(names, "phi", k, count)
                p_stat[row], p_p[row] = wald(
                    detail.subset_phase[k],
                    covariance[np.ix_(index, index)],
                    circular=True,
                )

    spans = partition.spans(kernel.data.time, used=used.used)
    head = [state.component(cid) for cid in selected]
    return VariationResult(
        component_ids=tuple(selected),
        mode=mode,
        by=name,
        subset_labels=partition.labels,
        mid_time=np.mean(spans, axis=1),
        n_points=partition.counts(used.used),
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        phase=phase,
        phase_error=phase_error,
        common_amplitude=np.asarray([c.amplitude for c in head], dtype=np.float64),
        common_phase=np.asarray([c.phase for c in head], dtype=np.float64),
        amplitude_statistic=a_stat,
        amplitude_p_value=a_p,
        phase_statistic=p_stat,
        phase_p_value=p_p,
        # Counted as the persistent model would be: its frequencies are free
        # parameters (held here only to condition the covariance).
        delta_bic=float(
            replace(fit, n_parameters=fit.n_parameters + n_free).bic
            - derived.fit.bic
        ),
        frequencies_improved=bool(improve_frequencies),
    )


__all__ = ["compute", "contrast", "wald"]
