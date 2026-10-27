"""From a session state to a :class:`~cuperiod.PreWhitenResult` and to table rows.

:func:`build_result` is the tail of the automatic :func:`cuperiod.prewhiten` loop —
uncertainties, final signal-to-noise, combination identification, the spectrum's own
amplitude reading and the blend flags — applied to whatever state the session is in.
Before any of that runs, amplitudes are made canonical: a component whose fit
returned a negative amplitude (only possible under a fixed phase) is reported as
``|A|`` with its phase plus pi, so every magnitude consumer (S/N, blend checks,
combinations, spacing) reads a non-negative amplitude. A component that varies per
subset is judged by its RMS amplitude rather than its coherent (phasor-mean)
headline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten import engine as _engine
from cuperiod.prewhiten.combinations import Combination, identify_combinations
from cuperiod.prewhiten.fit import MultiSineFit
from cuperiod.prewhiten.result import (
    PreWhitenResult,
    Sinusoid,
    SubsetSolution,
    amplitude_ratio,
)
from cuperiod.prewhiten.session.kernel import SessionKernel
from cuperiod.prewhiten.session.state import (
    INTERACTIVE_STOP_REASON,
    Component,
    Derived,
    SessionState,
)
from cuperiod.prewhiten.session.stepper import significance_amplitude
from cuperiod.prewhiten.uncertainty import Uncertainties

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class RowStats:
    """What the frequency table shows beside a component's own values.

    NaN (or ``None``) for an inactive component, which is not in the fit.
    """

    id: int
    snr: float
    amplitude_rms: float | None
    spectrum_amplitude: float
    amplitude_ratio: float
    blended: bool
    passes: bool
    combination: str | None
    frequency_error: float
    amplitude_error: float
    phase_error: float


def _canonical(
    amplitude: FloatArray, phase: FloatArray
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """``(|A|, phase (+ pi where A < 0) in [0, 2 pi), negative mask)``."""
    negative = np.asarray(amplitude < 0.0)
    if not negative.any():
        return amplitude, phase, negative
    flipped = np.mod(np.where(negative, phase + np.pi, phase), _TWO_PI)
    return np.abs(amplitude), flipped, negative


def _fixed_text(c: Component) -> str:
    return (
        ("f" if c.fix_frequency else "")
        + ("A" if c.fix_amplitude else "")
        + ("p" if c.fix_phase else "")
    )


def _nan_errors(fit: MultiSineFit) -> Uncertainties:
    nan = np.full(fit.n_components, np.nan)
    return Uncertainties(
        frequency=nan,
        amplitude=nan.copy(),
        phase=nan.copy(),
        method="none",
        correlation_factor=1.0,
    )


def _errors(
    kernel: SessionKernel,
    state: SessionState,
    fit: MultiSineFit,
    *,
    method: str,
    light: bool,
    should_stop: Callable[[], bool] | None,
) -> Uncertainties:
    """The component uncertainties exactly as the automatic loop asks for them."""
    cfg = state.settings
    used = kernel.used(state)
    if light:
        if not fit.covariance.size:
            return _nan_errors(fit)
        return _engine.component_uncertainties(
            used.time, used.error, fit,
            method="covariance",
            correlation_correction=cfg.correlation_correction,
        )
    extra = {} if should_stop is None else {"should_stop": should_stop}
    # The bootstrap's replicate fits mirror the final polish: one cyclic sweep from
    # the best fit stops short of each replicate's joint optimum for correlated
    # components and under-reports their frequency scatter.
    return _engine.component_uncertainties(
        used.time, used.error, fit,
        method=method,  # type: ignore[arg-type]
        correlation_correction=cfg.correlation_correction,
        n_resamples=cfg.n_resamples,
        seed=cfg.seed,
        refine=kernel.polish_refine(state, fit.n_components),
        sweeps=max(cfg.sweeps, 3),
        frequency_bounds=kernel.bounds(state, fit.frequency),
        max_nfev=cfg.max_nfev,
        **extra,
    )


def _subset_solution(
    kernel: SessionKernel,
    state: SessionState,
    fit: MultiSineFit,
    errors: Uncertainties,
) -> SubsetSolution | None:
    """Per-subset zero points and amplitudes/phases of a state that has them."""
    active = state.active
    varying = state.variation_by is not None and any(
        c.variation != "none" for c in active
    )
    if state.offsets_by is None and not varying:
        return None
    used = kernel.used(state)
    detail = fit.detail
    k = len(active)

    offset_labels: tuple[str, ...] = ()
    offset = offset_error = np.zeros(0)
    if state.offsets_by is not None:
        partition = state.partition(state.offsets_by)
        offset_labels = partition.labels
        offset = np.full(partition.n_subsets, np.nan)
        offset_error = np.full(partition.n_subsets, np.nan)
        if state.subset_offsets is not None:
            offset[:] = state.subset_offsets
        subsets = used.offset_subsets
        fallback = None if detail is None else detail.subset_offset_error
        known = errors.subset_offset if errors.subset_offset is not None else fallback
        if subsets is not None and known is not None:
            offset_error[subsets] = known

    variation_labels: tuple[str, ...] = ()
    shape = (k, 0)
    if varying and state.variation_by is not None:
        partition = state.partition(state.variation_by)
        variation_labels = partition.labels
        shape = (k, partition.n_subsets)
    amplitude = np.full(shape, np.nan)
    amplitude_error = np.full(shape, np.nan)
    phase = np.full(shape, np.nan)
    phase_error = np.full(shape, np.nan)
    if varying:
        subsets = used.variation_subsets
        for i, c in enumerate(active):
            if c.subset_amplitude is not None:
                amplitude[i] = c.subset_amplitude
            if c.subset_phase is not None:
                phase[i] = c.subset_phase
            if subsets is None:
                continue
            amp_err = (
                errors.subset_amplitude
                if errors.subset_amplitude is not None
                else (detail.subset_amplitude_error if detail is not None else None)
            )
            ph_err = (
                errors.subset_phase
                if errors.subset_phase is not None
                else (detail.subset_phase_error if detail is not None else None)
            )
            if amp_err is not None and c.variation in ("amplitude", "both"):
                amplitude_error[i, subsets] = amp_err[i]
            if ph_err is not None and c.variation in ("phase", "both"):
                phase_error[i, subsets] = ph_err[i]
    return SubsetSolution(
        offsets_by=state.offsets_by,
        offset_labels=offset_labels,
        offset=offset,
        offset_error=offset_error,
        variation_by=state.variation_by if varying else None,
        variation_labels=variation_labels,
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        phase=phase,
        phase_error=phase_error,
    )


def build_result(
    kernel: SessionKernel,
    state: SessionState,
    derived: Derived,
    *,
    uncertainty: str | None = None,
    should_stop: Callable[[], bool] | None = None,
    light: bool = False,
) -> PreWhitenResult:
    """The :class:`~cuperiod.PreWhitenResult` of ``state`` as it is (pure).

    Parameters
    ----------
    kernel, state, derived
        The session's kernel, the state to report and its Derived.
    uncertainty : str, optional
        Estimator overriding ``settings.uncertainty``.
    should_stop : callable, optional
        Polled between bootstrap replicates.
    light : bool, default False
        Skip the covariance re-fit and any bootstrap: errors come from an existing
        covariance (else NaN, ``uncertainty_method="none"``). For live views.
    """
    cfg = state.settings
    if not light:
        derived = kernel.with_covariance(state, derived)
    fit = derived.fit
    used = kernel.used(state)
    active = state.active
    k = fit.n_components
    method = uncertainty or cfg.uncertainty
    errors = _errors(
        kernel, state, fit, method=method, light=light, should_stop=should_stop
    )
    detail = fit.detail
    # Canonical amplitudes first: every consumer below reads |A|.
    amplitude, phase, negative = _canonical(fit.amplitude, fit.phase)
    significance = (
        amplitude if detail is None else np.asarray(significance_amplitude(fit))
    )
    scan = derived.scan
    snr_final = np.empty(k, dtype=np.float64)
    for i in range(k):
        noise = scan.noise_at(
            float(fit.frequency[i]),
            window=cfg.snr_window,
            estimator=cfg.noise_estimator,
        )
        snr_final[i] = (
            float(significance[i]) / noise
            if np.isfinite(noise) and noise > 0.0
            else float("nan")
        )
    labels = tuple(c.label for c in active)
    tied = {i for i, c in enumerate(active) if c.tie is not None}
    matches: tuple[Combination, ...] = ()
    baseline = kernel.baseline(state)
    if cfg.combinations and k > 1:
        matches = identify_combinations(
            fit.frequency,
            amplitude,
            frequency_error=errors.frequency,
            labels=labels,
            rayleigh=1.0 / baseline,
            max_order=cfg.combination_max_order,
            max_parents=cfg.combination_parents,
            tolerance_rayleigh=cfg.combination_tolerance_rayleigh,
            n_sigma=cfg.combination_sigma,
        )
        matches = tuple(m for m in matches if m.index not in tied)
    by_index: dict[int, str] = {c.index: c.label for c in matches}
    for i in tied:
        tie = active[i].tie
        assert tie is not None
        by_index[i] = f"{labels[i]} = {tie.text()}"

    # The spectrum's own reading at each reported frequency: a single-frequency
    # measurement of the data, independent of the joint fit. Where the two disagree
    # the component's amplitude is entangled with a correlated neighbour.
    initial_scan = kernel.data_scan(state)
    direct = initial_scan.amplitude_at(fit.frequency)
    ratios = [
        amplitude_ratio(float(significance[i]), float(direct[i])) for i in range(k)
    ]
    inactive_parent: frozenset[int] = frozenset()
    if any(c.tie is not None for c in active):
        mm = kernel.model_map(state)
        if mm is not None:
            inactive_parent = mm.inactive_parent
    varying = [c.variation != "none" for c in active]
    rms = None if detail is None else np.asarray(detail.rms_amplitude)
    components = []
    for i, c in enumerate(active):
        flags = set() if detail is None else set(detail.flags[i])
        if negative[i]:
            flags.add("negative_amplitude")
        if c.id in inactive_parent:
            flags.add("parent_inactive")
        if c.tie is not None:
            # A tie whose parents are all inactive reaches the fit as a fixed value;
            # the user tied it, and the table says so. Its frequency is then a
            # constant (error 0), as under fixed parents, so "fixed_f" stays.
            flags.add("tied")
        components.append(
            Sinusoid(
                rank=i + 1,
                label=labels[i],
                frequency=float(fit.frequency[i]),
                frequency_error=float(errors.frequency[i]),
                amplitude=float(amplitude[i]),
                amplitude_error=float(errors.amplitude[i]),
                phase=float(phase[i]),
                phase_error=float(errors.phase[i]),
                snr=float(snr_final[i]),
                fap=c.fap,
                delta_bic=c.delta_bic,
                combination=by_index.get(i),
                spectrum_amplitude=float(direct[i]),
                blended=bool(
                    ratios[i] > cfg.blend_tolerance
                    or ratios[i] < 1.0 / cfg.blend_tolerance
                ),
                fixed=_fixed_text(c),
                tied=None if c.tie is None else c.tie.text(),
                variation=c.variation,
                amplitude_rms=(
                    float(rms[i]) if varying[i] and rms is not None else None
                ),
                flags=tuple(sorted(flags)),
            )
        )
    # Only the spectra the result keeps are materialised with their full-grid phase
    # and power; every in-loop evaluation was amplitude-only.
    spectrum = residual_spectrum = window = None
    if cfg.store_spectra:
        spectrum = kernel.data_spectrum(state)
        residual_spectrum = (
            spectrum if scan is initial_scan else derived.residual_spectrum
        )
        window = kernel.window(state)
    stop_reason = (
        state.stop_reason
        if state.status == "stopped" and state.stop_reason
        else INTERACTIVE_STOP_REASON
    )
    return PreWhitenResult(
        components=tuple(components),
        combinations=matches,
        offset=fit.offset + kernel.zero_point(state),
        offset_error=errors.offset,
        t_ref=fit.t_ref,
        time=used.time,
        residuals=fit.residuals,
        n_samples=used.n,
        baseline=baseline,
        stop_reason=stop_reason,
        n_iterations=state.n_iterations,
        n_pruned=state.n_pruned,
        rms=fit.rms,
        chi2=fit.chi2,
        reduced_chi2=fit.reduced_chi2,
        bic=fit.bic,
        # The D the estimator actually applied: it returns 1.0 whenever nothing was
        # inflated, including for the bootstrap, which is never corrected.
        correlation_factor=errors.correlation_factor,
        uncertainty_method=errors.method if light else method,
        backend=kernel.backend,
        spectrum=spectrum,
        residual_spectrum=residual_spectrum,
        window=window,
        meta=dict(kernel.data.meta),
        subsets=_subset_solution(kernel, state, fit, errors),
        n_deleted=kernel.data.n - used.n,
    )


def rows(
    kernel: SessionKernel, state: SessionState, derived: Derived
) -> tuple[RowStats, ...]:
    """Per-component statistics for a live table, in table order (light: no re-fit)."""
    cfg = state.settings
    result = build_result(kernel, state, derived, light=True)
    by_id = {c.id: s for c, s in zip(state.active, result.components, strict=True)}
    nan = float("nan")
    out = []
    for c in state.components:
        s = by_id.get(c.id)
        if s is None:
            out.append(
                RowStats(
                    id=c.id, snr=nan, amplitude_rms=None, spectrum_amplitude=nan,
                    amplitude_ratio=nan, blended=False, passes=False,
                    combination=None, frequency_error=nan, amplitude_error=nan,
                    phase_error=nan,
                )
            )
            continue
        judged = s.amplitude_rms if s.amplitude_rms is not None else s.amplitude
        passes = "snr" not in cfg.stop_criteria or s.snr >= cfg.snr_threshold
        if "amplitude" in cfg.stop_criteria and cfg.min_amplitude is not None:
            passes = passes and judged >= cfg.min_amplitude
        out.append(
            RowStats(
                id=c.id,
                snr=s.snr,
                amplitude_rms=s.amplitude_rms,
                spectrum_amplitude=s.spectrum_amplitude,
                amplitude_ratio=amplitude_ratio(judged, s.spectrum_amplitude),
                blended=s.blended,
                passes=bool(passes),
                combination=s.combination,
                frequency_error=s.frequency_error,
                amplitude_error=s.amplitude_error,
                phase_error=s.phase_error,
            )
        )
    return tuple(out)


__all__ = ["RowStats", "build_result", "rows"]
