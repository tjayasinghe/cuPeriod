"""The iterative extraction loop: :func:`prewhiten`.

This is the automated replacement for the interactive Period04 workflow. Starting from
the raw light curve it repeats

1. compute the amplitude spectrum of the current residuals,
2. take the tallest peak that is resolved from everything already extracted,
3. re-fit **all** components simultaneously (frequencies, amplitudes, phases, offset),
4. decide whether the new component survives the stopping criteria,

until a component fails, no resolved peak remains, or the frequency cap is reached. The
decision at step 4 is the part interactive workflows leave to the operator's judgement,
so it is made explicit here: every run records *why* it stopped, and the criteria are
settings rather than habits.

Cost is dominated by step 1, and :class:`~cuperiod.prewhiten.spectrum.SpectrumEngine`
makes it one NUFFT per iteration by caching everything that depends only on sampling.
Step 3 uses variable projection, so the optimiser only ever sees the ``K`` frequencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.config import PreWhitenSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.prewhiten.combinations import Combination, identify_combinations
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine
from cuperiod.prewhiten.result import PreWhitenResult, Sinusoid
from cuperiod.prewhiten.spectrum import AmplitudeSpectrum, SpectrumEngine
from cuperiod.prewhiten.uncertainty import component_uncertainties, correlation_factor

#: Upper bound on the prune/re-fit passes of the final solution (each strictly shrinks
#: the component count, so this is a safety net rather than a real limit).
_MAX_PRUNE_PASSES = 20

#: How many candidate peaks may be discarded (because their refinement merged into an
#: existing component) before the run gives up looking for more.
_MAX_REJECTED_CANDIDATES = 20


def default_prewhiten_grid(
    lc: LightCurve, settings: PreWhitenSettings | None = None
) -> GridSpec:
    """The default uniform frequency grid for pre-whitening ``lc``.

    Runs from ``1/T`` to the pseudo-Nyquist frequency of the sampling, oversampled by
    ``samples_per_peak`` (10 by default — a finer grid than a plain period search needs,
    because each peak's position seeds a non-linear fit).

    Parameters
    ----------
    lc : LightCurve
        The light curve (non-finite points are ignored).
    settings : PreWhitenSettings, optional
        Grid-controlling settings; defaults are used when omitted.

    Returns
    -------
    GridSpec
        A uniform frequency grid.
    """
    cfg = settings or PreWhitenSettings()
    finite = lc.finite()
    if finite.baseline <= 0.0:
        raise InsufficientDataError("pre-whitening: no usable time baseline")
    maximum = cfg.maximum_frequency
    if maximum is None:
        maximum = pseudo_nyquist_frequency(finite.time, cfg.nyquist_factor)
    minimum = cfg.minimum_frequency
    if minimum is None:
        minimum = 1.0 / finite.baseline
    if minimum >= maximum:
        raise InsufficientDataError(
            f"pre-whitening: empty search band ({minimum} .. {maximum} cycles/day)"
        )
    return uniform_frequency_grid(
        finite.baseline,
        maximum_frequency=maximum,
        minimum_frequency=minimum,
        samples_per_peak=cfg.samples_per_peak,
    )


def _peak_fap(
    time: FloatArray,
    residuals: FloatArray,
    error: FloatArray | None,
    power: float,
    *,
    f_min: float,
    f_max: float,
    fit_mean: bool,
) -> float:
    """Baluev false-alarm probability of a peak of normalized power ``power``.

    The amplitude spectrum's ``power`` is the standard-normalized generalized
    Lomb-Scargle power by construction, so astropy's analytic false-alarm machinery
    applies directly. Any failure (a degenerate spectrum, a missing optional dependency)
    yields NaN rather than aborting the extraction.
    """
    if not np.isfinite(power) or power <= 0.0:
        return float("nan")
    try:
        from astropy.timeseries import LombScargle

        ls = LombScargle(time, residuals, error, fit_mean=fit_mean)
        value = ls.false_alarm_probability(
            min(power, 1.0 - 1e-15),
            method="baluev",
            minimum_frequency=f_min,
            maximum_frequency=f_max,
        )
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except Exception:  # noqa: BLE001 - FAP is informational, never fatal
        return float("nan")


def _accept(
    *,
    snr: float,
    fap: float,
    delta_bic: float,
    amplitude: float,
    settings: PreWhitenSettings,
) -> tuple[bool, str]:
    """Whether a freshly extracted component survives; the reason when it does not.

    Every configured criterion must pass. A criterion whose statistic is undefined
    (NaN) counts as a failure — silently keeping a component whose significance could
    not be measured is exactly the mistake automated pre-whitening exists to avoid.
    """
    for criterion in settings.stop_criteria:
        if criterion == "snr":
            if not (snr >= settings.snr_threshold):
                measured = f"{snr:.2f}" if np.isfinite(snr) else "undefined"
                return False, f"S/N {measured} < {settings.snr_threshold:g}"
        elif criterion == "fap":
            if not (fap <= settings.fap_threshold):
                measured = f"{fap:.3g}" if np.isfinite(fap) else "undefined"
                return False, f"FAP {measured} > {settings.fap_threshold:g}"
        elif criterion == "bic":
            if not (delta_bic <= -settings.min_delta_bic):
                measured = (
                    f"{delta_bic:+.1f}" if np.isfinite(delta_bic) else "undefined"
                )
                return False, f"dBIC {measured} > -{settings.min_delta_bic:g}"
        elif criterion == "amplitude":
            floor = settings.min_amplitude
            if floor is not None and not (amplitude >= floor):
                return False, f"amplitude {amplitude:.4g} < {floor:g}"
    return True, ""


def _component_snr(
    spectrum: AmplitudeSpectrum, fit: MultiSineFit, settings: PreWhitenSettings
) -> FloatArray:
    """Breger signal-to-noise of every component against a residual spectrum."""
    out = np.empty(fit.n_components, dtype=np.float64)
    for k in range(fit.n_components):
        noise = spectrum.noise_at(
            float(fit.frequency[k]),
            window=settings.snr_window,
            estimator=settings.noise_estimator,
        )
        out[k] = (
            float(fit.amplitude[k]) / noise
            if np.isfinite(noise) and noise > 0.0
            else float("nan")
        )
    return out


def _as_light_curve(
    data: Any, columns: ColumnMap | None, domain: Domain | None
) -> LightCurve:
    """Coerce a user input to a single-band :class:`LightCurve`."""
    from cuperiod.api import to_input

    lc = to_input(data, columns=columns, domain=domain)
    if isinstance(lc, MultiBandLightCurve):
        raise ValueError(
            "pre-whitening works on a single band; pass one band "
            "(e.g. mblc.bands['V']) rather than a MultiBandLightCurve"
        )
    return lc


def prewhiten(
    data: Any,
    *,
    settings: PreWhitenSettings | None = None,
    grid: GridSpec | None = None,
    backend: str | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> PreWhitenResult:
    """Extract a pulsator's frequency solution automatically.

    Parameters
    ----------
    data : various
        The light curve, in any form :func:`cuperiod.to_input` accepts (a
        :class:`~cuperiod.LightCurve`, a ``(t, y[, dy])`` tuple, a table, ...). Must be
        a single band.
    settings : PreWhitenSettings, optional
        Grid, stopping, fitting and uncertainty controls. Defaults are the conservative
        published choices — see :class:`~cuperiod.PreWhitenSettings`.
    grid : GridSpec, optional
        A custom uniform frequency grid; defaults to
        :func:`default_prewhiten_grid`.
    backend : str, optional
        Overrides ``settings.backend`` for the amplitude spectrum.
    columns, domain : optional
        Column mapping and brightness-domain handling for table inputs.

    Returns
    -------
    PreWhitenResult
        The ranked component list with uncertainties, the residuals and their spectrum,
        the fit statistics, and the reason the extraction stopped.

    Raises
    ------
    InsufficientDataError
        If the light curve has too few finite points or no time baseline.

    Examples
    --------
    >>> import cuperiod as cup                                     # doctest: +SKIP
    >>> solution = cup.prewhiten((time, mag, mag_err))             # doctest: +SKIP
    >>> print(solution.summary())                                  # doctest: +SKIP
    >>> solution.frequency, solution.frequency_error               # doctest: +SKIP
    """
    cfg = settings or PreWhitenSettings()
    lc = _as_light_curve(data, columns, domain).finite()
    n = lc.n
    if n < cfg.min_detections:
        raise InsufficientDataError(
            f"pre-whitening: {n} finite points < min_detections {cfg.min_detections}"
        )
    if lc.baseline <= 0.0:
        raise InsufficientDataError("pre-whitening: no usable time baseline")

    trial_grid = grid if grid is not None else default_prewhiten_grid(lc, cfg)
    engine = SpectrumEngine(
        lc.time,
        lc.error,
        grid=trial_grid,
        normalization=cfg.normalization,
        backend=backend or cfg.backend,
        device=cfg.device,
        precision=cfg.precision,
        eps=cfg.nufft_eps,
        freq_batch=cfg.direct_freq_batch,
    )
    time, value, error = lc.time, lc.value, lc.error
    f_min = float(engine.frequency[0])
    f_max = float(engine.frequency[-1])
    grid_low, grid_high = max(0.5 * f_min, 1e-12), f_max
    separation = cfg.min_separation_rayleigh / lc.baseline
    max_shift = cfg.refine_bound_rayleigh / lc.baseline

    def refine_bounds(frequencies: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Per-frequency refinement box: at most one bound width from where it started.

        Without this a refinement started on a modest peak can slide down the wing of a
        much stronger neighbour and land on top of it, producing near-identical
        frequencies with enormous, mutually cancelling amplitudes.
        """
        low = np.maximum(frequencies - max_shift, grid_low)
        high = np.minimum(frequencies + max_shift, grid_high)
        return low, np.maximum(high, np.nextafter(low, np.inf))

    fit = fit_multisine(
        time, value, error, np.zeros(0),
        fit_mean=True, t_ref=engine.t_ref, refine="none", covariance=False,
    )
    initial_spectrum = engine.spectrum(value)
    spectrum = initial_spectrum
    faps: list[float] = []
    deltas: list[float] = []
    blocked: list[float] = []
    stop_reason = f"reached max_frequencies ({cfg.max_frequencies})"
    iterations = 0
    max_attempts = cfg.max_frequencies + _MAX_REJECTED_CANDIDATES

    while fit.n_components < cfg.max_frequencies and iterations < max_attempts:
        iterations += 1
        avoid = np.concatenate(
            [fit.frequency, np.asarray(blocked, dtype=np.float64)]
        )
        index = spectrum.peak_index(exclude=avoid, separation=separation)
        if index is None:
            stop_reason = "no peak resolved from the existing components remains"
            break
        f_guess, _ = spectrum.refine_peak(index)
        candidates = np.append(fit.frequency, f_guess)
        try:
            trial = fit_multisine(
                time, value, error, candidates,
                fit_mean=cfg.fit_mean,
                t_ref=engine.t_ref,
                refine=cfg.refine,
                sweeps=cfg.sweeps,
                frequency_bounds=refine_bounds(candidates),
                max_nfev=cfg.max_nfev,
                covariance=False,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            stop_reason = f"fit failed: {exc}"
            break
        new_frequency = float(trial.frequency[-1])
        if fit.n_components and float(
            np.min(np.abs(trial.frequency[:-1] - new_frequency))
        ) < separation:
            # The refinement merged the candidate into an existing component: the peak
            # was that component's residual, not a new mode. Block it and look further
            # down the spectrum rather than abandoning the run.
            blocked.append(f_guess)
            continue
        # Deferred until the candidate survives the collision guard: the false-alarm
        # probability costs a pass over the data; a discarded candidate never uses it.
        fap = _peak_fap(
            time, fit.residuals, error, float(spectrum.power[index]),
            f_min=f_min, f_max=f_max, fit_mean=cfg.fit_mean,
        )
        trial_spectrum = engine.spectrum(trial.residuals)
        amplitude = float(trial.amplitude[-1])
        noise = trial_spectrum.noise_at(
            new_frequency, window=cfg.snr_window, estimator=cfg.noise_estimator
        )
        snr = amplitude / noise if np.isfinite(noise) and noise > 0.0 else float("nan")
        delta_bic = trial.bic - fit.bic
        accepted, reason = _accept(
            snr=snr, fap=fap, delta_bic=delta_bic, amplitude=amplitude, settings=cfg
        )
        if not accepted:
            stop_reason = reason
            break
        fit, spectrum = trial, trial_spectrum
        faps.append(fap)
        deltas.append(delta_bic)
    else:
        if cfg.max_frequencies == 0:
            stop_reason = "max_frequencies is 0"
        elif iterations >= max_attempts:
            stop_reason = "too many candidates merged with existing components"

    def polish(frequencies: FloatArray) -> MultiSineFit:
        """Re-fit at ``frequencies`` as accurately as the component count allows."""
        return fit_multisine(
            time, value, error, frequencies,
            fit_mean=fit.fit_mean,
            t_ref=engine.t_ref,
            refine=(
                "simultaneous"
                if frequencies.size <= cfg.max_simultaneous
                else "cyclic"
            ),
            sweeps=cfg.sweeps,
            frequency_bounds=refine_bounds(frequencies),
            max_nfev=cfg.max_nfev,
            covariance=True,
        )

    # The loop only refines each frequency as it is extracted; the polish optimises the
    # whole solution jointly (or, when there are too many components for that to be
    # affordable, sweeps them cyclically) so no frequency is left at a stale value.
    if cfg.final_refine and fit.n_components > 0:
        fit = polish(np.ascontiguousarray(fit.frequency))
        spectrum = engine.spectrum(fit.residuals)
    elif fit.covariance.size == 0:
        # Every fit inside the loop skips the covariance (it is only needed once); the
        # accepted solution still has to carry one for the reported uncertainties.
        fit = fit_multisine(
            time, value, error, fit.frequency,
            fit_mean=fit.fit_mean, t_ref=engine.t_ref, refine="none", covariance=True,
        )

    # The joint polish redistributes power between close components, so a frequency that
    # cleared the threshold when it was extracted can end up insignificant in the final
    # solution. Drop those and re-fit until every surviving component passes the same
    # test it was admitted by; each pass strictly shrinks the solution, so this ends.
    kept = list(range(fit.n_components))
    n_pruned = 0
    if cfg.prune and "snr" in cfg.stop_criteria and fit.n_components:
        for _ in range(_MAX_PRUNE_PASSES):
            measured = _component_snr(spectrum, fit, cfg)
            survivors = [
                i
                for i in range(fit.n_components)
                if measured[i] >= cfg.snr_threshold
            ]
            if len(survivors) == fit.n_components:
                break
            n_pruned += fit.n_components - len(survivors)
            kept = [kept[i] for i in survivors]
            if not survivors:
                fit = fit_multisine(
                    time, value, error, np.zeros(0),
                    fit_mean=True, t_ref=engine.t_ref, refine="none", covariance=True,
                )
                spectrum = engine.spectrum(fit.residuals)
                break
            fit = polish(fit.frequency[survivors])
            spectrum = engine.spectrum(fit.residuals)
        if n_pruned:
            stop_reason = (
                f"{stop_reason}; {n_pruned} insignificant component"
                f"{'' if n_pruned == 1 else 's'} pruned"
            )

    errors = component_uncertainties(
        time, error, fit,
        method=cfg.uncertainty,
        correlation_correction=cfg.correlation_correction,
        n_resamples=cfg.n_resamples,
        seed=cfg.seed,
        refine=cfg.refine,
        sweeps=cfg.sweeps,
    )
    snr_final = _component_snr(spectrum, fit, cfg)
    labels = tuple(f"F{i + 1}" for i in range(fit.n_components))
    matches: tuple[Combination, ...] = ()
    if cfg.combinations and fit.n_components > 1:
        matches = identify_combinations(
            fit.frequency,
            fit.amplitude,
            frequency_error=errors.frequency,
            labels=labels,
            rayleigh=1.0 / lc.baseline,
            max_order=cfg.combination_max_order,
            max_parents=cfg.combination_parents,
            tolerance_rayleigh=cfg.combination_tolerance_rayleigh,
            n_sigma=cfg.combination_sigma,
        )
    by_index = {c.index: c.label for c in matches}

    components = tuple(
        Sinusoid(
            rank=k + 1,
            label=labels[k],
            frequency=float(fit.frequency[k]),
            frequency_error=float(errors.frequency[k]),
            amplitude=float(fit.amplitude[k]),
            amplitude_error=float(errors.amplitude[k]),
            phase=float(fit.phase[k]),
            phase_error=float(errors.phase[k]),
            snr=float(snr_final[k]),
            fap=faps[kept[k]] if kept[k] < len(faps) else float("nan"),
            delta_bic=deltas[kept[k]] if kept[k] < len(deltas) else float("nan"),
            combination=by_index.get(k),
        )
        for k in range(fit.n_components)
    )
    keep = cfg.store_spectra
    return PreWhitenResult(
        components=components,
        combinations=matches,
        offset=fit.offset,
        offset_error=fit.offset_error,
        t_ref=fit.t_ref,
        time=time,
        residuals=fit.residuals,
        n_samples=n,
        baseline=lc.baseline,
        stop_reason=stop_reason,
        n_iterations=iterations,
        n_pruned=n_pruned,
        rms=fit.rms,
        chi2=fit.chi2,
        reduced_chi2=fit.reduced_chi2,
        bic=fit.bic,
        correlation_factor=(
            correlation_factor(fit.residuals) if cfg.correlation_correction else 1.0
        ),
        uncertainty_method=cfg.uncertainty,
        backend=engine.backend,
        spectrum=initial_spectrum if keep else None,
        residual_spectrum=spectrum if keep else None,
        meta=dict(lc.meta),
    )


__all__ = ["default_prewhiten_grid", "prewhiten"]
