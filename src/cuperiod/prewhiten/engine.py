"""The iterative extraction loop: :func:`prewhiten`.

This is the automated replacement for the interactive Period04 workflow. Starting from
the raw light curve it repeats

1. compute the amplitude spectrum of the current residuals,
2. take the tallest peak that is resolved from everything already extracted,
3. re-solve every amplitude, phase and the offset jointly and refine the new frequency
   non-linearly (``refine``, ``"last"`` by default; the final polish sweeps them all),
4. decide whether the new component survives the stopping criteria,

until a component fails, no resolved peak remains, or the frequency cap is reached. The
decision at step 4 is the part interactive workflows leave to the operator's judgement,
so it is made explicit here: every run records *why* it stopped, and the criteria are
settings rather than habits.

Cost is dominated by step 1, and :class:`~cuperiod.prewhiten.spectrum.SpectrumEngine`
makes it one NUFFT per iteration by caching everything that depends only on sampling.
Step 3 uses variable projection, so the optimiser only ever sees the ``K`` frequencies.

The loop itself is :mod:`cuperiod.prewhiten.session.stepper` — one ``step`` of a
:class:`~cuperiod.PreWhitenSession` per iteration, so the automatic run and the
interactive one cannot drift apart. This module keeps the grid defaults and the
helpers both share (the acceptance test, the false-alarm probability, the S/N).
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
    check_grid_size,
    pseudo_nyquist_frequency,
    regular_cadence_nyquist,
    uniform_frequency_grid,
    uniform_grid_size,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.prewhiten.fap import baluev_fap
from cuperiod.prewhiten.fit import MultiSineFit
from cuperiod.prewhiten.result import PreWhitenResult
from cuperiod.prewhiten.spectrum import (
    _SpectrumReadout,
)

# Re-exported on purpose: the session's report calls it through this module, so
# patching engine.component_uncertainties still reaches every pre-whitening run.
from cuperiod.prewhiten.uncertainty import (
    component_uncertainties as component_uncertainties,
)

#: Upper bound on the prune/re-fit passes of the final solution (each strictly shrinks
#: the component count, so this is a safety net rather than a real limit).
_MAX_PRUNE_PASSES = 20

#: How many candidate peaks may be discarded (because their refinement merged into an
#: existing component) before the run gives up looking for more.
_MAX_REJECTED_CANDIDATES = 20

#: Floor of the automatic search band (cycles/day). The median-gap pseudo-Nyquist of
#: nightly ground-based sampling collapses to a few cycles/day — far below the
#: delta Scuti / HADS frequencies (periods down to ~0.02 d) this tool exists to
#: extract, whose non-sinusoidal light curves also need the first few harmonics in
#: band. Irregular sampling has no true Nyquist limit, so searching above the
#: pseudo-Nyquist is legitimate; dense space cadences exceed the floor and ignore it.
DEFAULT_MAX_FREQUENCY_FLOOR = 50.0


def default_maximum_frequency(time: FloatArray, nyquist_factor: int = 5) -> float:
    """The automatic top of the pre-whitening band for this sampling (cycles/day).

    ``nyquist_factor`` times the median-gap pseudo-Nyquist frequency, but never below
    :data:`DEFAULT_MAX_FREQUENCY_FLOOR`: sparse ground-based sampling would otherwise
    put the whole classical-pulsator regime out of band and the extraction would fit
    the daily aliases of the real signal instead.

    A strictly regular cadence (a lattice — gaps and a smooth barycentric timing term
    allowed, see :func:`cuperiod.core.grid.regular_cadence_nyquist`) has a true Nyquist
    frequency ``0.5/cadence``, and the band stops there: above it the spectrum mirrors
    the baseband, every mode has equal-height images at ``k/cadence +/- f``, and the
    extraction reported whichever image the grid happened to sample best. Space
    photometry is the common case; set ``maximum_frequency`` explicitly for
    super-Nyquist work.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days).
    nyquist_factor : int, default 5
        Multiple of the pseudo-Nyquist frequency to allow.

    Returns
    -------
    float
        The default ``maximum_frequency`` in cycles/day.
    """
    automatic = max(
        pseudo_nyquist_frequency(time, nyquist_factor), DEFAULT_MAX_FREQUENCY_FLOOR
    )
    nyquist = regular_cadence_nyquist(np.asarray(time, dtype=np.float64))
    return automatic if nyquist is None else min(automatic, nyquist)


def default_prewhiten_grid(
    lc: LightCurve, settings: PreWhitenSettings | None = None
) -> GridSpec:
    """The default uniform frequency grid for pre-whitening ``lc``.

    Runs from ``1/T`` to :func:`default_maximum_frequency` — the pseudo-Nyquist of the
    sampling, floored at :data:`DEFAULT_MAX_FREQUENCY_FLOOR` so short-period pulsators
    stay in band on sparse ground-based data, and capped at the true Nyquist frequency
    of a regular cadence — oversampled by ``samples_per_peak`` (10 by default — a finer
    grid than a plain period search needs, because each peak's position seeds a
    non-linear fit).

    The size is checked before anything is allocated
    (:func:`cuperiod.core.grid.check_grid_size`): one stray timestamp (a ``t = 0``
    placeholder among Julian dates) inflates the baseline, and with it the grid, by
    orders of magnitude.

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

    Raises
    ------
    InsufficientDataError
        If there is no time baseline or the band is empty.
    GridTooLargeError
        If the grid would exceed :data:`cuperiod.core.grid.MAX_GRID_SIZE` samples.

    Warns
    -----
    GridSizeWarning
        If an automatic grid is very large (with a diagnosis of outlying timestamps).
    """
    cfg = settings or PreWhitenSettings()
    finite = lc.finite()
    if finite.baseline <= 0.0:
        raise InsufficientDataError("pre-whitening: no usable time baseline")
    maximum = cfg.maximum_frequency
    if maximum is None:
        maximum = default_maximum_frequency(finite.time, cfg.nyquist_factor)
    minimum = cfg.minimum_frequency
    if minimum is None:
        minimum = 1.0 / finite.baseline
    if minimum >= maximum:
        # An explicit minimum above the automatic maximum is a super-Nyquist request
        # (or one above the pseudo-Nyquist): say which knob opens that band.
        hint = (
            "; the automatic maximum stops at the (pseudo-)Nyquist frequency of the "
            "sampling, so set maximum_frequency to search above it"
            if cfg.maximum_frequency is None
            else ""
        )
        raise InsufficientDataError(
            f"pre-whitening: empty search band ({minimum} .. {maximum} cycles/day)"
            f"{hint}"
        )
    check_grid_size(
        uniform_grid_size(
            finite.baseline,
            maximum_frequency=maximum,
            minimum_frequency=minimum,
            samples_per_peak=cfg.samples_per_peak,
        ),
        automatic=cfg.maximum_frequency is None,
        time=finite.time,
        baseline=finite.baseline,
        maximum_frequency=maximum,
        context="pre-whitening default grid",
    )
    return uniform_frequency_grid(
        finite.baseline,
        maximum_frequency=maximum,
        minimum_frequency=minimum,
        samples_per_peak=cfg.samples_per_peak,
    )


def _peak_fap(
    time: FloatArray,
    error: FloatArray | None,
    power: float,
    *,
    f_max: float,
) -> float:
    """Baluev false-alarm probability of a peak of normalized power ``power``.

    The amplitude spectrum's ``power`` is the standard-normalized generalized
    Lomb-Scargle power by construction, so :func:`baluev_fap` applies directly. An
    undefined statistic (a degenerate spectrum, too few points) yields NaN rather than
    aborting the extraction.
    """
    if not np.isfinite(power) or power <= 0.0:
        return float("nan")
    return float(
        baluev_fap(min(power, 1.0 - 1e-15), time, error, maximum_frequency=f_max)
    )


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
    spectrum: _SpectrumReadout, fit: MultiSineFit, settings: PreWhitenSettings
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
    data: Any, columns: ColumnMap | None, domain: Domain | str | None
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


def _zero_model_fit(
    time: FloatArray, value: FloatArray, error: FloatArray | None, t_ref: float
) -> MultiSineFit:
    """The empty model ``y = 0`` as a :class:`MultiSineFit` (no free parameter).

    The ``K = 0`` solution of a run with ``fit_mean=False``, whose values have already
    had their weighted mean removed; :func:`fit_multisine` itself refuses an empty
    model.
    """
    weight = np.ones(time.size) if error is None else 1.0 / (error * error)
    empty = np.zeros(0, dtype=np.float64)
    residuals = np.array(value, dtype=np.float64)
    return MultiSineFit(
        frequency=empty,
        amplitude=empty.copy(),
        phase=empty.copy(),
        offset=0.0,
        t_ref=t_ref,
        residuals=residuals,
        covariance=np.zeros((0, 0), dtype=np.float64),
        rss=float(np.dot(weight, residuals * residuals)),
        n_samples=int(time.size),
        n_parameters=0,
        weighted=error is not None,
        fit_mean=False,
    )


def prewhiten(
    data: Any,
    *,
    settings: PreWhitenSettings | None = None,
    grid: GridSpec | None = None,
    backend: str | None = None,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
) -> PreWhitenResult:
    """Extract a pulsator's frequency solution automatically.

    Parameters
    ----------
    data : various
        The light curve, in any form :func:`cuperiod.to_input` accepts (a
        :class:`~cuperiod.LightCurve`, a ``(t, y[, dy])`` tuple, a table, ...). Must be
        a single band. Rows may come in any order.
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
        If the light curve has too few finite points, no time baseline, or an error
        column with no usable value.
    GridTooLargeError
        If the default grid would be unreasonably large (typically a stray timestamp).

    Notes
    -----
    Besides the configured stopping criteria, the extraction always stops when another
    component would leave fewer than half the points as degrees of freedom
    (``2 (3K + 1) > N``: a sparse curve otherwise interpolates its own noise with
    ever more sinusoids) and when the residuals are at numerical round-off.

    With ``fit_mean=False`` the offset is not a free parameter of the fits: it is fixed
    at the weighted mean, which is removed once up front — the same zero point the
    amplitude spectrum uses — and reported as ``offset`` (with no error).

    Examples
    --------
    >>> import cuperiod as cup                                     # doctest: +SKIP
    >>> solution = cup.prewhiten((time, mag, mag_err))             # doctest: +SKIP
    >>> print(solution.summary())                                  # doctest: +SKIP
    >>> solution.frequency, solution.frequency_error               # doctest: +SKIP
    """
    # The extraction is a session stepped to its stopping rule with nothing edited:
    # each step is one iteration of the loop this function used to run inline, so
    # the solution is the same, number for number (tests/test_prewhiten_golden.py).
    from cuperiod.prewhiten.session import PreWhitenSession

    session = PreWhitenSession(
        data,
        settings=settings,
        grid=grid,
        backend=backend,
        columns=columns,
        domain=domain,
        record=False,
    )
    session.run()
    return session.finalize()


__all__ = [
    "DEFAULT_MAX_FREQUENCY_FLOOR",
    "default_maximum_frequency",
    "default_prewhiten_grid",
    "prewhiten",
]
