"""Analytic false-alarm probability of a periodogram peak — Baluev (2008).

The extraction loop reports a false-alarm probability for every candidate peak and can
stop on it (``stop_criteria=("fap", ...)``). The statistic is Baluev's alias-free upper
bound (MNRAS 385, 1279, eqn 6) for the *standard-normalized* generalized Lomb-Scargle
power, which is exactly the ``power`` carried by an
:class:`~cuperiod.prewhiten.spectrum.AmplitudeSpectrum`:

.. math::

    \\mathrm{FAP}(z) \\le 1 - \\big(1 - \\mathrm{FAP}_1(z)\\big)\\,e^{-\\tau(z)},

with :math:`\\mathrm{FAP}_1(z) = (1-z)^{(N-3)/2}` the single-frequency tail probability
and :math:`\\tau` the expected number of up-crossings over the searched band,

.. math::

    \\tau(z) = \\gamma\\, f_{\\max} \\sqrt{4\\pi\\,\\mathrm{Var}_w(t)}\\;
               (1-z)^{(N-4)/2} \\sqrt{\\tfrac{1}{2}(N-1)\\,z}.

The implementation reproduces ``astropy.timeseries.LombScargle``'s
``false_alarm_probability(..., method="baluev")`` (see the function's notes on time
zero-points) — it exists so a pre-whitening run never has to import
``astropy.timeseries``, whose first import costs around a second: with thousands of
light curves per worker that is noise, but for the single-star CLI and GUI paths it
roughly doubled the time to the first solution.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core._typing import FloatArray


def baluev_fap(
    power: float | FloatArray,
    time: FloatArray,
    error: FloatArray | None = None,
    *,
    maximum_frequency: float,
) -> float | FloatArray:
    """Baluev (2008) false-alarm probability of a standard-normalized power.

    Parameters
    ----------
    power : float or numpy.ndarray
        Standard-normalized Lomb-Scargle power in ``[0, 1)`` — the ``power`` attribute
        of an :class:`~cuperiod.prewhiten.spectrum.AmplitudeSpectrum`. Values are
        clipped to ``[0, 1]``; a power of exactly 1 would be a perfect fit, so pass
        ``1 - eps`` for numerical sanity.
    time : numpy.ndarray
        Observation times (days). Only their weighted variance enters.
    error : numpy.ndarray, optional
        1-sigma uncertainties; weights the time variance by ``1/error**2``. ``None``
        weights uniformly.
    maximum_frequency : float
        Upper edge of the searched frequency band (cycles/day). The bound scales with
        the band, so quote the band actually searched.

    Returns
    -------
    float or numpy.ndarray
        The false-alarm probability, matching the input's scalar-or-array shape.
        NaN when fewer than four observations make the statistic undefined.

    Notes
    -----
    This is an *upper bound* that ignores aliasing; for strongly aliased sampling it is
    conservative (the true FAP is lower). Matches
    ``astropy.timeseries.LombScargle.false_alarm_probability(method="baluev")`` to
    machine precision for centred times; for raw Julian dates the two differ at the
    ~1e-4 level because astropy evaluates the time variance in one-pass form, which is
    where this implementation is the more accurate of the two.
    """
    from scipy.special import gammaln

    t = np.asarray(time, dtype=np.float64)
    n = int(t.size)
    z = np.clip(np.asarray(power, dtype=np.float64), 0.0, 1.0)
    if n < 4:
        result = np.full(z.shape, np.nan)
        return result if np.ndim(power) else float("nan")
    # The weighted time variance, computed in centred (two-pass) form: the one-pass
    # ``E[t^2] - E[t]^2`` loses ~11 digits on full Julian dates, where the FAP must not
    # depend on the zero-point of the time axis at all.
    if error is None:
        w = np.full(t.shape, 1.0 / n)
    else:
        w = 1.0 / np.square(np.asarray(error, dtype=np.float64))
        w /= w.sum()
    centred = t - float(np.dot(w, t))
    variance = float(np.dot(w, np.square(centred)))
    n_h = n - 1  # degrees of freedom of the constant-only null hypothesis
    n_k = n - 3  # degrees of freedom of the sinusoid-plus-constant model
    # Baluev's Gamma-function prefactor, ~(1 - 0.75/N) for large N.
    prefactor = np.sqrt(2.0 / n_h) * np.exp(
        gammaln(0.5 * n_h) - gammaln(0.5 * (n_h - 1))
    )
    bandwidth = maximum_frequency * np.sqrt(4.0 * np.pi * variance)
    tau = (
        prefactor
        * bandwidth
        * (1.0 - z) ** (0.5 * (n_k - 1))
        * np.sqrt(0.5 * n_h * z)
    )
    single = (1.0 - z) ** (0.5 * n_k)
    # 1 - (1 - single) * exp(-tau), written to stay precise for small probabilities.
    fap = -np.expm1(-tau) + single * np.exp(-tau)
    return fap if np.ndim(power) else float(fap)


__all__ = ["baluev_fap"]
