"""Multi-band multiharmonic Analysis of Variance.

Each band of the same star is fit with its own trigonometric polynomial (independent
harmonic amplitudes and phases) at a *shared* trial frequency, and the per-band
regression and residual sums of squares are pooled into a single F-statistic. This lets
a periodic signal be detected jointly across filters even when no single band constrains
it well, while still allowing the (filter-dependent) harmonic amplitudes to differ.
"""

from __future__ import annotations

from cuperiod.core.config import MHAOVSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.mhaov import aov_multiband_power


def mhaov_multiband_power(
    grid: GridSpec,
    mblc: MultiBandLightCurve,
    settings: MHAOVSettings,
    backend: str,
) -> Periodogram:
    """Compute the pooled multi-band MHAOV power on ``grid``.

    Parameters
    ----------
    grid : GridSpec
        Shared frequency grid (built from the stacked baseline).
    mblc : MultiBandLightCurve
        Two or more bands of one star.
    settings : MHAOVSettings
        MHAOV settings (harmonic order, backend, ...).
    backend : str
        Concrete backend (``"numpy"`` or ``"cupy"``).

    Returns
    -------
    Periodogram
        Pooled AOV F-statistic across bands.

    Raises
    ------
    InsufficientDataError
        If no band has enough finite points for the harmonic order.
    """
    finite = mblc.finite()
    min_points = 2 * settings.n_harmonics + 2
    bands = [
        (lc.time, lc.value)
        for lc in finite.bands.values()
        if lc.n >= min_points
    ]
    if not bands:
        raise InsufficientDataError(
            "MHAOV multiband: no band has enough finite points for "
            f"{settings.n_harmonics} harmonics"
        )
    stacked_time, _, _, _ = finite.stacked()
    baseline = float(stacked_time.max() - stacked_time.min())
    if baseline <= 0.0:
        raise InsufficientDataError("MHAOV multiband: no usable time baseline")

    frequency = grid.frequency
    power = aov_multiband_power(
        frequency,
        bands,
        n_harmonics=settings.n_harmonics,
        backend=backend,  # type: ignore[arg-type]
        batch=settings.batch_periods,
        precision=settings.precision,
    )
    n_total = sum(t.size for t, _ in bands)
    return Periodogram.from_spectrum(
        method="MHAOV",
        backend=backend,
        frequency=frequency,
        power=power,
        objective_sense="max",
        n_samples=n_total,
        baseline=baseline,
        meta={**dict(finite.meta), "bands": finite.band_names},
    )


__all__ = ["mhaov_multiband_power"]
