"""Result objects for pre-whitening: :class:`Sinusoid` and :class:`PreWhitenResult`.

A :class:`PreWhitenResult` is the frequency solution of one light curve — the ranked
list of extracted sinusoids with their uncertainties and signal-to-noise, the residuals
and their spectrum, the fit statistics, and an explicit record of *why the extraction
stopped*. That last part matters: a frequency list without its stopping criterion cannot
be reproduced or compared with anyone else's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.prewhiten.combinations import Combination
from cuperiod.prewhiten.spectrum import AmplitudeSpectrum

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Sinusoid:
    """One extracted sinusoidal component.

    Attributes
    ----------
    rank : int
        1-based extraction order (rank 1 was the strongest peak of the original data).
    label : str
        Display label ``F1``, ``F2``, ... matching ``rank``.
    frequency, frequency_error : float
        Frequency and its 1-sigma uncertainty in cycles/day.
    amplitude, amplitude_error : float
        Amplitude and its 1-sigma uncertainty, in the units of the input.
    phase, phase_error : float
        Phase and its 1-sigma uncertainty in radians, for the convention
        ``A sin(2*pi*f*(t - t_ref) + phase)``.
    snr : float
        Amplitude divided by the local noise level of the residual spectrum after this
        component was removed (the Breger et al. 1993 criterion).
    fap : float
        False-alarm probability of the peak in the spectrum it was drawn from, or NaN if
        it could not be computed.
    delta_bic : float
        Change in the Bayesian information criterion when this component was added.
        Negative means the component improved the model.
    combination : str or None
        Identification as a combination of stronger components, if any.
    """

    rank: int
    label: str
    frequency: float
    frequency_error: float
    amplitude: float
    amplitude_error: float
    phase: float
    phase_error: float
    snr: float
    fap: float = float("nan")
    delta_bic: float = float("nan")
    combination: str | None = None

    @property
    def period(self) -> float:
        """Period in days."""
        return 1.0 / self.frequency if self.frequency != 0.0 else float("inf")

    @property
    def period_error(self) -> float:
        """1-sigma period uncertainty in days (``sigma_P = sigma_f / f^2``)."""
        if self.frequency == 0.0:
            return float("nan")
        return self.frequency_error / (self.frequency * self.frequency)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict (also the batch/CSV row layout)."""
        return {
            "rank": self.rank,
            "label": self.label,
            "frequency": self.frequency,
            "frequency_error": self.frequency_error,
            "period": self.period,
            "period_error": self.period_error,
            "amplitude": self.amplitude,
            "amplitude_error": self.amplitude_error,
            "phase": self.phase,
            "phase_error": self.phase_error,
            "snr": self.snr,
            "fap": self.fap,
            "delta_bic": self.delta_bic,
            "combination": self.combination,
        }


@dataclass(frozen=True)
class PreWhitenResult:
    """The frequency solution of one light curve.

    Attributes
    ----------
    components : tuple of Sinusoid
        The extracted sinusoids, in extraction order (strongest first).
    combinations : tuple of Combination
        Components identified as integer combinations of stronger ones.
    offset, offset_error : float
        The fitted constant term and its uncertainty.
    t_ref : float
        Epoch the phases are referenced to (days).
    time, residuals : numpy.ndarray
        The times used and ``value - model``, aligned.
    n_samples : int
        Number of finite points used.
    baseline : float
        Time span in days (the Rayleigh resolution is ``1/baseline``).
    stop_reason : str
        Why the extraction stopped — always populated.
    n_iterations : int
        Extraction attempts made, including the rejected final one.
    n_pruned : int
        Components dropped by the final significance re-check (see
        :attr:`~cuperiod.PreWhitenSettings.prune`).
    rms, chi2, reduced_chi2, bic : float
        Fit statistics of the accepted solution.
    correlation_factor : float
        Schwarzenberg-Czerny ``D`` applied to the uncertainties (1.0 if uncorrected).
    uncertainty_method : str
        Which estimator produced the reported errors.
    backend : str
        Concrete spectrum backend that ran.
    spectrum, residual_spectrum : AmplitudeSpectrum or None
        Amplitude spectra of the original data and of the residuals (``None`` when the
        run was told not to keep them, as in batch mode).
    window : AmplitudeSpectrum or None
        The spectral window ``|W(f)|`` of the sampling, for alias diagnosis (``None``
        when spectra are not kept).
    meta : Mapping
        Free-form metadata carried from the light curve.
    """

    components: tuple[Sinusoid, ...]
    combinations: tuple[Combination, ...]
    offset: float
    offset_error: float
    t_ref: float
    time: FloatArray
    residuals: FloatArray
    n_samples: int
    baseline: float
    stop_reason: str
    n_iterations: int
    n_pruned: int
    rms: float
    chi2: float
    reduced_chi2: float
    bic: float
    correlation_factor: float
    uncertainty_method: str
    backend: str
    spectrum: AmplitudeSpectrum | None = None
    residual_spectrum: AmplitudeSpectrum | None = None
    window: AmplitudeSpectrum | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    # -- convenience views -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Any:
        return iter(self.components)

    def __getitem__(self, index: int) -> Sinusoid:
        return self.components[index]

    @property
    def n_components(self) -> int:
        """Number of extracted sinusoids."""
        return len(self.components)

    @property
    def rayleigh(self) -> float:
        """Rayleigh frequency resolution ``1/baseline`` (cycles/day)."""
        return 1.0 / self.baseline if self.baseline > 0.0 else float("inf")

    def _column(self, name: str) -> FloatArray:
        return np.asarray(
            [getattr(c, name) for c in self.components], dtype=np.float64
        )

    @property
    def frequency(self) -> FloatArray:
        """Component frequencies (cycles/day)."""
        return self._column("frequency")

    @property
    def frequency_error(self) -> FloatArray:
        """Component frequency uncertainties (cycles/day)."""
        return self._column("frequency_error")

    @property
    def period(self) -> FloatArray:
        """Component periods (days)."""
        return self._column("period")

    @property
    def amplitude(self) -> FloatArray:
        """Component amplitudes."""
        return self._column("amplitude")

    @property
    def amplitude_error(self) -> FloatArray:
        """Component amplitude uncertainties."""
        return self._column("amplitude_error")

    @property
    def phase(self) -> FloatArray:
        """Component phases (radians)."""
        return self._column("phase")

    @property
    def snr(self) -> FloatArray:
        """Component signal-to-noise ratios."""
        return self._column("snr")

    def independent(self) -> tuple[Sinusoid, ...]:
        """Components not identified as combinations of stronger ones.

        The candidate independent mode list — what a mode-identification or
        period-spacing analysis should be run on.
        """
        return tuple(c for c in self.components if c.combination is None)

    # -- evaluation --------------------------------------------------------------
    def model(self, time: FloatArray, *, include_offset: bool = True) -> FloatArray:
        """Evaluate the fitted multi-sine model at ``time``."""
        t = np.asarray(time, dtype=np.float64)
        out = np.full(t.shape, self.offset if include_offset else 0.0, dtype=np.float64)
        dt = t - self.t_ref
        for c in self.components:
            out += c.amplitude * np.sin(_TWO_PI * c.frequency * dt + c.phase)
        return out

    # -- serialization -----------------------------------------------------------
    def to_table(self) -> list[dict[str, Any]]:
        """The component list as plain dicts, one row per component."""
        return [c.to_dict() for c in self.components]

    def to_dataframe(self) -> Any:
        """The component list as a pandas ``DataFrame`` (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.to_table())

    def to_dict(self, *, include_residuals: bool = False) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict."""
        out: dict[str, Any] = {
            "n_components": self.n_components,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "rayleigh": self.rayleigh,
            "t_ref": self.t_ref,
            "offset": self.offset,
            "offset_error": self.offset_error,
            "stop_reason": self.stop_reason,
            "n_iterations": self.n_iterations,
            "n_pruned": self.n_pruned,
            "rms": self.rms,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "bic": self.bic,
            "correlation_factor": self.correlation_factor,
            "uncertainty_method": self.uncertainty_method,
            "backend": self.backend,
            "components": self.to_table(),
            "combinations": [c.to_dict() for c in self.combinations],
        }
        if include_residuals:
            out["time"] = self.time.tolist()
            out["residuals"] = self.residuals.tolist()
        return out

    def summary(self, *, max_rows: int = 50) -> str:
        """A formatted, human-readable report of the solution."""
        head = (
            f"Pre-whitening: {self.n_components} component"
            f"{'' if self.n_components == 1 else 's'} from {self.n_samples} points "
            f"over {self.baseline:.4g} d  (backend={self.backend})"
        )
        stop = f"  stopped: {self.stop_reason}"
        stats = (
            f"  residual rms {self.rms:.6g}   reduced chi2 {self.reduced_chi2:.4g}"
            f"   D={self.correlation_factor:.2f}   errors: {self.uncertainty_method}"
        )
        header = (
            f"  {'ID':<4} {'frequency (1/d)':>17} {'+/-':>11} "
            f"{'amplitude':>13} {'+/-':>11} {'phase':>8} {'S/N':>7}  note"
        )
        lines = [head, stop, stats, "", header]
        for c in self.components[:max_rows]:
            note = c.combination or ""
            lines.append(
                f"  {c.label:<4} {c.frequency:>17.9g} {c.frequency_error:>11.3g} "
                f"{c.amplitude:>13.6g} {c.amplitude_error:>11.3g} "
                f"{c.phase:>8.4f} {c.snr:>7.2f}  {note}"
            )
        if self.n_components > max_rows:
            lines.append(f"  ... {self.n_components - max_rows} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PreWhitenResult({self.n_components} components, "
            f"n={self.n_samples}, stop={self.stop_reason!r})"
        )


__all__ = ["PreWhitenResult", "Sinusoid"]
