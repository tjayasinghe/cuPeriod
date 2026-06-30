"""Per-method settings models.

Each periodogram has a pydantic settings model with documented, defaulted fields.
They are plain data — pass one to :func:`cuperiod.periodogram` to tune a run. Every
field is also overridable from the environment via ``CUPERIOD_<METHOD>_<FIELD>`` (for
example ``CUPERIOD_GLS_SAMPLES_PER_PEAK=7``), which is convenient for batch jobs.

Defaults aim at general variable-star light curves; the N-best default is 10
throughout, matching :meth:`cuperiod.Periodogram.best_periods`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Backend selector accepted by every method (concrete names are method-specific).
BackendName = str


def _require_lt(lo: float | None, hi: float | None, lo_name: str, hi_name: str) -> None:
    """Raise ``ValueError`` if both bounds are set and ``lo`` is not below ``hi``."""
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"{lo_name} ({lo}) must be < {hi_name} ({hi})")


class _DeviceSettings(BaseSettings):
    """Device and precision selectors shared by torch-capable methods.

    Orthogonal to ``backend``: ``device`` chooses the torch device when the portable
    ``torch`` backend is selected (``"auto"`` picks the best present), and ``precision``
    controls the compute dtype. ``precision="auto"`` is float64 everywhere it is
    supported and float32 only where the device forces it (Apple MPS cannot do float64);
    an explicit ``"float64"`` on MPS raises rather than silently downgrading. Both
    are environment-overridable like every other setting (``CUPERIOD_<METHOD>_DEVICE``).
    """

    device: Literal["auto", "cpu", "cuda", "mps", "xpu"] = Field(
        default="auto", description="Torch device when the torch backend is used."
    )
    precision: Literal["auto", "float64", "float32"] = Field(
        default="auto",
        description="Compute precision; 'auto' is float64 except on MPS (float32).",
    )


class GLSSettings(_DeviceSettings):
    """Settings for the generalized Lomb-Scargle (GLS) periodogram."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_GLS_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description="Highest trial frequency (cycles/day); None -> pseudo-Nyquist.",
    )
    nyquist_factor: int = Field(
        default=5, ge=1, description="Pseudo-Nyquist multiple when max is None."
    )
    samples_per_peak: int = Field(
        default=5, ge=1, description="Frequency oversampling factor."
    )
    fit_mean: bool = Field(
        default=True, description="Floating-mean (generalized) Lomb-Scargle."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    fap_method: Literal["baluev", "naive", "davies", "none"] = Field(
        default="baluev", description="False-alarm-probability method."
    )
    min_detections: int = Field(
        default=10, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal[
        "auto", "cpu", "gpu", "finufft", "cufinufft", "torch", "astropy"
    ] = Field(default="auto", description="Compute backend.")
    nufft_eps: float = Field(
        default=1e-9, gt=0.0, description="NUFFT relative tolerance."
    )
    direct_freq_batch: int = Field(
        default=4096,
        ge=1,
        description="Frequency chunk for the portable (torch) direct trig-sum path.",
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class BLSSettings(_DeviceSettings):
    """Settings for the box least squares (BLS) search."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_BLS_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.min_period_days, self.max_period_days,
            "min_period_days", "max_period_days",
        )
        _require_lt(
            self.duration_min_frac, self.duration_max_frac,
            "duration_min_frac", "duration_max_frac",
        )
        return self

    min_period_days: float = Field(default=0.2, gt=0.0, description="Minimum period.")
    max_period_days: float = Field(default=100.0, gt=0.0, description="Maximum period.")
    min_transits: int = Field(
        default=2, ge=1, description="Cap max period at baseline/min_transits."
    )
    duration_min_frac: float = Field(
        default=0.01, gt=0.0, description="Shortest box as a fraction of period."
    )
    duration_max_frac: float = Field(
        default=0.12, gt=0.0, description="Longest box as a fraction of period."
    )
    n_durations: int = Field(
        default=8, ge=1, description="Trial durations per period segment."
    )
    min_duration_days: float = Field(
        default=0.01, gt=0.0, description="Floor on absolute box duration."
    )
    segment_factor: float = Field(
        default=2.0, gt=1.0, description="Period ratio per log segment."
    )
    grid_duration_frac: float = Field(
        default=0.05, gt=0.0, description="Phase-grid resolution as a duration frac."
    )
    grid_oversample: int = Field(
        default=2, ge=1, description="Frequency-grid oversampling."
    )
    bins_per_duration: int = Field(
        default=10, ge=1, description="Phase bins per shortest duration (oversample)."
    )
    objective: Literal["snr", "likelihood"] = Field(
        default="snr", description="Box objective."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    alias_freq_tolerance: float = Field(
        default=0.0035, ge=0.0, description="Alias-diversity frequency tolerance."
    )
    harmonic_max: int = Field(
        default=8, ge=1, description="Harmonic order for alias diversity."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal[
        "auto", "cpu", "gpu", "numpy", "astropy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=2048, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class PDMSettings(BaseSettings):
    """Settings for phase dispersion minimization (PDM)."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_PDM_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description="Highest trial frequency (cycles/day); None -> pseudo-Nyquist.",
    )
    nyquist_factor: int = Field(
        default=5, ge=1, description="Pseudo-Nyquist multiple when max is None."
    )
    samples_per_peak: int = Field(
        default=5, ge=1, description="Frequency oversampling factor."
    )
    n_bins: int = Field(default=10, ge=2, description="Number of phase bins.")
    n_covers: int = Field(
        default=3, ge=1, description="Overlapping bin sets (Stellingwerf covers)."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal["auto", "cpu", "gpu", "numpy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    batch_periods: int = Field(
        default=2048, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class MHAOVSettings(BaseSettings):
    """Settings for the multiharmonic Analysis of Variance (MHAOV) periodogram."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_MHAOV_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description="Highest trial frequency (cycles/day); None -> pseudo-Nyquist.",
    )
    nyquist_factor: int = Field(
        default=5, ge=1, description="Pseudo-Nyquist multiple when max is None."
    )
    samples_per_peak: int = Field(
        default=5, ge=1, description="Frequency oversampling factor."
    )
    n_harmonics: int = Field(
        default=3, ge=1, description="Harmonic order H of the trig-polynomial model."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    min_detections: int = Field(
        default=20, ge=5, description="Skip if fewer finite points (need > 2H+1)."
    )
    backend: Literal["auto", "cpu", "gpu", "numpy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    batch_periods: int = Field(
        default=512, ge=1, description="Trial frequencies per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class CESettings(BaseSettings):
    """Settings for the conditional-entropy (CE) period search."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_CE_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description="Highest trial frequency (cycles/day); None -> pseudo-Nyquist.",
    )
    nyquist_factor: int = Field(
        default=5, ge=1, description="Pseudo-Nyquist multiple when max is None."
    )
    samples_per_peak: int = Field(
        default=5, ge=1, description="Frequency oversampling factor."
    )
    n_phase_bins: int = Field(default=10, ge=2, description="Phase histogram bins.")
    n_mag_bins: int = Field(default=10, ge=2, description="Magnitude histogram bins.")
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal["auto", "cpu", "gpu", "numpy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    batch_periods: int = Field(
        default=1024, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class StringLengthSettings(BaseSettings):
    """Settings for the string-length (Lafler-Kinman / Dworetsky) period search."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_SL_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description="Highest trial frequency (cycles/day); None -> pseudo-Nyquist.",
    )
    nyquist_factor: int = Field(
        default=5, ge=1, description="Pseudo-Nyquist multiple when max is None."
    )
    samples_per_peak: int = Field(
        default=5, ge=1, description="Frequency oversampling factor."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal["auto", "cpu", "gpu", "numpy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    batch_periods: int = Field(
        default=1024, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class TLSSettings(BaseSettings):
    """Settings for the transit least squares (TLS) search."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_TLS_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.min_period_days, self.max_period_days,
            "min_period_days", "max_period_days",
        )
        _require_lt(
            self.duration_min_frac, self.duration_max_frac,
            "duration_min_frac", "duration_max_frac",
        )
        return self

    min_period_days: float = Field(default=0.5, gt=0.0, description="Minimum period.")
    max_period_days: float = Field(default=100.0, gt=0.0, description="Maximum period.")
    min_transits: int = Field(
        default=2, ge=1, description="Cap max period at baseline/min_transits."
    )
    oversample: int = Field(
        default=2, ge=1, description="Period-grid frequency oversampling."
    )
    grid_duration_frac: float = Field(
        default=0.05,
        gt=0.0,
        description="Period-grid spacing as a transit-width fraction (peak sampling).",
    )
    n_phase_bins: int = Field(
        default=256, ge=16, description="Phase bins for the folded matched filter."
    )
    duration_min_frac: float = Field(
        default=0.01, gt=0.0, description="Shortest transit as a fraction of period."
    )
    duration_max_frac: float = Field(
        default=0.10, gt=0.0, description="Longest transit as a fraction of period."
    )
    n_durations: int = Field(
        default=5, ge=1, description="Trial transit durations."
    )
    limb_dark_u1: float = Field(
        default=0.4, description="Quadratic limb-darkening coefficient u1."
    )
    limb_dark_u2: float = Field(
        default=0.3, description="Quadratic limb-darkening coefficient u2."
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    alias_freq_tolerance: float = Field(
        default=0.0035, ge=0.0, description="Alias-diversity frequency tolerance."
    )
    harmonic_max: int = Field(
        default=8, ge=1, description="Harmonic order for alias diversity."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal["auto", "cpu", "gpu", "numpy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    period_batch: int = Field(
        default=256, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class BatchSettings(BaseSettings):
    """Settings for batch processing of many light curves."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_BATCH_", extra="forbid")

    device: Literal["cpu", "gpu", "hybrid"] = Field(
        default="cpu", description="Where to run the workers."
    )
    workers: int | None = Field(
        default=None, description="Worker count; None -> auto (cores or GPU sizing)."
    )
    n_best: int = Field(default=10, ge=1, description="Peaks stored per light curve.")
    store_raw: bool = Field(
        default=False, description="Store the downsampled raw periodogram."
    )
    resume: bool = Field(default=True, description="Skip already-computed inputs.")
    gpu_headroom: float = Field(
        default=0.20, ge=0.0, lt=1.0, description="GPU free-memory headroom fraction."
    )


__all__ = [
    "BackendName",
    "BLSSettings",
    "BatchSettings",
    "CESettings",
    "GLSSettings",
    "MHAOVSettings",
    "PDMSettings",
    "StringLengthSettings",
    "TLSSettings",
]
