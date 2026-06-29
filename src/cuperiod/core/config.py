"""Per-method settings models.

Each periodogram has a pydantic settings model with documented, defaulted fields.
They are plain data — pass one to :func:`cuperiod.periodogram` to tune a run. Every
field is also overridable from the environment via ``CUPERIOD_<METHOD>_<FIELD>`` (for
example ``CUPERIOD_GLS_SAMPLES_PER_PEAK=7``), which is convenient for batch jobs.

Defaults aim at general variable-star light curves; the N-best default is 10
throughout, matching :meth:`cuperiod.Periodogram.best_periods`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Backend selector accepted by every method (concrete names are method-specific).
BackendName = str


class GLSSettings(BaseSettings):
    """Settings for the generalized Lomb-Scargle (GLS) periodogram."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_GLS_", extra="forbid")

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
    backend: Literal["auto", "cpu", "gpu", "finufft", "cufinufft", "astropy"] = Field(
        default="auto", description="Compute backend."
    )
    nufft_eps: float = Field(
        default=1e-9, gt=0.0, description="NUFFT relative tolerance."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class BLSSettings(BaseSettings):
    """Settings for the box least squares (BLS) search."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_BLS_", extra="forbid")

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
    backend: Literal["auto", "cpu", "gpu", "numpy", "astropy", "cupy"] = Field(
        default="auto", description="Compute backend."
    )
    batch_periods: int = Field(
        default=2048, ge=1, description="Trial periods per vectorized batch."
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


__all__ = ["BackendName", "BLSSettings", "BatchSettings", "GLSSettings"]
