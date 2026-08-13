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
        if (
            self.mb_model == "flex"
            and self.mb_nterms_base == 0
            and self.mb_nterms_band == 0
        ):
            raise ValueError(
                "flex multi-band model: at least one of mb_nterms_base and "
                "mb_nterms_band must be greater than 0"
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
    mb_model: Literal["offsets", "perband", "flex"] = Field(
        default="offsets",
        description=(
            "Multi-band model: a shared sinusoid with per-band offsets "
            "('offsets', the VanderPlas & Ivezić shared-phase (1,0) model), "
            "independent per-band sinusoids combined with chi2_0 weights "
            "('perband', their multi-phase (0,1) model, eq. 23), or the "
            "flexible regularized model with per-band harmonics ('flex')."
        ),
    )
    mb_nterms_base: int = Field(
        default=1, ge=0, description="Flex model: shared (base) harmonic terms."
    )
    mb_nterms_band: int = Field(
        default=1, ge=0, description="Flex model: per-band harmonic terms."
    )
    mb_reg_base: float | None = Field(
        default=None, description="Flex model: ridge on the base terms (None = 0)."
    )
    mb_reg_band: float | None = Field(
        default=1e-6, description="Flex model: ridge on the per-band terms."
    )
    mb_regularize_by_trace: bool = Field(
        default=True,
        description="Flex model: scale the ridge by the normal-matrix trace.",
    )
    mb_fap_bootstrap: int = Field(
        default=0,
        ge=0,
        description=(
            "Within-band bootstrap resamples for multi-band false-alarm "
            "probabilities (0 disables; the smallest resolvable FAP is "
            "1/(n+1))."
        ),
    )
    mb_fap_seed: int = Field(
        default=0, description="Seed for the multi-band FAP bootstrap."
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
        "auto", "cpu", "gpu", "numba", "numpy", "astropy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=2048, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class PDMSettings(_DeviceSettings):
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
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=2048, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class SuperSmootherSettings(_DeviceSettings):
    """Settings for the SuperSmoother (Friedman variable-span) periodogram."""

    model_config = SettingsConfigDict(
        env_prefix="CUPERIOD_SUPERSMOOTHER_", extra="forbid"
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        spans = tuple(self.primary_spans)
        if any(not (0.0 < s <= 1.0) for s in spans):
            raise ValueError(f"primary_spans must lie in (0, 1], got {spans}")
        if any(b <= a for a, b in zip(spans, spans[1:], strict=False)):
            raise ValueError(f"primary_spans must be strictly increasing, got {spans}")
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
    primary_spans: tuple[float, ...] = Field(
        default=(0.05, 0.2, 0.5),
        min_length=1,
        description=(
            "Candidate span fractions (Friedman's tweeter/midrange/woofer); the "
            "best span is chosen per phase point by cross-validation."
        ),
    )
    middle_span: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Span used to smooth the CV residuals and the chosen spans.",
    )
    final_span: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Span of the final smoothing pass over the blended curve.",
    )
    bass_enhancement: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description=(
            "Friedman's alpha: pull chosen spans toward the largest primary span "
            "(0 = none, 10 = always the largest). None disables the adjustment."
        ),
    )
    n_peaks: int = Field(default=10, ge=1, description="Default stored peak count.")
    peak_separation_rayleigh: float = Field(
        default=3.0, gt=0.0, description="Min peak separation in Rayleigh widths."
    )
    min_detections: int = Field(
        default=20, ge=3, description="Skip if fewer finite points."
    )
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=0,
        ge=0,
        description=(
            "Trial periods per vectorized batch; 0 auto-sizes the batch from a "
            "transient-memory budget (much larger on device backends)."
        ),
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class MHAOVSettings(_DeviceSettings):
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
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=0,
        ge=0,
        description=(
            "Trial frequencies per vectorized batch; 0 auto-sizes the batch from "
            "a transient-memory budget (much larger on device backends)."
        ),
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class CESettings(_DeviceSettings):
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
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=1024, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class StringLengthSettings(_DeviceSettings):
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
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    batch_periods: int = Field(
        default=1024, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class TLSSettings(_DeviceSettings):
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
    backend: Literal[
        "auto", "cpu", "gpu", "numba", "numpy", "cupy", "torch"
    ] = Field(default="auto", description="Compute backend.")
    period_batch: int = Field(
        default=256, ge=1, description="Trial periods per vectorized batch."
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class PreWhitenSettings(_DeviceSettings):
    """Settings for automated iterative pre-whitening of a pulsator.

    The defaults are the conservative, widely-cited choices: a search band from ``1/T``
    to the pseudo-Nyquist frequency — floored at 50 cycles/day so short-period
    pulsators stay in band on sparse ground-based sampling — oversampled ten times,
    extraction until a component fails the Breger et al. (1993) signal-to-noise 4.0
    criterion, a Loumos & Deeming (1978) resolution guard of 1.5 Rayleigh widths
    between components, and least-squares covariance uncertainties inflated by the
    Schwarzenberg-Czerny correlation factor.
    """

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_PREWHITEN_", extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        _require_lt(
            self.minimum_frequency, self.maximum_frequency,
            "minimum_frequency", "maximum_frequency",
        )
        return self

    # -- search grid -------------------------------------------------------------
    minimum_frequency: float | None = Field(
        default=None,
        description="Lowest trial frequency (cycles/day); None -> 1/baseline.",
    )
    maximum_frequency: float | None = Field(
        default=None,
        description=(
            "Highest trial frequency (cycles/day); "
            "None -> pseudo-Nyquist, floored at 50."
        ),
    )
    nyquist_factor: int = Field(
        default=5,
        ge=1,
        description="Pseudo-Nyquist multiple when max is None (as the periodograms).",
    )
    samples_per_peak: int = Field(
        default=10, ge=1, description="Frequency oversampling factor."
    )
    normalization: Literal["lsq", "dft"] = Field(
        default="lsq",
        description="Amplitude convention: least-squares, or classical Deeming DFT.",
    )

    # -- extraction --------------------------------------------------------------
    max_frequencies: int = Field(
        default=30, ge=0, description="Hard cap on extracted components."
    )
    min_separation_rayleigh: float = Field(
        default=1.5,
        ge=0.0,
        description="Resolution guard between components, in Rayleigh widths.",
    )
    refine_bound_rayleigh: float = Field(
        default=1.0,
        gt=0.0,
        description="How far a frequency may move when refined, in Rayleigh widths.",
    )
    fit_mean: bool = Field(
        default=True, description="Fit a free constant term alongside the sinusoids."
    )

    # -- stopping criteria -------------------------------------------------------
    stop_criteria: tuple[Literal["snr", "fap", "bic", "amplitude"], ...] = Field(
        default=("snr",),
        description="Criteria a new component must pass; failing one stops the run.",
    )
    snr_threshold: float = Field(
        default=4.0, gt=0.0, description="Minimum Breger signal-to-noise to accept."
    )
    snr_window: float = Field(
        default=1.0,
        gt=0.0,
        description="Half-width (cycles/day) of the residual noise box.",
    )
    noise_estimator: Literal["mean", "median"] = Field(
        default="mean",
        description="Box noise statistic; 'mean' matches the classical S/N scale.",
    )
    fap_threshold: float = Field(
        default=1e-3, gt=0.0, le=1.0, description="Maximum accepted false-alarm prob."
    )
    min_delta_bic: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum BIC improvement required of a new component.",
    )
    min_amplitude: float | None = Field(
        default=None, description="Absolute amplitude floor; None disables the check."
    )
    prune: bool = Field(
        default=True,
        description="Re-check significance after the final fit and drop failures.",
    )
    blend_tolerance: float = Field(
        default=2.0,
        gt=1.0,
        description=(
            "Flag a component as blended when its fitted amplitude and the spectrum's "
            "own reading differ by more than this factor either way."
        ),
    )

    # -- fitting -----------------------------------------------------------------
    refine: Literal["none", "last", "cyclic", "simultaneous"] = Field(
        default="last", description="Per-iteration frequency-refinement policy."
    )
    sweeps: int = Field(
        default=1, ge=1, description="Cyclic sweeps per iteration and per final polish."
    )
    final_refine: bool = Field(
        default=True,
        description="Polish the accepted solution with a simultaneous (or cyclic) fit.",
    )
    max_simultaneous: int = Field(
        default=60,
        ge=1,
        description="Largest component count given a simultaneous final polish.",
    )
    max_nfev: int = Field(
        default=200, ge=1, description="Optimiser evaluation cap per non-linear solve."
    )

    # -- uncertainties -----------------------------------------------------------
    uncertainty: Literal["covariance", "analytic", "bootstrap"] = Field(
        default="covariance", description="Uncertainty estimator."
    )
    correlation_correction: bool = Field(
        default=True,
        description="Inflate errors by sqrt(D) for correlated residuals.",
    )
    n_resamples: int = Field(
        default=200,
        ge=2,
        description="Bootstrap replicates when uncertainty='bootstrap'.",
    )
    seed: int = Field(default=0, description="Seed for the bootstrap resampling.")

    # -- combination frequencies -------------------------------------------------
    combinations: bool = Field(
        default=True, description="Identify combination frequencies and harmonics."
    )
    combination_max_order: int = Field(
        default=2, ge=1, description="Largest sum|n_i| considered."
    )
    combination_parents: int = Field(
        default=5, ge=1, description="Highest-amplitude components usable as parents."
    )
    combination_tolerance_rayleigh: float = Field(
        default=0.25, ge=0.0, description="Tolerance floor in Rayleigh widths."
    )
    combination_sigma: float = Field(
        default=3.0, ge=0.0, description="Tolerance in propagated sigma_f units."
    )

    # -- execution ---------------------------------------------------------------
    min_detections: int = Field(
        default=20, ge=4, description="Skip if fewer finite points."
    )
    backend: Literal[
        "auto", "cpu", "gpu", "finufft", "cufinufft", "torch", "numpy"
    ] = Field(default="auto", description="Amplitude-spectrum backend.")
    nufft_eps: float = Field(
        default=1e-9, gt=0.0, description="NUFFT relative tolerance."
    )
    direct_freq_batch: int = Field(
        default=4096, ge=1, description="Frequency chunk for the direct (torch) path."
    )
    store_spectra: bool = Field(
        default=True,
        description="Keep the full initial and residual spectra in the result.",
    )
    downsample_points: int = Field(
        default=2000, ge=2, description="Stored downsampled-spectrum size."
    )


class SpacingSettings(BaseSettings):
    """Settings for the g-mode period-spacing tools."""

    model_config = SettingsConfigDict(env_prefix="CUPERIOD_SPACING_", extra="forbid")

    minimum_spacing: float | None = Field(
        default=None,
        description="Shortest trial spacing (days); None -> half the smallest gap.",
    )
    maximum_spacing: float | None = Field(
        default=None,
        description="Longest trial spacing (days); None -> the full period range.",
    )
    oversample: int = Field(
        default=20, ge=1, description="Oversampling of the spacing search grid."
    )
    amplitude_weighted: bool = Field(
        default=True, description="Weight the comb response by mode amplitude."
    )
    max_gap: int = Field(
        default=3,
        ge=1,
        description="Largest number of missing modes bridged inside a series.",
    )
    tolerance: float = Field(
        default=0.25,
        gt=0.0,
        description="Series membership tolerance as a fraction of the local spacing.",
    )
    min_length: int = Field(
        default=4, ge=3, description="Shortest reported period-spacing series."
    )
    ell: int = Field(
        default=1, ge=1, description="Spherical degree assumed for the buoyancy radius."
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
    "PreWhitenSettings",
    "SpacingSettings",
    "StringLengthSettings",
    "SuperSmootherSettings",
    "TLSSettings",
]
