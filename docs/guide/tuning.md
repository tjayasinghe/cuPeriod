# Tuning: settings & grids

Defaults aim at general variable-star light curves and work out of the box. When you need
to tune a run — change the frequency range, the number of harmonics, the box durations —
you pass a settings model, set an environment variable, or supply your own grid.

## Settings models

Each method has a settings model with documented, defaulted fields (a pydantic model):
{class}`~cuperiod.GLSSettings`, {class}`~cuperiod.BLSSettings`,
{class}`~cuperiod.PDMSettings`, {class}`~cuperiod.CESettings`,
{class}`~cuperiod.MHAOVSettings`, {class}`~cuperiod.StringLengthSettings`,
{class}`~cuperiod.SuperSmootherSettings`, {class}`~cuperiod.TLSSettings`. Pass one via
`settings=`:

```python
pg = cup.periodogram(lc, "GLS",
                     settings=cup.GLSSettings(samples_per_peak=10, fap_method="baluev"))
```

For several methods at once, pass a `{method: settings}` mapping — each method picks out
its own:

```python
res = cup.periodogram(lc, ["GLS", "BLS"], settings={
    "GLS": cup.GLSSettings(samples_per_peak=10),
    "BLS": cup.BLSSettings(min_period_days=0.5, max_period_days=30.0),
})
```

### Settings shared by most methods

The Fourier and fold methods (GLS, PDM, CE, String-Length, MHAOV, SuperSmoother) share a
common set of knobs that shape the **trial grid** and **peak selection**:

```{list-table}
:header-rows: 1
:widths: 30 14 56

* - Field
  - Default
  - Meaning
* - `minimum_frequency`
  - `None`
  - Lowest trial frequency (cycles/day). `None` → `1/baseline`.
* - `maximum_frequency`
  - `None`
  - Highest trial frequency. `None` → a pseudo-Nyquist limit.
* - `nyquist_factor`
  - `5`
  - Multiple of the pseudo-Nyquist used when `maximum_frequency` is `None` — the *average*
    rate `0.5·N/T` for GLS, the *median*-sampling rate `0.5/median(Δt)` for every other
    method in this table (MHAOV included).
* - `samples_per_peak`
  - `5`
  - Frequency oversampling — higher = finer grid, more compute.
* - `n_peaks`
  - `10`
  - Default number of peaks stored/reported.
* - `peak_separation_rayleigh`
  - `3.0`
  - Minimum peak separation, in Rayleigh widths.
* - `min_detections`
  - `10`–`20`
  - Skip the light curve if it has fewer finite points (raises
    {exc}`~cuperiod.InsufficientDataError`).
* - `backend`
  - `"auto"`
  - Default backend for this method ({doc}`backends`).
* - `downsample_points`
  - `2000`
  - Size of the stored peak-preserving downsampled spectrum.
```

### Method-specific highlights

```{list-table}
:header-rows: 1
:widths: 18 82

* - Method
  - Distinctive settings
* - **GLS**
  - `fit_mean` (floating-mean GLS, default `True`), `fap_method`
    (`"baluev"`/`"naive"`/`"davies"`/`"none"`), `nufft_eps`.
* - **BLS**
  - `min_period_days` / `max_period_days`, `min_transits` (cap the max period so this
    many transits fit), `duration_min_frac` / `duration_max_frac`, `n_durations`,
    `objective` (`"snr"` or `"likelihood"`), `batch_periods`.
* - **TLS**
  - `min_period_days` / `max_period_days`, `n_phase_bins`, `limb_dark_u1` /
    `limb_dark_u2` (quadratic limb darkening), the duration fractions, `period_batch`.
* - **MHAOV**
  - `n_harmonics` (order `H` of the trig-polynomial model).
* - **PDM**
  - `n_bins`, `n_covers` (overlapping Stellingwerf bin sets).
* - **CE**
  - `n_phase_bins`, `n_mag_bins` (the 2-D histogram resolution).
* - **String-Length**
  - the shared grid/peak settings only.
* - **SuperSmoother**
  - `primary_spans` (the candidate span fractions, default `(0.05, 0.2, 0.5)` — strictly
    increasing, each in `(0, 1]`), `middle_span` and `final_span` (the CV-residual and
    final smoothing passes), `bass_enhancement` (Friedman's alpha, `0`–`10`, pulls the
    chosen spans toward the largest; `None` disables it), `batch_periods`. `minimum_frequency`
    earns extra attention here: an integer *multiple* of the true period folds to a
    coherent curve too, so bounding the trial periods from above (the longest period
    searched is `1/minimum_frequency`) is the cleanest way to keep `2P`, `3P`, … from
    crowding the peak list.
```

The {doc}`API reference <../api/index>` lists every field of every model with its type,
default, and constraints.

## Environment-variable overrides

Every field is also settable from the environment as
`CUPERIOD_<METHOD>_<FIELD>` — handy for batch jobs and CI without touching code:

```bash
export CUPERIOD_GLS_SAMPLES_PER_PEAK=7
export CUPERIOD_BLS_MAX_PERIOD_DAYS=30
```

(The BLS/PDM/etc. prefixes follow the model's `env_prefix`; String-Length uses
`CUPERIOD_SL_` and SuperSmoother `CUPERIOD_SUPERSMOOTHER_`.) A settings object you pass
explicitly takes precedence over the environment.

## Custom grids

Normally each method builds a sensible default grid for your light curve. To control the
trial frequencies/periods directly, pass a {class}`~cuperiod.GridSpec` via `grid=`.

A `GridSpec` holds either a **frequency** grid or a **period** grid and exposes both views
(`.frequency`, `.period`, both ascending). Two builders cover the common cases:

```python
# A uniform frequency grid (matches astropy's autofrequency — good for GLS/MHAOV):
grid = cup.uniform_frequency_grid(
    baseline=lc.baseline, maximum_frequency=20.0, samples_per_peak=5,
)
pg = cup.periodogram(lc, "GLS", grid=grid)

# A log-spaced period grid (good for box/fold methods):
grid = cup.log_period_grid(minimum_period=0.2, maximum_period=50.0, n_periods=20000)
pg = cup.periodogram(lc, "PDM", grid=grid)
```

You can also build one from raw values:

```python
import numpy as np
from cuperiod import GridSpec

grid = GridSpec(kind="period", values=np.geomspace(0.5, 10.0, 5000))
```

## Inspecting a grid without computing

To see the default grid a method would use for a light curve — its size and period range —
without running the periodogram, use the CLI:

```bash
cuperiod grid-info star.csv --method GLS
```

This is the quickest way to sanity-check that your period window and resolution are what
you expect before launching a big run.

---

Next: {doc}`multiband` — combining several filters.
