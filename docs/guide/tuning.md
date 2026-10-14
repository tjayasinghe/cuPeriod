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

Selection is strict. Mapping keys are matched like method names (`"String-Length"`,
`"string_length"` and `"StringLength"` are one key); a key naming no method raises
{exc}`~cuperiod.UnknownMethodError`, two keys for one method raise `ValueError`, and a
settings object of the wrong class raises `TypeError` instead of being silently replaced
by defaults. Keys for registered methods that are not in the run are ignored. Values are
validated when the model is built: trial-frequency bounds must be finite with
`minimum_frequency >= 0` (pre-whitening: `> 0`) and `maximum_frequency > 0`, BLS/TLS
duration fractions below 1, TLS limb darkening finite with `u1 + u2 <= 1`, and
thresholds such as `min_amplitude` finite and non-negative.

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
  - Highest trial frequency. `None` → a pseudo-Nyquist limit (see [below](#the-automatic-frequency-ceiling)).
* - `nyquist_factor`
  - `5`
  - Multiple of the pseudo-Nyquist used when `maximum_frequency` is `None` — the *average*
    rate `0.5·N/T` for GLS, the *median*-sampling rate `0.5/median(Δt)` for every other
    method in this table (MHAOV included).
* - `samples_per_peak`
  - `5`
  - Frequency oversampling — higher = finer grid, more compute. (The GUI's default is
    10.)
* - `n_peaks`
  - `10`
  - Default number of peaks of {meth}`~cuperiod.Periodogram.best_periods` and
    `to_dict()`.
* - `peak_separation_rayleigh`
  - `3.0`
  - Default minimum peak separation of `best_periods()`, in Rayleigh widths.
* - `min_detections`
  - `10`–`20`
  - Skip the light curve if it has fewer finite points (raises
    {exc}`~cuperiod.InsufficientDataError`). For multi-band input it counts the points
    of the bands that pass each method's per-band floor.
* - `backend`
  - `"auto"`
  - Default backend for this method ({doc}`backends`): `cup.periodogram(...,
    backend="auto")` (or `None`) defers to it, any other `backend=` overrides it. A pin
    such as `"gpu"` on a machine without one raises
    {exc}`~cuperiod.BackendUnavailableError` rather than quietly running on the CPU.
* - `downsample_points`
  - `2000`
  - Default size of `downsample()` — the stored peak-preserving downsampled spectrum.
```

The peak fields (`n_peaks`, `peak_separation_rayleigh`, `downsample_points`, and for
BLS/TLS `harmonic_max` and `alias_freq_tolerance`) travel with every result of
{func}`cuperiod.periodogram`, the batch drivers, the LINCC adapters and the GUI as its
defaults; an explicit argument to `best_periods()`, `downsample()` or `to_dict()` still
wins. `alias_freq_tolerance` (BLS/TLS, 0.0035 c/d) is the alias/harmonic tolerance of
`best_periods(alias_diverse=True)`; a bare `Periodogram` built by hand falls back to one
Rayleigh width.

### The automatic frequency ceiling

- **Regular cadences.** For GLS on strictly regular sampling (constant cadence, gaps
  allowed — Kepler, TESS) the default grid stops at the true Nyquist frequency
  `0.5/cadence`, above which the spectrum is a mirror image and the best period often
  an image. Set `maximum_frequency` or `nyquist_factor` to search past it (super-Nyquist
  work); {doc}`prewhitening` does the same.
- **Multi-exposure cadences.** When most gaps fall inside visits of at most ten
  back-to-back exposures — dithered triplets, filter sequences, near-simultaneous bands
  stacked by a multi-band fit — the median-gap pseudo-Nyquist would be enormous, so the
  automatic ceiling is limited to 1e5 Rayleigh widths, i.e. `1e5 / baseline` c/d: about
  55 c/d on a 5-year baseline and 27 c/d on a 10-year survey such as LSST, which can cut
  off the fastest δ Scuti modes. A `nyquist_factor` above 5 widens the budget in
  proportion (`nyquist_factor=20` gives ~110 c/d on 10 years), and `maximum_frequency`
  restores any band. The capped grid is a prefix of the uncapped one; ordinary sampling
  is unchanged. The budget is `cuperiod.core.grid.MULTI_EXPOSURE_RAYLEIGH_BUDGET`.
- **Size guard.** Every default grid is sized before it is allocated. Above 5 million
  automatic samples it warns ({class}`~cuperiod.GridSizeWarning`) and is built; when a
  few epochs dominate the baseline the warning says so — legitimate for archival points
  combined with a new campaign, a bug for a `0` or sentinel time or a JD/MJD mix-up.
  Above `MAX_GRID_SIZE` (250 million samples, 2 GB per array) a grid raises
  {class}`~cuperiod.GridTooLargeError`; pass `max_size=` to
  {func}`~cuperiod.uniform_frequency_grid` or raise
  `cuperiod.core.grid.MAX_GRID_SIZE` for a deliberately larger grid.
  {func}`~cuperiod.uniform_grid_size` and {func}`~cuperiod.check_grid_size` apply the same
  rules to a grid you are about to build, {func}`~cuperiod.default_frequency_grid` is the
  shared recipe of the fold methods' default grids, and
  {func}`~cuperiod.regular_cadence_nyquist` returns `0.5/cadence` for lattice sampling
  (or `None`).

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
    `objective` (`"snr"` or `"likelihood"`), `batch_periods` (an upper bound: the
    numpy/torch batch is capped at 512 MiB on the host, 1 GiB on a GPU). A custom grid
    is searched exactly, split into log-period segments (`segment_factor`) whose box
    durations track the period, as on the default grid.
* - **TLS**
  - `min_period_days` / `max_period_days`, `n_phase_bins`, `limb_dark_u1` /
    `limb_dark_u2` (quadratic limb darkening), the duration fractions, `period_batch`
    (an upper bound, capped by a 128 MiB host / 256 MiB GPU budget).
    `grid_duration_frac` is capped at `2 × duration_min_frac`, so the default period grid
    resolves the shortest searched transit; raise `duration_min_frac` to coarsen it.
* - **MHAOV**
  - `n_harmonics` (order `H` of the trig-polynomial model).
* - **PDM**
  - `n_bins`, `n_covers` (overlapping Stellingwerf bin sets), `batch_periods`.
* - **CE**
  - `n_phase_bins`, `n_mag_bins` (the 2-D histogram resolution), `batch_periods`.
* - **String-Length**
  - the shared grid/peak settings, and `batch_periods`.
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

For PDM, CE and String-Length `batch_periods` is an upper bound: on long light curves
the chunk of trial periods is lowered so the transients stay near 128 MiB (512 MiB on a
GPU); results do not change.

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
explicitly takes precedence over the environment. `CUPERIOD_<METHOD>_BACKEND` sets the
method's default backend, which `backend="auto"` defers to; the GUI seeds its Options
form from these variables too. (`CUPERIOD_BATCH_*` has no effect: `BatchSettings` is
reserved.)

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

A `GridSpec` validates its values: an unknown `kind`, non-finite, descending,
negative-frequency or non-positive-period values, and `uniform=True` on values that are
not an arithmetic progression raise `ValueError` (rounding-level steps back, up to 1e-9
of the grid's scale — as between concatenated `np.arange` segments — are accepted). The
builders raise a `ValueError` naming the argument for a non-finite or non-positive
baseline or `samples_per_peak`, non-finite bounds or a negative `minimum_frequency`.

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
