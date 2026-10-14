# Choosing a method

cuPeriod ships eight period-search methods. They all take the same inputs and return the
same {class}`~cuperiod.Periodogram`, so trying several is cheap — but picking the right one
for your signal saves time and gives cleaner peaks. This page is a decision guide.

## Quick decision guide

```{list-table}
:header-rows: 1
:widths: 30 18 52

* - Your signal
  - Method
  - Why
* - General periodic variable (smooth, roughly sinusoidal)
  - **GLS**
  - Fast, well-calibrated, gives a false-alarm probability. The default and the right
    first thing to try.
* - Eclipses or box-like transits
  - **BLS**
  - Matches a flat-bottomed box; returns depth, duration, and mid-transit time.
* - Exoplanet transit (rounded, limb-darkened)
  - **TLS**
  - A physical limb-darkened transit template — more sensitive than a box on real
    planet transits.
* - Sharply non-sinusoidal (e.g. RR Lyrae, Cepheids)
  - **MHAOV**
  - A multiharmonic model captures sharp/asymmetric shapes a single sinusoid misses.
* - Non-sinusoidal folds, general shape
  - **PDM**
  - Minimizes the scatter of the folded curve — no assumption about the waveform.
* - Sparse, gappy survey light curves
  - **CE**
  - Conditional entropy is robust when coverage is poor and amplitudes are uneven.
* - Eclipsing / eccentric, want a shape-free statistic
  - **String-Length**
  - Minimizes the path length through the folded curve; cheap and assumption-light.
* - Any repeating shape, without picking a harmonic budget
  - **SuperSmoother**
  - Fits the fold itself with a variable-span smoother — fully non-parametric. Watch the
    integer-multiple caveat below.
```

Not sure? Run a few at once and compare — see {ref}`several-methods` below.

## Objective sense: bigger vs smaller is better

Each method's statistic is either **maximized** or **minimized** at the true period:

- **Maximized** (a tall peak = significant): GLS, BLS, MHAOV, TLS, SuperSmoother.
- **Minimized** (a deep trough = significant): PDM, CE, String-Length.

You don't have to track this — {meth}`~cuperiod.Periodogram.best_periods` knows each
method's `objective_sense` and always returns the *most significant* periods first. It
matters only if you inspect the raw `power` array yourself ({doc}`results`).

## Multi-band

Seven of the eight take several filters of the same star and fit them jointly: **GLS, BLS,
MHAOV, PDM, CE, String-Length, and SuperSmoother**. Only TLS is single-band. Pass a
{class}`~cuperiod.MultiBandLightCurve` instead of a {class}`~cuperiod.LightCurve` and the
method's joint model runs; a single-band method asked for a multi-band run raises a clear
error. See {doc}`multiband`.

## The methods in detail

### GLS — generalized Lomb–Scargle

The workhorse for periodic variable stars. cuPeriod computes the floating-mean
Lomb–Scargle power of Zechmeister & Kürster (2009) — the same statistic as astropy's
`LombScargle(..., fit_mean=True)` — but evaluates the trigonometric sums with a
non-uniform FFT, matching astropy to ~1e-9 while running much faster. Each peak carries a
false-alarm probability (`extra["fap"]`).

```python
pg = cup.periodogram(lc, "GLS")
```

Key settings ({class}`~cuperiod.GLSSettings`): `samples_per_peak`, `nyquist_factor`,
`fit_mean`, `fap_method`, `minimum_frequency` / `maximum_frequency`.

- On strictly regular sampling the default grid stops at the true Nyquist frequency
  `0.5/cadence` (above it the spectrum is a mirror image); set `maximum_frequency` or
  `nyquist_factor` to search past it, e.g. for super-Nyquist Kepler long-cadence work
  ({doc}`tuning`).
- FAPs are computed on relative times (on absolute JDs
  astropy's effective-baseline estimate was rounding noise for runs shorter than a
  day), and a best peak on the edge of the grid now gets a FAP — a conservative upper
  bound, because the true maximum may lie outside the band.
- The data are centred in float64 before the transform, so precision no longer drops
  with (mean/scatter)²; power is clipped to [0, 1] and an exactly constant light curve
  has zero power on every backend.
- `PeriodogramMethod.make_engine("finufft")` returns a plan-reusing
  {class}`~cuperiod.methods.gls.FinufftGLS` engine (one per thread or process), used by
  {func}`~cuperiod.pdot_search`, {func}`~cuperiod.sliding_periodogram` and batch runs.

{doc}`Multi-band <multiband>` GLS is native on every backend and offers three joint
models (`mb_model`): a shared-phase sinusoid with per-band offsets (`"offsets"`, the
default), independent per-band sinusoids (`"perband"`), and a regularized model with
per-band harmonics (`"flex"`). Multi-band false-alarm probabilities come from a
within-band bootstrap ({func}`~cuperiod.multiband_fap`, or `mb_fap_bootstrap`).

### BLS — box least squares

For eclipsing binaries and box-shaped transits. Searches log-spaced period segments so the
box-duration grid tracks the period, exactly as a transit's duration scales. **Works in
flux** (magnitudes are converted automatically), because an eclipse is a dip. Each peak
carries the box parameters: `depth`, `duration`, `t0` (mid-transit time), `depth_snr`, and
the signal-detection efficiency `sde`. Supports {doc}`multi-band <multiband>`.

```python
pg = cup.periodogram(lc, "BLS")
for peak in pg.best_periods(5, alias_diverse=True):     # alias-aware for box searches
    print(peak.period, peak.extra["depth"], peak.extra["duration"])
```

With the `[fast]` extra, BLS's CPU backend is a multicore `numba` search ~20× faster than
astropy (PDM, CE, String-Length, MHAOV, TLS, and SuperSmoother gain `numba` CPU kernels
too — see {doc}`backends`). Key settings ({class}`~cuperiod.BLSSettings`):
`min_period_days` / `max_period_days`, `duration_min_frac` / `duration_max_frac`,
`n_durations`, `objective`.

A grid you pass is searched exactly — split into log-period segments
(`segment_factor`) whose box durations track the period, as on the default grid. The flux
is median-centred in float64 first, as astropy does. Unlike astropy, boxes holding all or
none of the inverse-variance weight are rejected by a **relative** floor rather than an
absolute `DBL_EPSILON`: on clustered sampling the absolute floor fabricated likelihood
peaks of ~1e6, and with the relative one results no longer depend on the flux or error
scale.

### TLS — transit least squares

A matched filter against a **limb-darkened transit** template (not a box), more sensitive
than BLS on real exoplanet transits. Works in flux. Key settings
({class}`~cuperiod.TLSSettings`): `min_period_days` / `max_period_days`, `n_phase_bins`,
`limb_dark_u1` / `limb_dark_u2`, the duration fractions.

The out-of-transit level is fitted jointly with the depth, so depths are unbiased (a
box-free fit used to underestimate them by about the duty cycle). Each peak's `t0` is an
absolute epoch on your time axis, like BLS's. The default period grid step is tied to
`2 × duration_min_frac`, so it resolves the shortest transit searched, and a constant
flux gives an all-zero spectrum. Float32 backends (torch, Apple MPS, `precision="float32"`
on CUDA) rescale the flux internally, so physical flux units work too.

```python
pg = cup.periodogram(lc, "TLS")
```

### MHAOV — multiharmonic analysis of variance

A least-squares F-statistic against a trigonometric polynomial of order `H`. Excellent for
**sharply non-sinusoidal** signals (RR Lyrae, Cepheids) where a single sinusoid
under-fits. Supports {doc}`multi-band <multiband>`. Key setting
({class}`~cuperiod.MHAOVSettings`): `n_harmonics` (the order `H`).

```python
pg = cup.periodogram(lc, "MHAOV", settings=cup.MHAOVSettings(n_harmonics=4))
```

Every backend fits mean-centred values (float32 — torch, Apple MPS — used to return an
all-zero spectrum for magnitudes). On uniform frequency grids the `numpy` backend, and
`numba` for curves above 64 points per numba thread, evaluate the harmonic sums with
type-1 NUFFTs (finufft, all cores for large transforms, like GLS); the values agree with
the direct sums to ~1e-11. `min_detections` and the harmonic-order limit are reported
separately when a light curve is too short.

### PDM — phase dispersion minimization

Stellingwerf's method: bin the folded light curve and minimize the within-bin variance
relative to the total. Makes **no assumption about the waveform**, so it suits arbitrary
non-sinusoidal shapes. Key settings ({class}`~cuperiod.PDMSettings`): `n_bins`, `n_covers`.
Values are centred before binning on every backend (the float32 path lost the within-bin
variance to cancellation).
Supports {doc}`multi-band <multiband>`: each band is folded and binned on its own and the
`Theta` values are pooled by within-bin degrees of freedom.

```python
pg = cup.periodogram(lc, "PDM")
```

### CE — conditional entropy

Graham et al. (2013): minimize the conditional entropy of the folded phase–magnitude
diagram. Robust on **sparse, unevenly sampled survey data**. Key settings
({class}`~cuperiod.CESettings`): `n_phase_bins`, `n_mag_bins`. Supports
{doc}`multi-band <multiband>`: one histogram per band, entropies pooled by point count.

```python
pg = cup.periodogram(lc, "CE")
```

### String-Length

Dworetsky / Lafler–Kinman: minimize the total length of the "string" connecting
phase-ordered points in the folded curve. Cheap and shape-agnostic; handy for eclipsing or
eccentric systems. Settings: {class}`~cuperiod.StringLengthSettings`. Supports
{doc}`multi-band <multiband>`: one string per band, lengths pooled by point count.

```python
pg = cup.periodogram(lc, "String-Length")     # or "StringLength"
```

(`String-Length` is the canonical name; lookup is case-insensitive and tolerant of the
hyphen.)

The low-level kernels of the fold statistics (`pdm_theta`, `conditional_entropy`,
`string_length`, `aov_power`) return `NaN` at trial values that cannot be folded (a period
of 0 or `NaN`, a non-finite frequency) and clean values elsewhere. PDM, CE, String-Length
and MHAOV do not use error bars, but an error column of placeholders (all 0, −1 or `NaN`)
is still reported as the reason a curve is empty.

### SuperSmoother

Friedman's (1984) variable-span smoother applied to every phase-fold: three local-linear
smooths over span fractions 0.05 / 0.2 / 0.5 of the points, leave-one-out cross-validation
to pick the best span at each phase point, and a final pass over the blended curve. The
statistic follows gatspy's `SuperSmoother` — `1 - mean|y - model|/dy / mean|y - mu|/dy`,
the fractional reduction in mean absolute (error-standardized) deviation about the
inverse-variance weighted mean `mu`. It is **maximized**: `1` is a perfect fit, `0` is no
better than a constant, and a slightly negative value means the fold fits worse than the
mean.

Being fully non-parametric, it captures **any repeating shape** — an RR Lyrae sawtooth, an
eclipsing binary, a fold nothing analytic describes — without your choosing a harmonic
budget. It is the classic period finder of the Stripe 82 RR Lyrae work (Sesar et al.) and
one of the methods compared by VanderPlas & Ivezić (2015). The price is cost (a sort plus
several smooths per trial period) and a soft spectrum.

**Integer multiples score nearly as high.** A fold at `2P`, `3P`, … is still a coherent
repeating curve — it just draws the shape twice — so the smoother fits it about as well as
`P` itself. Read the *shortest* period of a high-scoring family as the candidate, bound
the trial periods from above by raising `minimum_frequency` (the longest period searched
is `1/minimum_frequency`), or let {func}`~cuperiod.alias_diagnostics` arbitrate.

```python
pg = cup.periodogram(lc, "SuperSmoother")     # or "super-smoother"
```

Key settings ({class}`~cuperiod.SuperSmootherSettings`): `primary_spans` (the candidate
span fractions, default `(0.05, 0.2, 0.5)`), `middle_span`, `final_span`,
`bass_enhancement` (Friedman's alpha, `0`–`10`, pulls the chosen spans toward the largest;
`None` disables it), `min_detections` (20 here). Runs on `numpy`, the multicore `numba`
tier (the CPU default with `[fast]`), `cupy`, and the portable `torch` path, with the same
statistic on all four ({doc}`backends`) — tied phases (duplicate timestamps, trial periods
commensurate with the cadence) are broken identically on every backend and independently
of the input row order. A constant light curve scores 0 at every period. Supports
{doc}`multi-band <multiband>`: each band is smoothed independently on the shared grid and
the per-band scores are combined with baseline-error weights, exactly as in gatspy's
`SuperSmootherMultiband`.

(several-methods)=
## Running several at once

Pass a list to compare methods on one light curve. You get a
{class}`~cuperiod.MultiResult` keyed by method name:

```python
res = cup.periodogram(lc, ["GLS", "PDM", "MHAOV"])
for name, peaks in res.best_periods(3).items():
    print(name, [round(p.period, 4) for p in peaks])
```

## Listing what's installed

```python
for info in cup.list_methods():
    print(info.name, info.objective_sense, info.supports_multiband, info.available_backends)
```

Or from the CLI: `cuperiod methods`.

---

Next: {doc}`results` — reading peaks and the raw spectrum.
