# Choosing a method

cuPeriod ships seven period-search methods. They all take the same inputs and return the
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
```

Not sure? Run a few at once and compare — see {ref}`several-methods` below.

## Objective sense: bigger vs smaller is better

Each method's statistic is either **maximized** or **minimized** at the true period:

- **Maximized** (a tall peak = significant): GLS, BLS, MHAOV, TLS.
- **Minimized** (a deep trough = significant): PDM, CE, String-Length.

You don't have to track this — {meth}`~cuperiod.Periodogram.best_periods` knows each
method's `objective_sense` and always returns the *most significant* periods first. It
matters only if you inspect the raw `power` array yourself ({doc}`results`).

## The methods in detail

### GLS — generalized Lomb–Scargle

The workhorse for periodic variable stars. cuPeriod computes the floating-mean
Lomb–Scargle power of Zechmeister & Kürster (2009) — the same statistic as astropy's
`LombScargle(..., fit_mean=True)` — but evaluates the trigonometric sums with a
non-uniform FFT, matching astropy to ~1e-9 while running much faster. Each peak carries a
false-alarm probability (`extra["fap"]`). Supports {doc}`multi-band <multiband>`.

```python
pg = cup.periodogram(lc, "GLS")
```

Key settings ({class}`~cuperiod.GLSSettings`): `samples_per_peak`, `nyquist_factor`,
`fit_mean`, `fap_method`, `minimum_frequency` / `maximum_frequency`.

### BLS — box least squares

For eclipsing binaries and box-shaped transits. Searches log-spaced period segments so the
box-duration grid tracks the period, exactly as a transit's duration scales. **Works in
flux** (magnitudes are converted automatically), because an eclipse is a dip. Each peak
carries the box parameters: `depth`, `duration`, `transit_time` (t0), `depth_snr`, and the
signal-detection efficiency `sde`. Supports {doc}`multi-band <multiband>`.

```python
pg = cup.periodogram(lc, "BLS")
for peak in pg.best_periods(5, alias_diverse=True):     # alias-aware for box searches
    print(peak.period, peak.extra["depth"], peak.extra["duration"])
```

With the `[fast]` extra, BLS's CPU backend is a multicore `numba` search ~20× faster than
astropy. Key settings ({class}`~cuperiod.BLSSettings`): `min_period_days` /
`max_period_days`, `duration_min_frac` / `duration_max_frac`, `n_durations`, `objective`.

### TLS — transit least squares

A matched filter against a **limb-darkened transit** template (not a box), more sensitive
than BLS on real exoplanet transits. Works in flux. Key settings
({class}`~cuperiod.TLSSettings`): `min_period_days` / `max_period_days`, `n_phase_bins`,
`limb_dark_u1` / `limb_dark_u2`, the duration fractions.

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

### PDM — phase dispersion minimization

Stellingwerf's method: bin the folded light curve and minimize the within-bin variance
relative to the total. Makes **no assumption about the waveform**, so it suits arbitrary
non-sinusoidal shapes. Key settings ({class}`~cuperiod.PDMSettings`): `n_bins`, `n_covers`.

```python
pg = cup.periodogram(lc, "PDM")
```

### CE — conditional entropy

Graham et al. (2013): minimize the conditional entropy of the folded phase–magnitude
diagram. Robust on **sparse, unevenly sampled survey data**. Key settings
({class}`~cuperiod.CESettings`): `n_phase_bins`, `n_mag_bins`.

```python
pg = cup.periodogram(lc, "CE")
```

### String-Length

Dworetsky / Lafler–Kinman: minimize the total length of the "string" connecting
phase-ordered points in the folded curve. Cheap and shape-agnostic; handy for eclipsing or
eccentric systems. Settings: {class}`~cuperiod.StringLengthSettings`.

```python
pg = cup.periodogram(lc, "String-Length")     # or "StringLength"
```

(`String-Length` is the canonical name; lookup is case-insensitive and tolerant of the
hyphen.)

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
