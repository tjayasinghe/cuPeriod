# Multi-band light curves

A survey rarely spends its epochs on one filter. Rubin/LSST spreads its visits across
*ugrizy*, so no single band is densely sampled in the early years; a cross-matched
ZTF + ATLAS + ASAS-SN light curve is three sparse curves rather than one dense one.
Searching each band on its own throws away the fact that they are the **same star with the
same period** — and in the sparse regime that is most of the information there is.

A multi-band search fits all bands jointly at every trial frequency: one period, but each
band keeps its own mean magnitude, amplitude, and (depending on the model) its own phase.
Seven of the eight methods do it — **GLS, BLS, MHAOV, PDM, CE, String-Length, and
SuperSmoother**; only TLS is single-band. Asking a single-band method for a multi-band run
raises a clear error.

:::{note}
How much this buys you depends on how sparse the data are. At 30 total epochs across six
bands, single-band GLS recovers nothing and the shared-phase multi-band model recovers
82% — see {ref}`the recovery table <how-much-does-it-help>` below.
:::

## Building a `MultiBandLightCurve`

From a `{band: LightCurve}` mapping:

```python
from cuperiod import LightCurve, MultiBandLightCurve

lc_g = LightCurve.from_arrays(t_g, mag_g, err_g)
lc_r = LightCurve.from_arrays(t_r, mag_r, err_r)

mb = MultiBandLightCurve.from_light_curves({"g": lc_g, "r": lc_r})
print(mb.band_names, mb.n_bands)       # ('g', 'r'), 2
```

From a **long-format** table — one row per epoch, with a band/filter column:

```python
mb = MultiBandLightCurve.from_dataframe(df, band_column="filter")
```

Or straight from a long-format **file** (CSV / ECSV / FITS / Parquet — no pandas needed):

```python
mb = MultiBandLightCurve.from_file("star_ugrizy.parquet", band_column="band")
pg = cup.periodogram(mb, "GLS")
```

The band column is auto-detected (`band`, `filter`, `passband`, `fid`, …) when you don't
name it, and the time/value/error columns resolve exactly as for a single
{doc}`light curve <light-curves>` (with {class}`~cuperiod.ColumnMap` if needed). Bands keep
their first-appearance order.

The same switch works from the command line and in batch:

```bash
cuperiod run star_ugrizy.csv --band filter --method GLS
```

```python
cup.batch_periodograms("survey/*.parquet", "GLS", band_column="band", sink="out/")
```

`--band` (and `band_column=` for file, glob, directory, or `(DataFrame, group_column)`
inputs) makes each object a joint multi-band fit rather than a single-band one.

## Running a multi-band periodogram

Pass the {class}`~cuperiod.MultiBandLightCurve` to {func}`~cuperiod.periodogram` like any
other input:

```python
pg = cup.periodogram(mb, "GLS")        # shared-phase model, the default
print(pg.best_period())
print(pg.meta["bands"], pg.meta["mb_model"])
```

The result is an ordinary {class}`~cuperiod.Periodogram`, so peak finding, the raw
spectrum, and serialization all work as in {doc}`results`.

The default grid is built from the **stacked** baseline of all bands (their combined time
span), so the frequency resolution reflects all your data; override it with `grid=`
({doc}`tuning`).

## The three GLS models

`GLSSettings.mb_model` selects how the bands share the signal. All three run natively on
every backend — finufft on the CPU, cufinufft on CUDA, and the portable torch path on any
device. (`backend="astropy"` still delegates to `LombScargleMultiband`; it is kept as the
reference the native paths are tested against, not as a fast path.)

### `"offsets"` — shared phase (the default)

One sinusoid on a **shared phase** plus an independent constant offset per band: the
`(N_base, N_band) = (1, 0)` model of VanderPlas & Ivezić (2015), and their recommended
search model for sparse multi-band data.

```{math}
y_k(t) = a\cos(2\pi f t) + b\sin(2\pi f t) + c_k
```

All phase information pools into the two signal parameters `a, b` while each band `k`
spends only one nuisance parameter `c_k`. cuPeriod profiles the offsets out analytically,
which leaves the ordinary Zechmeister & Kürster assembly with every trigonometric sum
replaced by its band-centered counterpart, and evaluates it in `K + 2` NUFFTs for `K`
bands — so a joint fit costs about what one single-band GLS over the stacked points would.
With `K = 1` it reduces exactly to single-band GLS.

That closed form is also the speed story. On a six-band, 100-point star over 200 000 trial
frequencies, delegating to astropy's `LombScargleMultiband` took **55 s**; the native path
takes **0.13 s** — roughly 400×.

### `"perband"` — independent per-band sinusoids

The multi-phase `(0, 1)` model: each band gets its own floating-mean sinusoid, with an
independent amplitude *and phase*, and the per-band standard powers are combined with the
reference-chi-squared weights of VanderPlas & Ivezić (2015, eq. 23):

```{math}
P(f) = \frac{\sum_k \chi^2_{0,k}\, P_k(f)}{\sum_k \chi^2_{0,k}}
```

Nothing ties the bands' phases together, so this is the model to reach for when you want a
quick per-band look with one combined statistic — but it is also why it recovers fewer
sparse-cadence periods than `"offsets"` (below).

:::{note}
astropy's `LombScargleMultiband(method="fast")` intends this same combination but weighs
the bands by the summed squared periodogram rather than by `chi2_0k`, which makes its
output depend on the frequency grid it was evaluated on. This differs from the published
weighting; `gatspy` uses `chi2_0k`, and so does cuPeriod.
:::

### `"flex"` — flexible regularized model

astropy's flexible model: `mb_nterms_base` shared harmonics plus `mb_nterms_band`
harmonics-with-offset per band, ridge-regularized to lift the base/band degeneracy.
Per-band chromatic light-curve *shape* — an amplitude ratio and a phase lag that differ
from filter to filter, as in an RR Lyrae or a Cepheid — is what the extra per-band
harmonics buy.

```python
settings = cup.GLSSettings(mb_model="flex", mb_nterms_base=2, mb_nterms_band=1)
pg = cup.periodogram(mb, "GLS", settings=settings)
```

cuPeriod builds the per-frequency normal equations from per-band harmonic trig sums and
solves them in batch, on CPU, CUDA, or any torch device. It reproduces astropy's power to
~2e-10 across term counts and under both ridge conventions
(`mb_regularize_by_trace`) — see `tests/test_multiband_gls.py`.

| Setting | Default | What it does |
| --- | --- | --- |
| `mb_model` | `"offsets"` | `"offsets"` / `"perband"` / `"flex"` |
| `mb_nterms_base` | `1` | flex: shared (base) harmonic terms |
| `mb_nterms_band` | `1` | flex: per-band harmonic terms |
| `mb_reg_base` | `None` | flex: ridge on the base columns (`None` = 0) |
| `mb_reg_band` | `1e-6` | flex: ridge on the per-band columns |
| `mb_regularize_by_trace` | `True` | flex: scale the ridge by the normal-matrix trace |
| `mb_fap_bootstrap` | `0` | within-band bootstrap resamples (0 disables) |
| `mb_fap_seed` | `0` | seed for that bootstrap |

(how-much-does-it-help)=
## Which model, and how much does it help?

`benchmarks/multiband_recovery.py` simulates faint RRab-like stars on a Rubin-like
six-band cadence — a 3-year span, WFD epoch shares (*r*/*i* deepest), 0.20 mag per-point
noise (about an *r* ≈ 23 halo RR Lyrae in single visits) — and sweeps the *total* number
of epochs across all six bands. Recovery means the periodogram's top period is within 1%
of the truth, with no harmonic credit; 300 stars per cell.

| strategy | 30 epochs | 60 epochs | 120 epochs |
| --- | --- | --- | --- |
| best single band (*r*) | 0.0% | 37.3% | 97.3% |
| any single band | 0.0% | 49.3% | 99.0% |
| multi-band `perband` (0,1) | 17.7% | 97.0% | 100.0% |
| multi-band `flex` (1,1) | 20.0% | 97.0% | 100.0% |
| multi-band `offsets` (1,0) | **81.7%** | **99.7%** | 100.0% |

*"any single band"* counts a star as recovered if **any** of the six per-band searches
lands on the right period — an optimistic upper bound on per-band searching, since in
practice you don't know which band was right.

:::{warning}
These are simulations, on a deliberately simplified cadence: epochs land on random nights
with no rolling cadence and no lunation weighting. Read the *ordering*, not the absolute
numbers. What it shows is that **spending parameters on a shared phase** is what buys
sparse-cadence recovery: the models that give each band its own phase freedom (`perband`,
and `flex` through its per-band harmonic) sit far below the shared-phase model at 30
epochs, and only catch up once each band is close to separately solvable.
That ordering is consistent with Rubin's own alert-production study and with VanderPlas &
Ivezić (2015), and it is why `"offsets"` is the default.
:::

Practical guidance:

- **`"offsets"`** — the default, and the right first choice for a period *search*,
  especially when bands are sparse.
- **`"flex"`** — when the fold shape is genuinely chromatic and you want the model to say
  so; also the model to use if you want to compare against astropy's flexible method.
- **`"perband"`** — a decoupled quick look, or when you suspect the bands are not
  phase-coherent (e.g. blended sources, or photometry from instruments you don't trust to
  share a time system).

## The fold-based methods

PDM, conditional entropy, String-Length, and SuperSmoother join the existing MHAOV (pooled
`F`) and BLS (shared ephemeris, stacked depth-SNR) with the same structure: **each band
keeps its own mean curve, histogram, or normalization**, and only the resulting statistics
are pooled.

```{math}
S_{\rm mb}(f) = \frac{\sum_k w_k\, S_k(f)}{\sum_k w_k}
```

Forcing the bands onto one common curve would be wrong: filters differ in mean magnitude
and amplitude, so a joint fold would be smeared by the band offsets alone and would look
disordered at *every* trial period. The per-band statistics are already scale-free — PDM's
`Theta` divides by that band's own variance, CE rescales magnitudes to the band's range,
String-Length rescales to Dworetsky's span, SuperSmoother divides by that band's own mean
absolute deviation — so pooling needs no separate standardization step.

| Method | Pooled quantity | Weight `w_k` |
| --- | --- | --- |
| **PDM** | Stellingwerf `Theta_k` | `max(n_k - n_bins, 1)` (within-bin degrees of freedom) |
| **CE** | conditional entropy `H_k` | `n_k` (makes `H_mb` the entropy per observation) |
| **String-Length** | string length `L_k` | `n_k` (a string over `n_k` points is `n_k` steps) |
| **SuperSmoother** | smoother score `S_k` | `B_k`, band `k`'s baseline error (its mean absolute standardized deviation about its own mean) |
| **MHAOV** | pooled `F`-statistic | — |
| **BLS** | stacked depth-SNR, shared ephemeris | — |

```python
pg = cup.periodogram(mb, "PDM")            # dof-weighted mean of the per-band Theta
pg = cup.periodogram(mb, "CE")             # point-count-weighted mean entropy
pg = cup.periodogram(mb, "String-Length")  # point-count-weighted mean length
pg = cup.periodogram(mb, "SuperSmoother")  # baseline-error-weighted mean score
pg = cup.periodogram(mb, "MHAOV")          # pooled F-statistic across bands
pg = cup.periodogram(mb, "BLS")            # joint box search across bands
```

The pooled PDM, CE, and String-Length statistics are **minimized** at the true period and
SuperSmoother's is **maximized**, exactly as their single-band counterparts are;
{meth}`~cuperiod.Periodogram.best_periods` handles the sense for you ({doc}`results`). A
band too sparse to be searched alone is skipped rather than fatal (the thresholds are
`n_bins + 2` points for PDM, `max(n_phase_bins, 8)` for CE, 8 for String-Length, and 3 —
the smallest span window — for SuperSmoother).

SuperSmoother's weights are gatspy's `SuperSmootherMultiband`: `B_k` is the denominator of
band `k`'s own score, so the combined statistic is the *total* fractional reduction in mean
absolute deviation across all bands, a noisy or flat band contributes little weight, and
with one band it collapses to the single-band score.

:::{note}
SuperSmoother pools **scores, not phases** — being non-parametric there is no shared-phase
model to fit, so each band's fold is smoothed on its own. For sparse Rubin-cadence data,
where no single band is separately solvable, the GLS `"offsets"` model above remains the
right search tool; multi-band SuperSmoother is for characterizing an arbitrary fold shape
when the individual bands are already decent.
:::

## False-alarm probabilities

astropy's `LombScargleMultiband` has no false-alarm probabilities at all — its FAP methods
raise `NotImplementedError`, because the single-band analytic formulas assume one sinusoid
fit to one band. cuPeriod calibrates the multi-band periodogram by **within-band
bootstrap** instead.

Under the null hypothesis of no coherent signal, each band's `(value, error)` *pairs* are
exchangeable across that band's epochs, so they are resampled with replacement while every
observation **time stays fixed**. That preserves the window function, the per-band sample
sizes, and the heteroskedastic error distribution while destroying phase coherence. The
maximum power over the searched grid is recorded for each resample, and an observed peak's
false-alarm probability is its rank in that null sample:

```{math}
{\rm FAP}(z) = \frac{1 + \#\{\max_r \ge z\}}{R + 1}
```

Two ways to get it. Standalone, when you want the calibration object:

```python
calib = cup.multiband_fap(mb, n_bootstrap=1000, seed=0)
pg = cup.periodogram(mb, "GLS")

print(calib.fap(pg.best_periods(1)[0].power))    # FAP of the observed peak
print(calib.level(0.01))                         # 1% false-alarm power threshold
```

Or attached to the run, which annotates the spectrum's peaks directly:

```python
settings = cup.GLSSettings(mb_fap_bootstrap=500)
pg = cup.periodogram(mb, "GLS", settings=settings)

top = pg.best_periods(1)[0]
print(top.period, top.extra["fap"])
print(pg.meta["fap_level_10pct"], pg.meta["fap_level_1pct"])
```

:::{note}
The smallest resolvable false-alarm probability is `1 / (n_bootstrap + 1)`; asking
{meth}`~cuperiod.MultibandFAP.level` for anything below it raises rather than
extrapolating. `meta["fap_level_1pct"]` appears only from `n_bootstrap >= 99`. Because the
level of a *maximum* depends on the grid that maximum was taken over, calibrate on the
same grid you searched (the default does) — and with the same `mb_model`, which
`multiband_fap` takes as its second argument.
:::

For the `"offsets"` model on a NUFFT backend the bootstrap is nearly free: the times never
change, so every resample reuses the same nonuniform points and the whole batch runs as
multi-transform NUFFTs — the same `K + 2` transforms as one power evaluation, each
carrying a stack of bootstrap strengths. The `"perband"` and `"flex"` models and the torch
backend fall back to an explicit loop over resamples, which is still native speed per
iteration but scales linearly in `n_bootstrap`.

`multiband_fap` also accepts a single-band {class}`~cuperiod.LightCurve`, which gives an
honest bootstrap FAP for the ordinary GLS.

## Is the peak an alias?

Multi-band data pool several observing cadences, and the combined spectral window can
still have a strong daily or yearly comb. {func}`~cuperiod.alias_diagnostics` measures the
window of *this* sampling, predicts where it would place competing peaks, and reports how
well the periodogram actually supports each one:

```python
report = cup.alias_diagnostics(pg, mb)
print(report.summary())
if report.ambiguous:
    print("a non-harmonic competitor is nearly as good — quote with a caveat")
```

The bands are stacked into one sampling for the window measurement. Harmonics and
subharmonics are reported but never make a result `ambiguous`, since `2f` is expected
structure in any non-sinusoidal signal. Works for both objective senses, so the pooled PDM
/ CE / String-Length / SuperSmoother periodograms can be checked the same way. See
{class}`~cuperiod.AliasReport` for the fields.

---

Next: {doc}`interop` — running these searches over LINCC/lsdb survey catalogs — or
{doc}`batch` for scaling to many light curves.
