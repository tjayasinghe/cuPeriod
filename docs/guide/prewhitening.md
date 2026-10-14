# Pre-whitening pulsators

A periodogram answers *"is there a period?"*. A multiperiodic pulsator needs a different
question answered: *"which frequencies are really there, how well do I know them, and
when should I stop looking?"* That is **pre-whitening** — find the strongest peak, fit a
sinusoid, subtract it, look again — and it is traditionally an interactive Period04
session, one star at a time, with the stopping decision left to the operator's eye.

{func}`cuperiod.prewhiten` automates the loop end to end and makes every judgement call
an explicit, recorded setting. When a light curve needs a hand on the wheel — a peak
taken by hand, a tied combination, deleted outliers, nightly zero points — the same loop
runs one action at a time in a {class}`~cuperiod.PreWhitenSession`, in a script or in the
desktop app, with undo, project files and reproducing scripts: see
{doc}`interactive-prewhitening`. `cup.prewhiten(...)` *is* such a session run to its
stopping rule and finalized, so the two routes give the same numbers.

```python
import cuperiod as cup

solution = cup.prewhiten((time, mag, mag_err))
print(solution.summary())
```

```text
Pre-whitening: 3 components from 2500 points over 26.96 d  (backend=cufinufft)
  stopped: S/N 2.88 < 4
  residual rms 0.00148   reduced chi2 0.984   D=1.00   errors: covariance

  ID     frequency (1/d)         +/-     amplitude         +/-    phase     S/N   A/Asp  note
  F1          12.3400341    7.16e-05     0.0119791    4.19e-05   6.1123  235.77    0.99
  F2          17.8099479    0.000122    0.00697566    4.25e-05   0.4786  143.80    1.00
  F3          24.6799702    0.000214    0.00399532    4.21e-05   5.9160   75.66    1.11  F3 = 2F1
```

Everything the run decided is in the object: the components with their uncertainties, the
residuals and their spectrum, the fit statistics, and — crucially — **why it stopped**.

## What one iteration does

1. **Amplitude spectrum of the current residuals.** Not power: pulsation work is done in
   millimagnitudes, and the signal-to-noise criterion below is defined on amplitudes. The
   default is the weighted least-squares amplitude, evaluated through the same NUFFT
   machinery as {doc}`GLS <methods>` (see {doc}`backends`). Where the sampling barely
   separates a sinusoid from the constant — near multiples of a regular cadence, near
   1, 2, … c/d for nightly sampling, towards zero frequency — it is replaced by the
   bounded amplitude of the variance the sinusoid explains (zero where the sampling
   carries no information), so such frequencies cannot pose as giant components.
2. **Pick the tallest peak** that is resolved from everything already extracted — at
   least `min_separation_rayleigh` (1.5 by default, after Loumos & Deeming 1978) Rayleigh
   widths away — and locate its apex by parabolic interpolation.
3. **Re-solve the whole model**: every amplitude, phase and the offset jointly, plus a
   non-linear refinement of the *new* frequency (`refine`, `"last"` by default — the
   established frequencies are swept again in the final polish).
4. **Test the new component** against the stopping criteria. If it fails, the run stops
   and the component is discarded.

Only step 1's *data-dependent* part is recomputed each iteration: the terms that depend
solely on the observation times are cached, so a fifty-frequency solution costs about
fifty-two transforms rather than a hundred and fifty. A 20 000-point TESS sector with
fifteen modes takes about a second on a laptop CPU.

## Stopping criteria

`stop_criteria` lists the tests a component must pass; failing any one ends the run.

| Criterion | Setting | Meaning |
| --- | --- | --- |
| `"snr"` *(default)* | `snr_threshold` (4.0) | Breger et al. (1993): amplitude over the mean amplitude of the residual spectrum in a ±`snr_window` c/d box. |
| `"fap"` | `fap_threshold` (1e-3) | Baluev false-alarm probability of the peak in the spectrum it was drawn from. |
| `"bic"` | `min_delta_bic` (10.0) | The component must improve the Bayesian information criterion by at least this much. |
| `"amplitude"` | `min_amplitude` | A hard amplitude floor, in the units of the input. |

```python
settings = cup.PreWhitenSettings(stop_criteria=("snr", "bic"), snr_threshold=4.6)
solution = cup.prewhiten(lc, settings=settings)
print(solution.stop_reason)      # e.g. "S/N 3.42 < 4.6"
```

:::{admonition} How pure do you need the list to be?
:class: tip
The classical S/N ≥ 4 rule is a convention, not a false-alarm guarantee: on pure noise it
still admits roughly 0.2 spurious frequencies per light curve. For catalogue work, raise
`snr_threshold` to ~4.6 (Baran & Koen 2021 for TESS-like data) or add `"fap"` / `"bic"`
to `stop_criteria`, which control the false-alarm rate directly.
:::

Two stops are always on, whatever `stop_criteria` says:

- **degrees of freedom** — the run stops when another component would leave fewer than
  half the points as degrees of freedom (`2(3K + 1) > N` for K components), because past
  that a sparse light curve interpolates its own noise with ever more sinusoids
  (`stop_reason` *too few points (N) to constrain another component (M parameters)*);
- **numerical round-off** — a noise-free or constant light curve stops instead of
  fitting rounding error (*residuals are at numerical round-off*).

The run also ends at `max_frequencies` (*reached max_frequencies (30)*). On sparse light
curves — below about a hundred points — the S/N criterion alone is unreliable, because
the residual spectrum it is measured against is itself thinned by every fit; add `"fap"`
to `stop_criteria` there.

The false-alarm probability is evaluated natively from Baluev's (2008) closed form —
matching astropy's `false_alarm_probability(method="baluev")` while staying accurate for
raw Julian dates — and every component's value is reported whether or not `"fap"` is a
stopping criterion. It is also available directly as {func}`~cuperiod.baluev_fap`.

After the loop, the accepted solution is polished with one simultaneous fit of all
frequencies and then **re-checked**: the joint fit redistributes power between close
components, so a frequency that cleared the threshold when it was extracted can end up
insignificant. Those are dropped and the solution re-fitted (`prune`, on by default; the
re-check is the S/N test, so it runs only when `"snr"` is among `stop_criteria`), and the
count appears in `n_pruned` and in `stop_reason`.

## Uncertainties

Reported errors are 1-sigma and come from one of three estimators, set by `uncertainty`:

`"covariance"` *(default)*
: The linearised least-squares covariance of the joint fit. The only one of the three
  that accounts for correlations *between* components, which matters as soon as two
  frequencies sit within a few Rayleigh widths of each other.

`"analytic"`
: The closed-form expressions of Montgomery & O'Donoghue (1999) — the numbers most
  pulsation papers quote — with the weighted residual rms for weighted fits.

`"bootstrap"`
: A wild (sign-flip) residual bootstrap: each of `n_resamples` replicates flips the
  sign of every residual at random (so heteroscedastic error bars are respected),
  scaled by the degrees-of-freedom correction `sqrt(N/(N − M))`, and is refined like
  the final polish (simultaneously, or with at least three cyclic sweeps), boxed by the
  same per-frequency bounds as the fit it characterises. No linearity assumption, at the
  price of that many extra fits; seeded values differ from cuPeriod 1.2's.

The covariance and analytic errors are inflated by `sqrt(D)` with `D` the
Schwarzenberg-Czerny (1991) correlation factor (`correlation_correction`, on by default),
because real photometry has residuals that are correlated point to point and the formal
errors are correspondingly optimistic. The bootstrap is left alone — it already resamples
the residuals as they are. `D = 1` means the residuals look white; it is computed from
the residuals in time order, so the row order of the input does not matter. The
`offset_error` goes through the same estimator and the same `sqrt(D)` inflation.

The covariance is the pseudo-inverse of the Jacobi-scaled normal matrix, built about the
weighted-mean epoch, so it stays accurate on long baselines, in flux units and far from
the epoch; a parameter the data cannot determine (for example of a zero-amplitude
component) reports `NaN` rather than `0`.

With `fit_mean=False` the offset is not a free parameter: the weighted mean — the same
zero point the amplitude spectrum uses — is removed once up front and reported as
`offset`, with no error (`NaN`).

Phases are referenced to `solution.t_ref`, the **weighted mean of the observation times**.
That epoch is not arbitrary: it is the one at which a phase is uncorrelated with its own
frequency, so the reported phase uncertainty is the smallest — and the most meaningful —
one available. Referencing to the first observation instead would inflate it by a factor
of a few for no gain.

## Combination frequencies

A non-linear pulsator's spectrum is not a list of independent modes: harmonics `2f₁`,
sums `f₁ + f₂` and differences `f₁ − f₂` are everywhere, and counting them as modes is a
classic way to over-count a δ Scuti star's mode density.

```python
for c in solution.components:
    print(c.label, c.frequency, c.combination)
# F1 12.3400341 None
# F2 17.8099479 None
# F3 24.6799702 F3 = 2F1

modes = solution.independent()   # the candidate independent-mode list
```

A peak is only ever explained by frequencies *stronger* than itself, and the match must
fall inside `max(3σ, 0.25/T)` where σ is the **propagated** uncertainty of the predicted
combination. Combinations have order Σ|nᵢ| ≥ 2 (a component is never "identified" as
another one), up to `combination_max_order`; `combination_max_order=1` switches the
search off. Each identification also carries `expected_false`, the rate at which a
random frequency in the band of the extracted frequencies would match one of the
coefficient vectors searched — if that approaches 1, the identification means nothing,
and you should know that without having to work it out. To make an identified
combination *follow* its parents in the fit, tie it in a session
({doc}`interactive-prewhitening`).

## Is an amplitude trustworthy on its own?

Every fitted amplitude comes from the *joint* solution of all components at once, while
the amplitude spectrum reads each frequency as if it were alone. For a mode resolved
from its neighbours the two agree, and each component records both so you can check:

```python
for c in solution.components:
    print(c.label, c.amplitude, c.spectrum_amplitude, c.amplitude_ratio, c.blended)

solution.n_blended            # how many disagree beyond blend_tolerance (default 2x)
```

A ratio far from 1 means the amplitude is **entangled** with a component it is
correlated with: change one and the other moves. Close pairs do this, and so — far more
often in ground-based data — does a mode sitting beside its own alias sidelobe. On the
bundled ASAS-SN HADS demo every harmonic carries yearly aliases at
Δf = 1/365.25 d, and the 3f component is fitted at nearly twice what the data holds
there.

:::{admonition} A blend flag is not a significance test
:class: caution
A blended component can be perfectly real — an alias sidelobe *is* present in the data,
and the flag never removes anything from the solution. It says the amplitude means
something only alongside the components it is correlated with, so quote them together
rather than treating the number as an independent measurement. Use `snr`, `fap` and
`delta_bic` to decide whether a component is real; use `amplitude_ratio` to decide
whether to trust its amplitude.
:::

`blend_tolerance` (default 2.0) sets the factor. Two is deliberately loose: the ratio
carries the noise of both measurements, roughly `sqrt(2)/(S/N)` in relative terms, so a
factor of two is a ≳3σ statement even for a component that only just cleared S/N ≥ 4.

## The spectral window

Irregular sampling convolves every real peak with the **spectral window**
`W(f) = Σ wⱼ exp(2πi f tⱼ)` of the observation times, so a candidate sitting where a
stronger component's window has a lobe — classically at ±1 c/d for single-site
ground-based data — deserves suspicion before it is called a mode.

```python
solution.window                 # AmplitudeSpectrum of |W(f)|, kept with the spectra

grid = cup.uniform_frequency_grid(t.max() - t.min(), maximum_frequency=5.0)
window = cup.spectral_window(t, dy, grid=grid)   # standalone, no brightness needed
```

`|W(f)|` is dimensionless with `|W| → 1` towards zero frequency. To vet a doubtful pair,
compare the residual spectrum around the weaker peak with the window displaced to the
stronger frequency: if the peak reproduces a window lobe in position *and* relative
height, it is the sampling talking. The GUI's spectrum view has a *window* toggle that
overlays it scaled to the tallest peak (the Period04 convention), and
`cuperiod prewhiten --save-spectrum` writes it into the `.npz` alongside the spectra.

Keeping the window costs nothing: its sums are already part of the cached normal
equations that make each pre-whitening iteration a single NUFFT.

{meth}`AmplitudeSpectrum.noise_curve() <cuperiod.AmplitudeSpectrum.noise_curve>` gives
the local noise level (the S/N criterion's denominator) at every grid frequency; times
`snr_threshold` it is the acceptance limit the GUI draws over the residual spectrum.

## g-mode period spacings

For γ Dor and SPB stars the next question is the period spacing, whose mean value fixes
the buoyancy travel time and whose slope traces near-core rotation.

```python
series = cup.find_period_spacing(
    [c.period for c in solution.independent()],
    [c.amplitude for c in solution.independent()],
)
print(series.summary())
# Period spacing: 16 modes, <dP> = 2962.1 s (0.0342835 d), slope +0.008011,
#                 rms 1.9 s, Pi_0(l=1) = 4189 s
```

{func}`~cuperiod.spacing_spectrum` scans trial spacings with a comb response — unlike a
histogram of consecutive differences it is unaffected by missing radial orders — and
{func}`~cuperiod.find_period_spacing` then extracts the longest chain of modes following
a *tilted* pattern `ΔP(P) = a + bP`, bridging steps of up to `max_gap` radial orders (so
up to `max_gap - 1` missing modes). Use {func}`~cuperiod.echelle` for the diagnostic plot
in which a clean series is a near-vertical ridge:

```python
x, y = cup.echelle(series.periods, series.mean_spacing)
```

The same functions work on frequencies, where a regular spacing is the p-mode large
separation or a rotational splitting. By default the trial spacings come from the bulk
of the values — twice their interquartile range, widened to at least eight median gaps
and capped at the full range — so one far outlier (a slow signal among g modes) cannot
hide the comb; non-finite values are dropped together with their weights.

## Many stars at once

{func}`~cuperiod.batch_prewhiten` runs the whole pipeline over a glob, directory, or
DataFrame with the same worker machinery as {doc}`batch`, and writes **one row per
extracted component** — the shape a frequency catalogue wants.

```python
if __name__ == "__main__":      # worker pools start fresh processes on every OS
    cup.batch_prewhiten(
        "lightcurves/*.csv",
        settings=cup.PreWhitenSettings(max_frequencies=20),
        device="cpu",
        sink="modes.parquet",
    )
```

**`batch_prewhiten` forces `store_spectra=False`**: the full amplitude spectra are large
and rarely wanted a million times over.

- `backend=None` (the default) takes `settings.backend` — and so
  `CUPERIOD_PREWHITEN_BACKEND` — as `cup.prewhiten` does. With `device="cpu"`, `"auto"`
  means the best CPU backend and a GPU-only backend (`"gpu"`, `"cupy"`, `"cufinufft"`)
  raises `ValueError` before any work: CPU-pool workers never open a GPU.
- Resuming is **by key**: a file or directory sink skips the light curves already in it
  and retries the ones that failed, so adding inputs, reordering them or resuming with
  another `chunk_size` neither drops nor duplicates a star. After changing the settings,
  run again with `resume=False`.
- Rows and errors come back in input order; `chunk_size` is an upper bound (small
  batches are spread over all workers).

## From the command line

```bash
cuperiod prewhiten star.csv --snr 4.6 -n 30 --spacing --csv modes.csv
```

```bash
cuperiod batch-prewhiten "lightcurves/*.csv" --out modes.parquet --workers 8
```

`--save-project star.cupw.json` keeps the run as a project to open, edit or verify
later, and `--export-script star.py` writes a script that reproduces it
({doc}`interactive-prewhitening`). See {doc}`cli` for the full option list, and
{doc}`gui` for the desktop app, which runs the same analysis with the amplitude
spectrum, the residual overlay, an editable frequency table and a period-spacing
explorer side by side.

## Settings reference

Every field of {class}`~cuperiod.PreWhitenSettings` is documented in the
{doc}`API reference <../api/index>` and overridable from the environment as
`CUPERIOD_PREWHITEN_<FIELD>`. The ones worth knowing first:

| Setting | Default | What it controls |
| --- | --- | --- |
| `max_frequencies` | 30 | Hard cap on extracted components. |
| `snr_threshold` | 4.0 | The Breger criterion (see the note above). |
| `min_separation_rayleigh` | 1.5 | Resolution guard between components. |
| `samples_per_peak` | 10 | Frequency oversampling of the search grid. |
| `maximum_frequency` | regular cadence: 0.5/cadence; otherwise max(5 × pseudo-Nyquist, 50 /d) | Top of the search band (see [below](#the-automatic-search-band)). |
| `uncertainty` | `"covariance"` | Error estimator. |
| `blend_tolerance` | 2.0 | Factor beyond which a fitted amplitude is flagged as blended. |
| `refine` | `"last"` | Per-iteration refinement; the final polish is simultaneous. |
| `combination_max_order` | 2 | Largest `Σ|nᵢ|` in the combination search; 1 switches it off. |
| `backend` | `"auto"` | GPU when available (see {doc}`backends`). |

`downsample_points` had no effect and is deprecated: setting it warns
(`DeprecationWarning`), and it will be removed.

### The automatic search band

- On a **strictly regular cadence** — space photometry such as TESS or Kepler, with
  gaps, dropped cadences and barycentric timing allowed — the band stops at the true
  Nyquist frequency `0.5/cadence`. Above it the spectrum mirrors the baseband, every
  mode has equal-height images at `k/cadence ± f`, and the extraction would report
  whichever image the grid happens to sample best. For **super-Nyquist** work set
  `maximum_frequency` explicitly; an explicit `minimum_frequency` alone above the
  automatic maximum raises `InsufficientDataError` naming the setting to change.
- On **irregular** sampling it is five times the median-gap pseudo-Nyquist frequency,
  but at least 50 c/d. The floor matters: the median-gap pseudo-Nyquist of nightly
  ground-based sampling is a few c/d, and a band capped there sees only the *daily
  aliases* of a δ Scuti or HADS star. On multi-exposure cadences (dithered triplets,
  filter sequences) the automatic ceiling is limited to 1e5 Rayleigh widths (see
  {doc}`tuning`).
- The grid's size is checked before it is allocated: a stray timestamp (a `0` among
  Julian dates, a JD/MJD mix-up) gives a `GridSizeWarning` naming the outlying epochs,
  or a `GridTooLargeError` above 250 million samples, instead of a multi-GiB allocation.

:::{note}
Results, spectra, fits and uncertainty objects (`PreWhitenResult`, `MultiSineFit`,
`AmplitudeSpectrum`, `Uncertainties`, `SpacingSpectrum`, `PeriodSpacingSeries`) compare
and hash by identity; compare `result.components` or `to_dict()` for content.
:::

## References

- Breger, M., et al. 1993, A&A 271, 482 — the S/N ≥ 4 criterion.
- Baluev, R. V. 2008, MNRAS 385, 1279 — the false-alarm probability bound.
- Loumos, G. L., & Deeming, T. J. 1978, Ap&SS 56, 285 — frequency resolution.
- Schwarzenberg-Czerny, A. 1991, MNRAS 253, 198 — correlated residuals.
- Montgomery, M. H., & O'Donoghue, D. 1999, DSSN 13, 28 — analytic uncertainties.
- Lenz, P., & Breger, M. 2005, CoAst 146, 53 — Period04.
- Van Reeth, T., et al. 2015, ApJS 218, 27 — g-mode period-spacing patterns.
- Baran, A. S., & Koen, C. 2021, AcA 71, 113 — significance thresholds for space data.
