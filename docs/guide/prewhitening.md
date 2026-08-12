# Pre-whitening pulsators

A periodogram answers *"is there a period?"*. A multiperiodic pulsator needs a different
question answered: *"which frequencies are really there, how well do I know them, and
when should I stop looking?"* That is **pre-whitening** — find the strongest peak, fit a
sinusoid, subtract it, look again — and it is traditionally an interactive Period04
session, one star at a time, with the stopping decision left to the operator's eye.

{func}`cuperiod.prewhiten` automates the loop end to end and makes every judgement call
an explicit, recorded setting.

```python
import cuperiod as cup

solution = cup.prewhiten((time, mag, mag_err))
print(solution.summary())
```

```text
Pre-whitening: 3 components from 2500 points over 26.96 d  (backend=cufinufft)
  stopped: S/N 2.88 < 4
  residual rms 0.00148   reduced chi2 0.984   D=1.00   errors: covariance

  ID     frequency (1/d)         +/-     amplitude         +/-    phase     S/N  note
  F1          12.3400341    7.16e-05     0.0119791    4.19e-05   6.1123  235.77
  F2          17.8099479    0.000122    0.00697566    4.25e-05   0.4786  143.80
  F3          24.6799702    0.000214    0.00399532    4.21e-05   5.9160   75.66  F3 = 2F1
```

Everything the run decided is in the object: the components with their uncertainties, the
residuals and their spectrum, the fit statistics, and — crucially — **why it stopped**.

## What one iteration does

1. **Amplitude spectrum of the current residuals.** Not power: pulsation work is done in
   millimagnitudes, and the signal-to-noise criterion below is defined on amplitudes. The
   default is the weighted least-squares amplitude, evaluated through the same NUFFT
   machinery as {doc}`GLS <methods>` (see {doc}`backends`).
2. **Pick the tallest peak** that is resolved from everything already extracted — at
   least `min_separation_rayleigh` (1.5 by default, after Loumos & Deeming 1978) Rayleigh
   widths away — and locate its apex by parabolic interpolation.
3. **Re-fit every component simultaneously**: all frequencies, amplitudes, phases and the
   offset, by non-linear least squares.
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

The false-alarm probability is evaluated natively from Baluev's (2008) closed form —
matching astropy's `false_alarm_probability(method="baluev")` while staying accurate for
raw Julian dates — and every component's value is reported whether or not `"fap"` is a
stopping criterion. It is also available directly as {func}`~cuperiod.baluev_fap`.

After the loop, the accepted solution is polished with one simultaneous fit of all
frequencies and then **re-checked**: the joint fit redistributes power between close
components, so a frequency that cleared the threshold when it was extracted can end up
insignificant. Those are dropped and the solution re-fitted (`prune`, on by default), and
the count appears in `n_pruned` and in `stop_reason`.

## Uncertainties

Reported errors are 1-sigma and come from one of three estimators, set by `uncertainty`:

`"covariance"` *(default)*
: The linearised least-squares covariance of the joint fit. The only one of the three
  that accounts for correlations *between* components, which matters as soon as two
  frequencies sit within a few Rayleigh widths of each other.

`"analytic"`
: The closed-form expressions of Montgomery & O'Donoghue (1999) — the numbers most
  pulsation papers quote.

`"bootstrap"`
: Resample the residuals, re-fit `n_resamples` times, take the scatter. No linearity
  assumption, at the price of that many extra fits. Every replicate re-optimises *all*
  frequencies, boxed by the same per-frequency bounds as the fit it characterises.

All three are inflated by `sqrt(D)` with `D` the Schwarzenberg-Czerny (1991) correlation
factor (`correlation_correction`, on by default), because real photometry has residuals
that are correlated point to point and the formal errors are correspondingly optimistic.
`D = 1` means the residuals look white.

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
combination. Each identification also carries `expected_false`, the number of chance
matches expected for the coefficient vectors that were searched — if that approaches 1,
the identification means nothing, and you should know that without having to work it out.

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
a *tilted* pattern `ΔP(P) = a + bP`, bridging up to `max_gap` missing orders. Use
{func}`~cuperiod.echelle` for the diagnostic plot in which a clean series is a
near-vertical ridge:

```python
x, y = cup.echelle(series.periods, series.mean_spacing)
```

The same functions work on frequencies, where a regular spacing is the p-mode large
separation or a rotational splitting.

## Many stars at once

{func}`~cuperiod.batch_prewhiten` runs the whole pipeline over a glob, directory, or
DataFrame with the same worker machinery as {doc}`batch`, and writes **one row per
extracted component** — the shape a frequency catalogue wants.

```python
cup.batch_prewhiten(
    "lightcurves/*.csv",
    settings=cup.PreWhitenSettings(store_spectra=False, max_frequencies=20),
    device="cpu",
    sink="modes.parquet",
)
```

Set `store_spectra=False` in batch runs (the default for this entry point): the full
amplitude spectra are large and rarely wanted a million times over.

## From the command line

```bash
cuperiod prewhiten star.csv --snr 4.6 -n 30 --spacing --csv modes.csv
```

```bash
cuperiod batch-prewhiten "lightcurves/*.csv" --out modes.parquet --workers 8
```

See {doc}`cli` for the full option list, and {doc}`gui` for the interactive version —
the desktop app runs the same analysis with the amplitude spectrum, the residual overlay,
the frequency table and a period-spacing explorer side by side.

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
| `maximum_frequency` | max(pseudo-Nyquist, 50 /d) | Top of the search band. The floor matters: the median-gap pseudo-Nyquist of nightly ground-based sampling is a few c/d, and a band capped there sees only the *daily aliases* of a δ Scuti or HADS star. |
| `uncertainty` | `"covariance"` | Error estimator. |
| `refine` | `"last"` | Per-iteration refinement; the final polish is simultaneous. |
| `combination_max_order` | 2 | Largest `Σ|nᵢ|` in the combination search. |
| `backend` | `"auto"` | GPU when available (see {doc}`backends`). |

## References

- Breger, M., et al. 1993, A&A 271, 482 — the S/N ≥ 4 criterion.
- Baluev, R. V. 2008, MNRAS 385, 1279 — the false-alarm probability bound.
- Loumos, G. L., & Deeming, T. J. 1978, Ap&SS 56, 285 — frequency resolution.
- Schwarzenberg-Czerny, A. 1991, MNRAS 253, 198 — correlated residuals.
- Montgomery, M. H., & O'Donoghue, D. 1999, DSSN 13, 28 — analytic uncertainties.
- Lenz, P., & Breger, M. 2005, CoAst 146, 53 — Period04.
- Van Reeth, T., et al. 2015, ApJS 218, 27 — g-mode period-spacing patterns.
- Baran, A. S., & Koen, C. 2021, AcA 71, 113 — significance thresholds for space data.
