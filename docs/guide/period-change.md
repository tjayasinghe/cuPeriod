# Period changes (dP/dt)

A periodogram assumes the period is constant. Real periods drift: mass transfer
reshapes binary orbits, tidal decay is dragging hot Jupiters inward, stellar
evolution slides pulsation periods, and Miras wander for no secular reason at
all. Measuring the *derivative* is traditionally a hand-built O−C (observed
minus calculated) analysis — collect times of minimum for decades, count
cycles, fit a parabola (Sterken 2005) — or, at the other extreme, a
pulsar-style coherent search. cuPeriod v1.3 covers the ground in between:

- {func}`cuperiod.pdot_search` — **detect and localize**: a joint (period,
  dP/dt) grid search that makes *any* registered periodogram sensitive to a
  drifting period.
- {func}`cuperiod.phase_track` / {func}`cuperiod.fit_ephemeris` — **measure**:
  an automated O−C with per-segment uncertainties, and the same quadratic
  ephemeris fit for your own timing lists.
- {func}`cuperiod.period_profile` / {func}`cuperiod.fit_period_law` —
  **when one rate will not do**: the period measured independently in
  consecutive intervals, and parametric P(t) laws fitted to that track.
- {func}`cuperiod.sliding_periodogram` / {func}`cuperiod.wwz` — **explore**:
  time-resolved maps for period wander that is not a clean secular trend.
- {func}`cuperiod.eddington_plakidis` — **vet**: the classical test that a
  smooth-looking O−C is not just an accumulating random walk.
- {func}`cuperiod.batch_pdot` — **scale**: the whole chain over a catalogue,
  with per-star tables and optional diagnostic figures.

```python
import cuperiod as cup

res = cup.pdot_search((time, mag, err), period=0.5168)
print(res.best.period, res.best.pdot_s_per_yr)

track = cup.phase_track((time, mag, err), period=res.best.period,
                        pdot=res.best.pdot)
print(track.pdot_s_per_yr, track.pdot_err, track.f_test_p)
```

```text
search:  best period 0.5167997 d   dP/dt +19.01 s/yr
         power 0.862 at the best cell vs 0.130 anywhere on the dP/dt = 0 row
track:   dP/dt +18.943 ± 0.021 s/yr   F-test p = 8e-40   (planted: +18.93)
```

The division of labor matters: the search *finds* a drifting signal that an
ordinary periodogram has already lost (0.13 vs 0.86 above — six years of a
~19 s/yr drift smears a half-day period beyond recognition), and the tracker
turns it into a measurement with an honest uncertainty.

## The joint search

A constant dP/dt means the phase in cycles is exactly

```{math}
\varphi(t) = f \, \tau, \qquad
\tau = \frac{\ln(1 + a\,\Delta t)}{a}, \qquad
a = \frac{\dot P}{P},
```

with {math}`\Delta t` measured from the weighted mean epoch and {math}`a` the
fractional rate in 1/day. That epoch zeroes the first moment of the times,
which decouples phase from frequency and removes the leading frequency–drift
correlation; what remains is set by the sampling's third central moment, so the
two axes are exactly orthogonal only for observations symmetric about the
epoch. A run whose cadence densifies sharply at one end leaves a visibly
tilted ridge in the map — it widens the joint uncertainty without biasing the
best cell. The warp {math}`\tau` does not depend on {math}`f` — so evaluating
an *ordinary* periodogram on warped times turns it into a drifting-period
search at zero algorithmic cost. {func}`~cuperiod.pdot_search` steps a uniform
{math}`a` axis, re-runs the method per row (reusing its engine — on the GPU
one cufinufft plan serves the whole scan), and stacks a
`(n_pdot, n_frequency)` statistic map. To second order
{math}`\tau \approx \Delta t (1 - \tfrac{a}{2}\Delta t)`, the familiar
quadratic ephemeris; cuPeriod evaluates the exact logarithm (the same phase
PHOEBE integrates for its `dpdt`), which matters at the edges of the search
range for light curves spanning only a few hundred cycles.

Both axes are sampled to equal phase smear: the dP/dt step keeps the residual
drift under {math}`1/(2 s)` cycles across the baseline, exactly as the
frequency step does. Methods that resolve structure sharper than a sinusoid
smear proportionally sooner, so the step automatically refines by a per-method
harmonic scale (MHAOV's harmonic count; half the phase-bin count for PDM and
conditional entropy; SuperSmoother's smallest span). The default extent is set
in accumulated drift *cycles* (`max_cycle_drift`), which keeps the row count
baseline-independent while reaching every drift big enough to defocus a
constant-period search — and means the physical dP/dt coverage adapts to your
baseline, from ms/yr on a decade of eclipse timing to d/yr for a Mira.

Every registered method works, single- or multi-band, except BLS and TLS
(their trial grids carry box/transit structure a plain frequency window cannot
represent — transit-timing dP/dt work belongs to
{func}`~cuperiod.fit_ephemeris` anyway). `res.zero_pdot_periodogram()` returns
the dP/dt = 0 row as an ordinary {class}`~cuperiod.Periodogram` — it *is* the
constant-period search of the same window, which makes the comparison above a
one-liner.

:::{admonition} The search localizes; it does not do inference
:class: caution
The map's cell values are the wrapped method's statistic on warped data — the
2-D trials are not independent, and no false-alarm calibration exists for this
statistic on irregular sampling. Read the best cell as a candidate, then
measure and test it with {func}`~cuperiod.phase_track`.
:::

## Measuring: the automated O−C

{func}`~cuperiod.phase_track` fits one multi-harmonic template at a fixed
period (order chosen by BIC), splits the data into contiguous segments, and
measures the best time *shift* of the rigid template in each segment — a
Fisher-information uncertainty comes free from the template's slopes, and
sharp-featured light curves (eclipses, sawtooth RR Lyrae) time better than
sinusoidal ones for exactly that reason. The shifts are then fit against
linear and quadratic ephemerides:

```{math}
\mathrm{O\!-\!C}(t) = c_0 + c_1\,\Delta t + c_2\,\Delta t^2,
\qquad \dot P = 2 P c_2 ,
```

with the F-test between the two models (Pringle 1975; Sterken 2005) and
ΔBIC = `linear_fit.bic − quadratic_fit.bic` (positive favors the drift; > 10
is conventionally strong) both reported. A drift left in the fold would smear
the template, so the tracker re-derives the template on drift-corrected times
until self-consistent; when the accumulated drift exceeds a cycle or so, pass
the search's `best.pdot` as the starting guess. Results are always reported
against the constant-period ephemeris.

For timing lists — decades of published minima, TESS mid-eclipse times —
{func}`~cuperiod.fit_ephemeris` runs the same machinery from times alone,
assigning each event the nearest cycle of your trial ephemeris. Watch its
`worst_phase_residual`: values approaching 0.5 mean cycle counts are becoming
ambiguous, the classical O−C failure (a one-cycle error anywhere invalidates
the diagram).

Three conventions for "the period derivative" coexist in the literature, and
mixing them up has reached refereed journals (a published hot-Jupiter decay
rate was once wrong by the period squared — the conversion was multiplied
instead of divided). cuPeriod reports all of them:

| Convention | Definition | Where you meet it | Accessor |
| --- | --- | --- | --- |
| dP/dt | per unit time, dimensionless | most of this page | `pdot` (+ `pdot_s_per_yr`, `pdot_ms_per_yr`, `pdot_d_per_myr`) |
| dP/dE | per cycle: {math}`P \dot P` | eclipse-timing ephemerides, `T_0 + PE + QE^2` with {math}`Q = \tfrac{1}{2} P \dot P` | `2 * quadratic term` — tested as {math}`\dot P = 2Q/P` |
| {math}`\dot P / P` | fractional rate, 1/time | theory; the search's own axis | `pdot_over_p`, `dlnp_dt_per_yr` |

## When one rate is not enough

Everything above fits a *single* rate over the whole baseline. When the rate
itself evolves — an inspiralling binary, a runaway mass-transfer episode — that
number is a baseline-weighted average of a moving quantity, and neither it nor
its error bar means what it appears to. The give-away is not the F-test, which
will happily confirm the drift is real; it is that the O−C keeps shedding
residual when the ephemeris is allowed to curve further:

```python
ladder = {d: np.polyfit(dt, track.delay, d, w=1/track.delay_err) for d in (2, 3, 4)}
```

A quadratic *is* constant dP/dt, so a cubic that removes most of what remains
means the rate is changing. {attr}`~cuperiod.EphemerisFit.pddot` measures that
directly from a degree-3 fit, and {attr}`~cuperiod.EphemerisFit.implied_model`
reports which law it favours (`pddot = 0` is constant dP/dt;
`pddot = pdot²/P` is exponential).

The model-free answer is to stop assuming a shape. {func}`~cuperiod.period_profile`
chops the light curve into intervals short enough that the period is constant
within each, measures it independently in every one by minimizing a
fixed-order harmonic χ² over frequency, and returns the P(t) track;
{func}`~cuperiod.fit_period_law` then fits laws to *that*. This two-step recipe
is how period changes are actually published for systems whose rate is not
constant, and reproducing a published number usually means reproducing both
steps.

```python
flat = cup.detrend(time, mag, 40.0)          # see below — real data needs it
prof = cup.period_profile((time, flat, err), period=1.4456,
                          points_per_interval=50)
for fit in prof.compare():
    print(fit.law, fit.bic, fit.params)
best = prof.fit("singular")
print(best.fractional_change(time[0], time[-1]))    # total change over the run
print(best.timescale_yr_at([time[0], time[-1]]))    # P/|dP/dt|, start and end
```

The registered laws are `constant`, `linear` (constant dP/dt), `exponential`
(constant d ln P/dt), `singular` — {math}`P = P_0 \exp(b/(t - t_0))`, a
finite-time singularity, the description Tylenda et al. (2011) fitted to V1309
Scorpii's orbital decay through its merger — and `pejcha`, which lets the
exponent of that singularity float (Pejcha 2014). BIC ranks them;
`law="auto"` picks the best of all but `pejcha`, which needs an exceptionally
well-sampled track.

:::{admonition} A singularity will chase your worst point
:class: warning
A finite-time singularity parked just outside the data puts nearly all of its
variation into the first or last step of the track: `b/(t - t0)` becomes a spike
on one measurement, which buys a large χ² drop while describing nothing. On a
track that is not changing at all, that three-parameter spike will out-score
`constant` on BIC.

Two things stop it. `t0` is held at least one median point-spacing away from the
data — closer than that, the divergence is unresolvable by construction — and
any fit whose variation is still dominated by a single end step is flagged
`at_bound` and ranked *below* every unpinned fit, whatever its BIC. Check that
flag before believing a singular law, and read
{attr}`~cuperiod.PeriodLawFit.at_bound` in the batch table as
`law_at_bound`.
:::

:::{admonition} What the track's error bars mean
:class: note
`period_err` is the formal precision of one interval's measurement. On real
stars the track scatters about any smooth law by considerably more — the
light-curve shape evolves, spots migrate, and many period-changing systems
genuinely wander on timescales shorter than the baseline. That excess is also
correlated between neighbouring intervals, which a per-interval error cannot
express. Use these errors for relative weighting, and let
{func}`~cuperiod.fit_period_law` rescale them by `sqrt(chi2_reduced)` (its
default) before quoting a law parameter.
:::

## Detrending

Survey photometry of a period-changing star usually carries a brightness trend
far slower than the period and far larger than the variability. A periodogram
sees it as power at very low frequency, and the sampling window scatters that
straight onto the 1 c/d alias comb — where it competes with the signal. On the
OGLE photometry of V1309 Sco the 0.8 mag secular brightening does exactly this,
and an undetrended search returns a one-day alias instead of the orbital period.

{func}`cuperiod.detrend` removes it with a running *median* — eclipses and
outliers are one-sided and would drag a running mean into the feature being
measured. Pick a width several times the period and a small fraction of the
trend's own timescale:

```python
flat = cup.detrend(time, mag, 40.0)              # 40-day window
trend = cup.running_median(time, mag, 40.0)      # or the trend itself
```

Nothing detrends automatically: the width is a physical judgement about a
particular star, too narrow a window eats the signal, and silently reshaping a
light curve before measuring its period is not a library's decision to make.

## Time-resolved views

When the period *wanders* rather than drifts — Miras, semi-regulars, spotted
binaries — a single quadratic is the wrong model, and the honest picture is a
time-frequency map. {func}`~cuperiod.sliding_periodogram` computes any
registered method in stepped windows and tracks the per-window peak;
{func}`~cuperiod.wwz` is Foster's (1996) Weighted Wavelet Z-transform, the
long-period-variable community's standard, whose Gaussian window adapts to
each trial frequency instead of being fixed in days. Both trade frequency
resolution for time localization — a window of length {math}`W` cannot
resolve fractional period changes below {math}`\sim P/W` — and both expose a
`period_trend()` that turns the ridge into a rough dP/dt. Treat these as
diagnostics: the coherent tools above are an order of magnitude more
sensitive to a genuinely secular trend, but they will happily average over a
reversal that a map shows at a glance.

## Before you believe your dP/dt

Parabolas are cheap. The literature's graveyard of retracted period-change
claims is well populated, and almost every burial follows one of these:

**Random walks fit parabolas beautifully.** A star whose period fluctuates
randomly cycle-to-cycle accumulates a random walk in its O−C; Sterken (2005)
demonstrates that cumulative dice rolls produce a convincing "quadratic"
diagram. This is not exotic: about a third of Cepheids (two-thirds of first
overtones) and most Miras show non-evolutionary wander. Run
{func}`~cuperiod.eddington_plakidis` on your timing series: a significant
slope in its lag profile says the wander is real, and its `epsilon` measures
it (Eddington & Plakidis 1929 — o Ceti's period fluctuates by ~1.4% per
cycle; the estimator here corrects the classical recipe's detrending bias).

**A distant companion mimics a decaying orbit.** The light-travel-time effect
of a third body sampled over less than its orbital period is
indistinguishable from a quadratic. Of the three famous hot-Jupiter decay
claims, two dissolved this way (WASP-4b's "decay" is a ~7 M_Jup companion at
8 au; Kepler-1658 "b" is a bound eclipsing binary) — only WASP-12b survives.
An RV *trend* bound is not enough to exclude it; the curvature (jerk) is what
breaks the degeneracy.

**Eclipsing binaries: watch the secondary.** Apsidal motion in an eccentric
orbit and migrating starspots both move eclipse times by hundreds of seconds
— and both shift primary and secondary minima in *opposite* directions,
while a true period change shifts them together. Fit both eclipse families
and compare before claiming mass transfer.

**Modulation and spots.** Roughly half of RRab stars are Blazhko-modulated
(instantaneous periods swing by ~0.5% within a modulation cycle); spot
crossings shift transit mid-times by up to ~1% of the transit duration.
Average over whole modulation cycles, or flag and fit them jointly.

**Time systems.** At seconds-per-century sensitivity, the time standard is
science: HJD vs BJD differs by 8 s, TT−UTC has accumulated 37 leap seconds,
and combining century-old visual maxima with TESS BJD_TDB without a uniform
conversion injects a spurious ~minute-scale drift. Convert everything to one
barycentric dynamical standard first (Eastman et al. 2010).

## Whole catalogues

{func}`cuperiod.batch_pdot` runs the entire chain — detrend, search, track,
ephemeris ladder, P(t) profile, law ranking — over many light curves on the same
worker machinery as {func}`~cuperiod.batch_periodograms`, and writes **three**
tables, because a period-change catalogue is genuinely three-shaped:

| Sink | One row per | Holds |
| --- | --- | --- |
| `sink` | light curve | Everything a survey-scale cut needs: search, O−C measurement and its significance, the curvature ladder, and the winning period law with its parameters. |
| `oc_sink` | O−C segment | The timing points behind each star's dP/dt — what you plot, refit, or hand to a referee. |
| `profile_sink` | P(t) interval | The model-free track: period, error, χ², and the window-edge flag. |

```python
summary = cup.batch_pdot(
    "lightcurves/*.csv",
    period=known_periods,          # a float, or a {key: period} mapping
    detrend_width=40.0,
    sink="pdot.parquet",
    oc_sink="oc.parquet",
    profile_sink="profile.parquet",
    plots="figures/", plot_keys=candidates,
    workers=8,
)
print(summary.n_done, summary.n_failed)
```

Sinks accept `.parquet`/`.csv` files or directories (resumable, one part per
chunk — the three must be *different* directories). Every stage is wrapped so
one pathological star cannot kill a survey run: a failure still contributes a
summary row whose `status` carries the reason, so nothing disappears from the
catalogue, and every law's parameter columns are present and NaN-filled in
every part so a directory sink reads back with one schema.

Useful cuts on the summary table:

| Column | Meaning |
| --- | --- |
| `track_f_test_p` | Is the drift real at all? |
| `track_pdot_snr` | `|dP/dt|` over its χ²-inflated error — the honest significance. |
| `ladder_curvature_ratio` | O−C residual, cubic over quadratic. Near 1: a constant rate is adequate. Well below: the rate is evolving. |
| `implied_model` | Which law the fitted `pddot` favours. |
| `law`, `law_delta_bic` | The winning P(t) law and its margin over the runner-up. |
| `law_at_bound` | The winning law ran onto a constraint — treat its parameters and its BIC as meaningless. |
| `search_ridge_correlation` | How strongly P and dP/dt trade off at the peak. |
| `wrap_risk`, `profile_at_window_edge` | Cycle-count slip, and periods clipped by the search window. |

`plots=` writes a six-panel diagnostic per star (light curve, drift-corrected
fold, search map, O−C, P(t) track with the fitted law, and the decay
timescale). It needs matplotlib — `pip install "cuperiod[plot]"` — and is slow
enough at catalogue scale that it is worth pairing with `plot_keys`. The same
figure is available directly as {func}`cuperiod.pdot.plots.pdot_figure`, and
its panel helpers can be composed into your own layout.

## Settings

All the knobs live on {class}`~cuperiod.PdotSettings`
(environment-overridable as `CUPERIOD_PDOT_<FIELD>`); the wrapped method
keeps its own settings via `method_settings=`.

| Setting | Default | What it controls |
| --- | --- | --- |
| `window_rayleigh` | 30 | Frequency half-window around the center period, in Rayleigh widths. |
| `samples_per_peak` | 10 | Frequency oversampling inside the window. |
| `max_cycle_drift` | 10 | Half-range of the dP/dt axis, in accumulated drift cycles at the half-baseline. |
| `samples_per_drift` | 5 | dP/dt oversampling (worst-case template mismatch ~0.9/s²: 3.5% at 5; use 2 for survey speed at ~22%). |
| `harmonic_scale` | auto | Extra dP/dt-step refinement for sharp-featured methods; `None` derives it from the method. |
| `pdot_over_p_range` | auto | Explicit (lo, hi) for the dP/dt axis, in 1/day. |
| `n_segments` | auto | Phase-tracking segment count. |
| `min_segment_points`, `min_segment_cycles` | 20, 3 | Segment floors (raise the cycles floor for beating-prone multiperiodic stars). |
| `harmonics`, `max_harmonics` | auto, 10 | Template order, or the cap on its BIC selection. |
| `segment_offsets` | on | Free magnitude offset per segment (calibration-drift protection). |
| `template_iterations` | 1 | Template re-derivations on drift-corrected times. |
| `profile_points_per_interval` | 50 | Target points per P(t) interval (the subsample size of the published V1309 Sco analysis). |
| `profile_min_points` | 15 | Fewest points an interval may hold and still be measured. |
| `profile_period_window` | 0.05 | Fractional half-width of each interval's period search. Must exceed the total period change; too wide invites 1 c/d aliases. |

## References

- Sterken, C. 2005, ASP Conf. Ser. 335, 3 — the O−C method, its ambiguities, and its abuses.
- Mikulášek, Z., et al. 2012, arXiv:1212.5527 — the exact constant-dP/dt phase.
- Eddington, A. S., & Plakidis, S. 1929, MNRAS 90, 65 — random period fluctuations.
- Foster, G. 1996, AJ 112, 1709 — the Weighted Wavelet Z-transform.
- Templeton, M. R., Mattei, J. A., & Willson, L. A. 2005, AJ 130, 776 — Mira period wander with the WWZ.
- Katz, M. L., et al. 2021, MNRAS 503, 2665 — the GPU (P, dP/dt) conditional-entropy search.
- Schwarzenberg-Czerny, A. 1996, ApJ 460, L107 — multi-harmonic AOV, the statistic behind the interval fits.
- Tylenda, R., et al. 2011, A&A 528, A114 — V1309 Sco's orbital decay; the `singular` law is their Eq. (1).
- Pejcha, O. 2014, ApJ 788, 22 — the `pejcha` law (their Eq. 12), and the quoted V1309 Sco decay timescales.
- Montgomery, M. H., & O'Donoghue, D. 1999, DSSN 13, 28 — the fallback analytic frequency error.
- Murphy, S. J., et al. 2014, MNRAS 441, 2515 — segment phase tracking as time delays.
- Patra, K. C., et al. 2017, AJ 154, 4; Yee, S. W., et al. 2020, ApJ 888, L5 — WASP-12b's orbital decay.
- Winn, J. N., & Stefánsson, G. 2025, PSJ 6, 300 — light-travel time masquerading as decay.
- Eastman, J., Siverd, R., & Gaudi, B. S. 2010, PASP 122, 935 — time standards.
- Deeg, H. J. 2021, Galaxies 9, 1 — modern eclipse-minimum timing errors.

---

Next: {doc}`multiband` — combining several filters.
