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
- {func}`cuperiod.ltte_search` — **when the O−C is not monotonic**: a
  light-travel-time orbit (a third body — or a magnetic cycle) fitted jointly
  with the ephemeris, so a cycle shorter than the baseline is not averaged
  into a spurious rate.
- {func}`cuperiod.sliding_periodogram` / {func}`cuperiod.wwz` — **explore**:
  time-resolved maps for period wander that is not a clean secular trend.
- {func}`cuperiod.eddington_plakidis` — **vet**: the classical test that a
  smooth-looking O−C is not just an accumulating random walk.
- {func}`cuperiod.detectable_pdot` / {func}`cuperiod.mass_loss_rate` /
  {func}`cuperiod.mass_transfer_rate` — **interpret**: how small a rate the
  timing data could have measured, and what a secular one would mean as mass
  loss or mass transfer.
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
harmonic scale: MHAOV's harmonic count (3 at its defaults); half the phase-bin count
for PDM and conditional entropy (5); a fixed 4 for String-Length; half the inverse of
SuperSmoother's smallest span (10); and 1 for every other method. The default extent is set
in accumulated drift *cycles* (`max_cycle_drift`), which keeps the row count
baseline-independent while reaching every drift big enough to defocus a
constant-period search — and means the physical dP/dt coverage adapts to your
baseline, from ms/yr on a decade of eclipse timing to d/yr for a Mira. The default
axis has exactly `4 × samples_per_drift × max_cycle_drift × harmonic_scale + 1` rows
(201 at the defaults); on a baseline of fewer than about 80 cycles it is trimmed to the
model's domain, and a baseline under about one cycle raises
{exc}`~cuperiod.InsufficientDataError`. An explicit `pdot_over_p_range` that spans zero
is laid on a zero-anchored lattice — it contains an exact dP/dt = 0 row, reaches the
wider bound exactly and never steps coarser than the derived step — so
`best.zero_pdot_power` is the true constant-period row; a range that excludes zero is
sampled end to end. `backend="auto"` defers to the wrapped method's settings backend
(`CUPERIOD_<METHOD>_BACKEND`), as {func}`~cuperiod.periodogram` does.

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
is conventionally strong) both reported. The F-test's `f_test_p` is `NaN` — not
"infinitely significant" — when both models fit the O−C exactly or a χ² is not
finite. The template's harmonic order is chosen with the scale-free BIC
{math}`N\ln(\chi^2/N) + k\ln N` whether or not errors are given, so optimistic error
bars cannot push it to `max_harmonics`. Tracking needs at least four segments (the
quadratic ephemeris needs four O−C points); when the data cannot form them the error
says which floor to relax. A drift left in the fold would smear
the template, so the tracker re-derives the template on drift-corrected times
until self-consistent; when the accumulated drift exceeds a cycle or so, pass
the search's `best.pdot` as the starting guess. Results are always reported
against the constant-period ephemeris.

For timing lists — decades of published minima, TESS mid-eclipse times —
{func}`~cuperiod.fit_ephemeris` runs the same machinery from times alone,
assigning each event the nearest cycle of your trial ephemeris. Watch its
`worst_phase_residual`: values approaching 0.5 mean cycle counts are becoming
ambiguous, the classical O−C failure (a one-cycle error anywhere invalidates
the diagram). Timings may come in any order, each keeping its own error, and a
non-finite time is dropped together with its error. With `errors=None` the common
timing error is estimated from the scatter about the fitted quadratic (so the
quadratic's `chi2_reduced` is 1 by construction). {func}`~cuperiod.fit_ephemeris`
always fits the quadratic; for a cubic see
[When one rate is not enough](#when-one-rate-is-not-enough). On any
{class}`~cuperiod.EphemerisFit`, `pdot_err_scaled` (and, on a cubic,
`pddot_err_scaled`) inflates the formal error by `sqrt(chi2_reduced)` when that
exceeds 1.

Timing lists exist for thousands of eclipsing binaries. The AAVSO hosts the
late Bob Nelson's database of over 5800 O−C files — one spreadsheet of minima
per star, linked from <https://www.aavso.org/bob-nelsons-o-c-files> — and the
Czech Variable Star and Exoplanet Section's O−C gateway at
<https://var.astro.cz> holds some 320,000 minima (registration required).
Both are the natural input to {func}`~cuperiod.fit_ephemeris` and
{func}`~cuperiod.ltte_search`, after the time-system conversion discussed
below.

:::{admonition} Contact binaries: the half-period branch
:class: note
A light curve with two equal minima is almost perfectly symmetric under a
half-period shift, so a segment's best template shift lands on either branch
by noise, and the O−C would jump by P/2 between segments. The tracker
measures the template's fold symmetry ({attr}`~cuperiod.PhaseTrackResult.symmetry_order`,
2 for such a star) and unwraps consecutive delays in steps of
`period / symmetry_order`: cycle continuity is the only information that
separates the branches, and the O−C is then defined modulo that step. This
was found on OGLE bulge contact binaries, where it turned hour-scale
"amplitudes" into the few-hundred-second light-time orbits their published
timings show.
:::

Three conventions for "the period derivative" coexist in the literature, and
mixing them up has reached refereed journals (a published hot-Jupiter decay
rate was once wrong by the period squared — the conversion was multiplied
instead of divided). cuPeriod reports all of them, with the same accessors on every
result that carries a rate — the search's {class}`~cuperiod.PdotBest`, a
{class}`~cuperiod.PhaseTrackResult`, an {class}`~cuperiod.EphemerisFit` and an
{class}`~cuperiod.LTTEFit` (whose values are the *secular* rate, with the cycle
removed):

| Convention | Definition | Where you meet it | Accessor |
| --- | --- | --- | --- |
| dP/dt | per unit time, dimensionless | most of this page | `pdot` (+ `pdot_s_per_yr`, `pdot_ms_per_yr`, `pdot_d_per_yr`, `pdot_d_per_myr`) |
| dP/dE | per cycle: {math}`P \dot P` | eclipse-timing ephemerides, `T_0 + PE + QE^2` with {math}`Q = \tfrac{1}{2} P \dot P` | `2 * quadratic term` — tested as {math}`\dot P = 2Q/P` |
| {math}`\dot P / P` | fractional rate, 1/time | theory; the search's own axis | `pdot_over_p` (1/day), `dlnp_dt_per_yr` |

`pdot_over_p` divides by `period` on a {class}`~cuperiod.PdotBest` and a
{class}`~cuperiod.PhaseTrackResult`, and by `period_ref` on an
{class}`~cuperiod.EphemerisFit` and an {class}`~cuperiod.LTTEFit`.
{func}`~cuperiod.convert_pdot` converts a bare number between the units.

## A third body: the cyclic O−C

A close binary with a distant companion orbits their common centre of mass,
and its eclipses arrive early or late by the light-travel time across that
orbit — the light-travel-time effect (LTTE; Irwin 1952). The binary's period
is constant; the O−C is not, and it is *periodic*:

```{math}
\mathrm{O\!-\!C}(t) = \sum_k c_k\,\Delta t^k
  + A\left[\frac{1-e^2}{1+e\cos\nu}\,\sin(\nu+\omega) + e\sin\omega\right],
\qquad A = \frac{a_{12}\sin i}{c},
```

with {math}`\nu` the true anomaly at the outer period {math}`P_2`. Which of
two regimes you are in is decided by the baseline:

- **{math}`P_2 \gg` baseline.** Less than one cycle is sampled and the
  sinusoid is indistinguishable from a parabola. This is how WASP-4b's
  "orbital decay" became a companion (see below); only curvature beyond a
  parabola — a cubic ephemeris's {attr}`~cuperiod.EphemerisFit.pddot`, see
  [When one rate is not enough](#when-one-rate-is-not-enough) — or an
  independent radial-velocity jerk can break it.
- **{math}`P_2 \lesssim` baseline.** The O−C is *not monotonic*, and the rate
  a parabola reports is a baseline-weighted average of a wobble: its value
  and its sign depend on which part of the cycle the data happen to cover.
  This regime is common. With 15 years of OGLE bulge photometry, Hong et al.
  (2022) found 307 of 14,127 contact binaries with a sinusoidal O−C and 104
  with a sinusoid on top of a parabola, at outer periods of 5–14 yr; Hajdu et
  al. (2019) list 992 LTTE candidates in OGLE-IV; Pietrukowicz et al. (2017)
  recognised 35 cyclic systems among the 22,462 OGLE-III binaries they
  searched for merger precursors; and Gazeas et al. (2021) find a cycle *and*
  a negative trend together in ultra-short-period contact binaries.

{func}`~cuperiod.ltte_search` fits the cycle and the ephemeris polynomial
*together*. At every trial outer period the model is linear in
{math}`A\cos\omega`, {math}`A\sin\omega` and the polynomial, so the scan is
a floating-polynomial periodogram of the O−C — the χ² improvement over the
polynomial alone at each trial — and the best trial is polished by nonlinear
least squares, with Irwin's eccentric terms freed on request (`eccentric=True`).
It takes a {class}`~cuperiod.PhaseTrackResult`, a list of event times (cycles
assigned like {func}`~cuperiod.fit_ephemeris`), or explicit O−C values:

```python
track = cup.phase_track((time, flat, err), period=0.445851,
                        settings=cup.PdotSettings(n_segments=40))
res = track.ltte_search()                       # or cup.ltte_search(track)
best = res.best
print(best.outer_period_yr, best.amplitude_s, res.delta_bic)
print(res.null.pdot_d_per_yr, best.pdot_d_per_yr)    # parabola alone vs joint
print(best.mass_function, best.minimum_third_mass(1.5))
```

```text
OGLE-BLG-ECL-289126: 14.3 yr of OGLE-III+IV photometry, 40 segments
  parabola alone  : dP/dt = -1.3e-7 +- 2.0e-7 d/yr         chi2_r = 58
  light-time orbit: P2 = 6.43 +- 0.09 yr   A = 660 +- 17 s   dBIC = +2081
                    (published: 6.2 +- 0.2 yr, Hong et al. 2022)
  secular dP/dt   : +1.5e-7 +- 0.5e-7 d/yr                  chi2_r = 1.3
  f(m) = 0.056 Msun  ->  m3 >= 0.63 Msun for a 1.5 Msun binary
```

Every error in this block is the formal one times `sqrt(chi2_reduced)` of its own
fit, floored at 1. For the orbit that factor is
{attr}`LTTEFit.error_inflation <cuperiod.LTTEFit.error_inflation>` (1.16 here); the
attributes `best.outer_period_err_yr` and `best.amplitude_err_s` return the formal
errors (0.08 yr and 15 s here), so multiply them by `best.error_inflation` to quote
the block's 0.09 yr and 17 s. The two dP/dt errors are `pdot_err_scaled`, which
already carries the factor. `cuperiod pdot` prints the orbit's errors and the secular
rate's the same way, and says so in its `errors  formal x ...` line.

The second number pair is the point. `res.null` is the polynomial-only fit —
the quadratic ephemeris {func}`~cuperiod.phase_track` already reported — and
`best.pdot` is the *secular* rate with the cycle removed; `res.pdot_change` is
how far the parabola sat from it. On six of Hong et al.'s triples
(`benchmarks/pdot_ogle_triples.py`, written up with figures in
`benchmarks/REPORT_ogle_triples.md`; raw survey photometry, no per-season
binary modelling) all six recovered outer periods agree with the published
ones within 2σ (median offset 4%), while the parabola alone claimed a ≥ 3σ
dP/dt in three of the six. On synthetic light curves
(`benchmarks/pdot_ltte_recovery.py`, 120–600 s orbits, no secular drift):

| {math}`P_2` / baseline | parabola alone: ≥ 3σ "dP/dt" | cycle detected (ΔBIC > 10) | {math}`P_2` within 10% |
| --- | --- | --- | --- |
| 0.25–0.5 | 0% | 89% | 100% |
| 0.5–0.75 | 11% | 83% | 83% |
| 0.75–1 | 64% | 82% | 45% |
| 1–1.5 | 73% | 12% | degenerate |

with no false cycle and no false rate on the 40 stars without an orbit. Read
the last row as the first regime above: once the baseline holds less than a
cycle the joint fit can no longer separate the two either, and only the
curvature test or outside data can.

What comes back with the orbit: `amplitude_s`, `a_sini_au`, the mass
function {math}`f(m_3) = (a_{12}\sin i)^3/P_2^2` in solar units and
{meth}`~cuperiod.LTTEFit.minimum_third_mass` for a given inner mass;
{meth}`~cuperiod.LTTEFit.period_at`, the *apparent* instantaneous period the
orbit produces — a light-time orbit shows up in {func}`~cuperiod.period_profile`
as a sinusoidal P(t) of semi-amplitude {math}`2\pi A P/P_2`, not as a trend;
and `n_cycles`, `at_grid_edge`, `delta_bic` and `f_test_p` to read the
detection by. Two cautions. The outer period was *searched*, so with a dozen
points the F-test and ΔBIC are indicative rather than calibrated — Hong et al.
required a Bayes factor above 10 *and* visual inspection. And an O−C alone
cannot tell a third body from a magnetic activity cycle (Applegate 1992),
whose amplitudes and quasi-periods fall in the same range; a mass function
that implies a companion brighter than the binary, or a cycle that is not
strictly periodic, argues for the latter, and only radial velocities or a
brightness/colour modulation in step with the O−C settle it.

Without `errors=`, the noise is estimated from the joint fit: `delta_bic`, `aic` and
`bic` come from the concentrated likelihood {math}`n\ln(\mathrm{RSS}/n) + k\ln n`,
`chi2_reduced` is 1 by construction, and `meta["errors"]` says which applied. The
default shortest trial outer period follows the spacing of *distinct* epochs
(`meta["epoch_spacing"]`, `meta["n_epochs"]`), so repeated or clustered timings do
not collapse it. {meth}`PdotSensitivity.sigma_amplitude()
<cuperiod.PdotSensitivity.sigma_amplitude>` is `inf` for outer periods the epochs
cannot constrain (aliased with the cadence, or far beyond the baseline).

## When one rate is not enough

Everything above fits a *single* rate over the whole baseline. When the rate
itself evolves — an inspiralling binary, a runaway mass-transfer episode — that
number is a baseline-weighted average of a moving quantity, and neither it nor
its error bar means what it appears to. The give-away is not the F-test, which
will happily confirm the drift is real; it is that the O−C keeps shedding
residual when the ephemeris is allowed to curve further:

```python
import numpy as np

dt = track.t_mid - track.t_ref
ladder = {d: np.polyfit(dt, track.delay, d, w=1/track.delay_err) for d in (2, 3, 4)}
```

A quadratic *is* constant dP/dt, so a cubic that removes most of what remains
means the rate is changing. {attr}`~cuperiod.EphemerisFit.pddot` measures that
directly from a degree-3 fit, and {attr}`~cuperiod.EphemerisFit.implied_model`
reports which law it favours (`pddot = 0` is constant dP/dt;
`pddot = pdot²/P` is exponential); both are `NaN` / `"unknown"` below degree 3.
{func}`~cuperiod.fit_ephemeris` and {func}`~cuperiod.phase_track` fit only the
linear and quadratic ephemerides. Two public paths give the cubic:

- {func}`~cuperiod.ltte_search` with `degree=3` (or
  `PdotSettings(ltte_degree=3)`): its `null` is the cubic
  {class}`~cuperiod.EphemerisFit` alone, fitted to the same O−C as the light-time
  orbit. It takes a track or a timing list, and raises
  {exc}`~cuperiod.InsufficientDataError` below `ltte_min_points` (8) O−C points or
  on a baseline too short for its cycle scan.
- {func}`~cuperiod.batch_pdot`: its summary table reports the cubic as the
  `ladder_*` columns and `implied_model` (see [Whole catalogues](#whole-catalogues)).

```python
cubic = cup.ltte_search(track, degree=3).null      # an EphemerisFit, degree 3
print(cubic.pddot, cubic.pddot_err_scaled, cubic.implied_model)
```

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
    print(fit.law, fit.bic_scaled, fit.params)
best = prof.fit("singular")
print(best.fractional_change(time[0], time[-1]))    # total change over the run
print(best.timescale_yr_at([time[0], time[-1]]))    # P/|dP/dt|, start and end
```

Each interval's frequency is converged on the χ² minimum (a bounded Brent search inside
a harmonic-scaled scan), so the track's scatter matches its error bars even for
many-harmonic, high-S/N light curves; `PeriodPoint.amplitude` is the fundamental's
semi-amplitude.

The registered laws are `constant`, `linear` (constant dP/dt), `exponential`
(constant d ln P/dt), `singular` — {math}`P = P_0 \exp(b/(t - t_0))`, a
finite-time singularity, the description Tylenda et al. (2011) fitted to V1309
Scorpii's orbital decay through its merger — and `pejcha`, which lets the
exponent of that singularity float (Pejcha 2014). They are ranked by
{attr}`~cuperiod.PeriodLawFit.bic_scaled`: the χ² divided by the excess scatter of the
most flexible free law (floored at 1), so understated period errors no longer buy a
period change (errors understated k-fold used to inflate BIC differences by k²), and
the "ΔBIC ≳ 10 is decisive" rule applies to it; `.bic` is still reported. Without
`period_err` the laws are compared on the residuals' own scale (the concentrated
likelihood), and `PeriodLawFit.weighted` says which applied. `law="auto"` picks the best
of all but `pejcha`, which needs an exceptionally well-sampled track. Singular and
Pejcha laws are fitted in coordinates about the mid-epoch (milliseconds instead of
seconds per fit); a singular fit in the exponential limit reports `t0 = ±inf`, and a
parameter the data do not constrain, or one held at a bound, reports `NaN`.

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
`at_bound` and ranked *below* every unpinned fit, whatever its BIC. Precisely, the flag
means the law changes by more over the first or last resolution element (one median
point spacing) than over half its whole swing, wherever the last samples happen to
fall. Check that flag before believing a singular law, and read
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

The exact running median uses a sliding sorted window and handles ~10⁵-point curves
quickly. `method="binned"` is the faster approximation: it visits only occupied bins (a
stray epoch such as a `0` among Julian dates no longer allocates gigabytes), and a bin
with fewer than `min_points` points (3 by default) is not an interpolation node — it
takes the trend interpolated from its populated neighbours, or the global median.
Masked or NA entries count as missing.

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
`period_trend()` that turns the ridge into a rough *fractional* rate: it returns
the pair `((dP/dt)/P, error)` in 1/day, so multiply by the period for dP/dt. Treat these as
diagnostics: the coherent tools above are an order of magnitude more
sensitive to a genuinely secular trend, but they will happily average over a
reversal that a map shows at a glance.

```python
res = cup.wwz((time, mag), frequency=freqs)     # or cup.sliding_periodogram(...)
res.peak_period                                 # the ridge: an array, one per tau
rate, rate_err = res.period_trend()             # (dP/dt)/P in 1/day; times P for dP/dt
res.to_dict(include_map=True)                   # same spelling on both maps
```

- The ridge is a property on both maps — `res.peak_frequency`, `res.peak_period` and
  (WWZ) `res.peak_wwa` are arrays, not methods.
- {func}`~cuperiod.wwz` runs a multicore numba kernel when the `[fast]` extra is
  installed and a vectorised NumPy path otherwise (`backend="auto" | "cpu" | "numba" |
  "numpy"`; the one that ran is in `meta["wwz"]["backend"]`) — 50–85× faster than before
  with numba. It centres the values first (a constant light curve gives an all-`NaN`
  map), and it keeps rows whose unused error is missing, zero or negative, such as
  AAVSO visual estimates.
- `WWZResult.period_trend()` counts its degrees of freedom in independent windows, so
  its error no longer shrinks with a denser tau grid, and it is `inf` when the taus span
  less than about one window.
- `SlidingPeriodogram.window_periodogram(i)` reports that window's own `baseline` and
  point count (`n_window_points`), so `best_periods()` separates peaks at the window's
  resolution; the run's values are in `meta["run_baseline"]` / `meta["run_n_samples"]`.
  {func}`~cuperiod.sliding_periodogram` honours the method's settings backend, no
  longer loses the last observation to rounding, and raises `ValueError` for a
  non-finite window, step or period, or a period longer than the window. The default
  window and step tile the baseline exactly. A custom pair that does not gets one
  extra window ending at the last epoch when more than half a step of the tail is
  left over; otherwise up to half a step of the tail stays outside the map (with
  `window=30, step=20` on a 100-day run, the windows end at day 90).

## How small a rate can you measure?

Not as small as a phase-shift rule of thumb suggests, and not as easily as the
baseline alone suggests. {func}`~cuperiod.detectable_pdot` gives the formal
floor of the quadratic-ephemeris method for a timing precision, a number of
timings and a baseline — or for the actual epochs of your timing points, since
a dense modern season on a sparse archival tail is a very different design
from uniform coverage. For uniform sampling of {math}`n` timings of precision
{math}`\sigma_t` over {math}`T`,

```{math}
\sigma(\dot P) \simeq 4\sqrt{45}\,\frac{P\,\sigma_t}{T^2\sqrt{n}}
  \approx 27\,\frac{P\,\sigma_t}{T^2\sqrt{n}},
```

improving with the *square* of the baseline and only with the root of the
number of timings:

```python
s = cup.detectable_pdot(0.4, 0.001, n_timings=15, baseline=15 * 365.25)
print(s.sigma_pdot_d_per_yr, s.threshold(3) * 365.25)   # 3.0e-8, 9.0e-8 d/yr
print(s.sigma_amplitude(6 * 365.25) * 86400)             # ~33 s light-time floor
```

The familiar estimate "a phase shift of 0.1 cycle accumulated over the
baseline", {math}`\dot P \sim 0.2\,P^2/T^2`, corresponds to a *single*
timing good to 0.1 P at one end of the span. Real eclipse timings reach
{math}`10^{-3}P` and there are dozens of them, which is why survey detections
sit two orders of magnitude lower: Hong et al. (2022)'s 13,716 parabolic rates
lie mostly within {math}`\pm 5\times10^{-6}` d/yr with errors down to
{math}`4\times10^{-9}` d/yr (the largest, −1.4 × 10⁻⁵ d/yr for
OGLE-BLG-ECL-169991; Pietrukowicz et al. 2017's strongest is −1.9 × 10⁻⁴ d/yr
in a 2.8-day detached binary), and a "typical" detected 10⁻⁶ d/yr is
2.7 × 10⁻⁹ s/s — {func}`~cuperiod.convert_pdot` moves between the units, and
every result now carries `pdot_d_per_yr` next to `pdot_s_per_yr`. The floor is
formal: real O−C diagrams scatter beyond their timing errors (an evolving light
curve, spots, genuine wander), which is what `pdot_err_scaled` inflates for.

## What a secular rate would mean

Two textbook channels change an orbital period secularly, and both follow from
Kepler's third law and angular-momentum bookkeeping. Isotropic (Jeans-mode)
mass loss widens the orbit,

```{math}
\frac{\dot P}{P} = 2\,\frac{\dot M_{\rm loss}}{M}
\quad\Longrightarrow\quad
\dot P = 2\times10^{-6}\left(\frac{P}{1\ \mathrm{yr}}\right)
\left(\frac{\dot M_{\rm loss}}{10^{-6}\,M_\odot\,\mathrm{yr^{-1}}}\right)
\left(\frac{M_\odot}{M}\right),
```

so a mass-losing system *lengthens* its period; conservative transfer from a
donor {math}`M_d` to an accretor {math}`M_a` gives (Kwee 1958)

```{math}
\frac{\dot P}{P} = 3\,\dot M_{\rm tr}\left(\frac{1}{M_d} - \frac{1}{M_a}\right)
  = 3\,\frac{\dot M_{\rm tr}}{M_d}\,(1 - q), \qquad q = M_d/M_a,
```

lengthening the period when the donor is the less massive star and shortening
it when it is the more massive one — and, when one star dominates the mass,
reducing to the same {math}`\dot P \sim (2\text{–}3)\,P\,\dot M/M` as the
wind. {func}`~cuperiod.mass_loss_rate`, {func}`~cuperiod.mass_transfer_rate`
and their inverses {func}`~cuperiod.pdot_from_mass_loss`,
{func}`~cuperiod.pdot_from_mass_transfer` do the arithmetic, in solar masses
per year with the period in days:

```python
pdot = cup.convert_pdot(1e-6, "d/yr", "d/d")
cup.mass_loss_rate(pdot, 0.4, 1.5)             # 1.9e-6 Msun/yr, if a wind did it
cup.mass_transfer_rate(pdot, 0.4, 0.4, 1.1)    # 5.2e-7 Msun/yr from the 0.4 Msun star
cup.pdot_from_mass_loss(1e-6, 365.25, 1.0)     # 2e-6: the scaling above
```

Use them for the *sign* check and for limits more than for measurements. A
negative rate returned by `mass_loss_rate` means the period is shortening,
which no wind can do; a negative `mass_transfer_rate` means the measured sign
is incompatible with the assumed donor. Schaefer (2024) collected dP/dt for 52
cataclysmic variables and found 44% of them positive — impossible under
magnetic-braking evolution — and the rest more negative than evolution allows
by two orders of magnitude; King & Lasota (2024) conclude that an O−C rate is,
in almost every case, dominated by something other than the secular evolution.
The two stellar mergers caught in the act make the same point from the other
side: V1309 Sco's decay (the `singular` law above) was recognised only in
hindsight, and the one prediction made in advance — KIC 9832227, an
exponentially shrinking contact-binary period extrapolated to a 2022 merger
(Molnar et al. 2017) — fell to archival timings: a 2003 Vulcan eclipse time and
an NSVS one disagreed with the decay model by up to an hour (Socia et al. 2018).
Pietrukowicz et al. (2017) found no V1309 analogue among 22,462 OGLE-III
binaries. What a non-detection *does* deliver is a limit: a 3σ threshold from
{func}`~cuperiod.detectable_pdot` through {func}`~cuperiod.mass_loss_rate` is
an upper bound on the wind a wide eclipsing binary can be driving.

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
`slope_significance` is a z-score under the no-wander null (about N(0, 1)), not a
t-statistic, and `epsilon_err` / `alpha_err` are the estimator's own uncertainties:
they scale with the estimate, so read them in log space. An exactly periodic series
reports 0.

**A distant companion mimics a decaying orbit.** The light-travel-time effect
of a third body sampled over less than its orbital period is
indistinguishable from a quadratic. Of the three famous hot-Jupiter decay
claims, two dissolved this way (WASP-4b's "decay" is a ~7 M_Jup companion at
8 au; Kepler-1658 "b" is a bound eclipsing binary) — only WASP-12b survives.
An RV *trend* bound is not enough to exclude it; the curvature (jerk) is what
breaks the degeneracy. A companion whose orbit is *shorter* than the baseline
does the opposite: a non-monotonic O−C that a parabola averages into a rate of
arbitrary sign — {func}`~cuperiod.ltte_search` fits the cycle and the rate
together (see above).

**Archival timings can be wrong by hours.** A decade-old minimum from a
different survey, time system, or typographic tradition anchors the far end of
the parabola with enormous leverage. KIC 9832227's merger prediction survived
every modern data set and died on a 2003 Vulcan timing and an NSVS one that
disagreed with the decay model by up to an hour (Socia et al. 2018); whether
those points or the model were wrong, the lesson is that an O−C built across
archives needs its oldest points re-derived from the original photometry, not
transcribed.

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
if __name__ == "__main__":         # worker pools start fresh processes (see the batch guide)
    summary = cup.batch_pdot(
        "lightcurves/*.csv",
        period=known_periods,      # a float, or a {key: period} catalogue
        detrend_width=40.0,
        sink="pdot.parquet",
        oc_sink="oc.parquet",
        profile_sink="profile.parquet",
        plots="figures/", plot_keys=candidates,
        workers=8,
    )
    print(summary.n_done, summary.n_failed)
```

Sinks accept `.parquet`/`.csv` files or directories (one part per chunk — the three
must be *different* directories), in any mix. Every stage is wrapped so one
pathological star cannot kill a survey run: a failure still contributes a summary row
whose `status` carries the reason (and the light curve's `n_samples` and `baseline`),
so nothing disappears from the catalogue, and every law's parameter columns are present
and NaN-filled in every part so a directory sink reads back with one schema.

**Resuming** works per light curve across the three tables: a star is skipped only when
its summary row is `ok` for the same method and every requested table holds all of its
rows, so interrupted runs, deleted part files and a long-table sink added later all heal
on the next call. Stars that failed are retried, and a recomputed star replaces its rows
in every table — the summary keeps one row per key, and a run with a different method
replaces the earlier rows. An identical re-run into a file sink computes nothing.

`period=` takes a float or a catalogue: a `{key: period}` mapping (integer keys
included), a pandas Series, NumPy scalars or 0-d arrays. Catalogue keys are matched as
strings, as batch keys are; a missing, `NaN` or non-positive entry means a blind search
for that star, and a catalogue that matches no input key warns. `method` (BLS and TLS
are refused), `model`, `detrend_method`, `domain` and law names are checked before any
light curve or sink is touched; a bare string for `laws=` or `plot_keys=` counts as one
name.

Useful cuts on the summary table:

| Column | Meaning |
| --- | --- |
| `track_f_test_p` | Is the drift real at all? |
| `track_pdot_snr` | `|dP/dt|` over its χ²-inflated error — the honest significance. |
| `ladder_curvature_ratio` | O−C residual, cubic over quadratic. Near 1: a constant rate is adequate. Well below: the rate is evolving. |
| `implied_model` | Which law the fitted `pddot` favours. `ladder_pddot_err` and `ladder_pdot_cubic_err` are inflated by `sqrt(chi2_reduced)` of the cubic fit when that exceeds 1. |
| `law`, `law_delta_bic`, `law_bic_scaled` | The winning P(t) law, its margin over the runner-up in `bic_scaled` (the ranking criterion; plain BIC for fits without one), and its `bic_scaled`. |
| `law_at_bound` | The winning law ran onto a constraint — treat its parameters and its BIC as meaningless. |
| `search_ridge_correlation` | How strongly P and dP/dt trade off at the peak. |
| `ltte_delta_bic`, `ltte_f_test_p` | Is the O−C cyclic? Positive ΔBIC favours a light-time orbit over the polynomial alone. |
| `ltte_outer_period_yr`, `ltte_amplitude_s`, `ltte_mass_function` | The orbit, if there is one. |
| `ltte_pdot`, `ltte_pdot_change` | The secular rate with the cycle removed, and how far the parabola-only rate sat from it. |
| `ltte_n_cycles`, `ltte_at_grid_edge` | Fewer than one cycle covered, or an outer period pinned at the scan's edge: treat the orbit as unresolved. |
| `wrap_risk`, `profile_at_window_edge` | Cycle-count slip, and periods clipped by the search window. |

`plots=` writes a six-panel diagnostic per star (light curve, drift-corrected
fold, search map, O−C, P(t) track with the fitted law, and the decay
timescale). It needs matplotlib — `pip install "cuperiod[plot]"`; without it
`plots=` raises `ImportError` naming that command before any light curve is read — and
is slow enough at catalogue scale that it is worth pairing with `plot_keys`. The O−C
panel draws the light-time orbit only when the O−C supports it (ΔBIC > 10 and the
outer period not at the edge of the scan); otherwise it shows the O−C alone. The summary row's
`figure` column gives each file's path. A key that is not plain `[A-Za-z0-9_.-]` (a
path-qualified `fieldA/lc_0000`), is longer than 120 characters or names a Windows
device gets a `-<8 hex>` digest suffix, so distinct keys never overwrite each other's
figures. Figures follow the brightness domain (flux light curves are labelled `flux`,
not drawn inverted), and the drift-corrected fold uses the track's fitted ephemeris.
The same figure is available directly as {func}`cuperiod.pdot.plots.pdot_figure`, and
its panel helpers can be composed into your own layout; those draw whatever light-time
fit they are given, so pass `ltte=` only for a cycle you consider detected.

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
| `pdot_over_p_range` | auto | Explicit (lo, hi) for the dP/dt axis, in 1/day; a range spanning zero always contains an exact dP/dt = 0 row. |
| `n_segments` | auto | Phase-tracking segment count; at least 4 (the quadratic ephemeris needs four O−C points). |
| `min_segment_points`, `min_segment_cycles` | 20, 3 | Segment floors (raise the cycles floor for beating-prone multiperiodic stars). |
| `harmonics`, `max_harmonics` | auto, 10 | Template order, or the cap on its BIC selection. |
| `segment_offsets` | on | Free magnitude offset per segment (calibration-drift protection). |
| `template_iterations` | 1 | Template re-derivations on drift-corrected times. |
| `profile_points_per_interval` | 50 | Target points per P(t) interval (the subsample size of the published V1309 Sco analysis). |
| `profile_min_points` | 15 | Fewest points an interval may hold and still be measured. |
| `profile_period_window` | 0.05 | Fractional half-width of each interval's period search. Must exceed the total period change; too wide invites 1 c/d aliases. |
| `ltte_min_points` | 8 | Fewest O−C points the cyclic search accepts (the batch chain skips it below this). |
| `ltte_samples_per_peak` | 10 | Oversampling of the outer-frequency grid relative to 1/baseline. |
| `ltte_degree` | 2 | Ephemeris polynomial fitted with the cycle (2 = parabola + sinusoid; 1 = sinusoid on a linear ephemeris). |
| `ltte_min_cycles` | 1.0 | The scan's longest trial outer period is `baseline / ltte_min_cycles`; below one cycle a light-time orbit and a parabola are nearly degenerate. |

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
- Irwin, J. B. 1952, ApJ 116, 211 — the light-time orbit fitted by `ltte_search`.
- Applegate, J. H. 1992, ApJ 385, 621 — magnetic activity cycles as an alternative source of cyclic O−C.
- Kwee, K. K. 1958, BAN 14, 131 — the conservative mass-transfer period-change relation.
- Hong, K., et al. 2022, AJ 163, 157 — period changes of 14,127 OGLE bulge contact binaries; the 307 sinusoidal (LTTE) systems the OGLE benchmark draws from.
- Hajdu, T., et al. 2019, MNRAS 485, 2562 — 992 LTTE triple candidates among OGLE-IV bulge eclipsing binaries.
- Pietrukowicz, P., et al. 2017, AcA 67, 115 — the search for V1309 Sco analogues among 22,462 OGLE-III binaries.
- Gazeas, K., et al. 2021, MNRAS 502, 2879 — the CoBiToM project: contact binaries towards merging, with cyclic O−C plus a negative trend.
- Molnar, L. A., et al. 2017, ApJ 840, 1 — the KIC 9832227 red-nova prediction; Socia, Q. J., et al. 2018, ApJ 864, L32 — its negation by archival timings.
- Schaefer, B. E. 2024, ApJ 966, 155 — dP/dt of 52 cataclysmic variables, 44% of the wrong sign; King, A., & Lasota, J.-P. 2024, "The Perils of Pdot" (arXiv:2406.03948).

---

Next: {doc}`multiband` — combining several filters.
