# Period changes (dP/dt)

A periodogram assumes the period is constant. Real periods drift: mass transfer
reshapes binary orbits, tidal decay is dragging hot Jupiters inward, stellar
evolution slides pulsation periods, and Miras wander for no secular reason at
all. Measuring the *derivative* is traditionally a hand-built O−C (observed
minus calculated) analysis — collect times of minimum for decades, count
cycles, fit a parabola (Sterken 2005) — or, at the other extreme, a
pulsar-style coherent search. cuPeriod v1.3 covers the ground in between with
four complementary tools:

- {func}`cuperiod.pdot_search` — **detect and localize**: a joint (period,
  dP/dt) grid search that makes *any* registered periodogram sensitive to a
  drifting period.
- {func}`cuperiod.phase_track` / {func}`cuperiod.fit_ephemeris` — **measure**:
  an automated O−C with per-segment uncertainties, and the same quadratic
  ephemeris fit for your own timing lists.
- {func}`cuperiod.sliding_periodogram` / {func}`cuperiod.wwz` — **explore**:
  time-resolved maps for period wander that is not a clean secular trend.
- {func}`cuperiod.eddington_plakidis` — **vet**: the classical test that a
  smooth-looking O−C is not just an accumulating random walk.

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

## References

- Sterken, C. 2005, ASP Conf. Ser. 335, 3 — the O−C method, its ambiguities, and its abuses.
- Mikulášek, Z., et al. 2012, arXiv:1212.5527 — the exact constant-dP/dt phase.
- Eddington, A. S., & Plakidis, S. 1929, MNRAS 90, 65 — random period fluctuations.
- Foster, G. 1996, AJ 112, 1709 — the Weighted Wavelet Z-transform.
- Templeton, M. R., Mattei, J. A., & Willson, L. A. 2005, AJ 130, 776 — Mira period wander with the WWZ.
- Katz, M. L., et al. 2021, MNRAS 503, 2665 — the GPU (P, dP/dt) conditional-entropy search.
- Murphy, S. J., et al. 2014, MNRAS 441, 2515 — segment phase tracking as time delays.
- Patra, K. C., et al. 2017, AJ 154, 4; Yee, S. W., et al. 2020, ApJ 888, L5 — WASP-12b's orbital decay.
- Winn, J. N., & Stefánsson, G. 2025, PSJ 6, 300 — light-travel time masquerading as decay.
- Eastman, J., Siverd, R., & Gaudi, B. S. 2010, PASP 122, 935 — time standards.
- Deeg, H. J. 2021, Galaxies 9, 1 — modern eclipse-minimum timing errors.

---

Next: {doc}`multiband` — combining several filters.
