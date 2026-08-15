# Validation & benchmarks

cuPeriod ships a reproducible validation + benchmark suite (`benchmarks/` in the
repository). This page summarizes the headline results; the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md),
with figures, lives in the repo and regenerates from bundled data with no network access.

**Validation data** — 126 real ASAS-SN *g*-band light curves across six variability classes
(eclipsing binaries, RR Lyrae, Cepheids, δ Scuti, long-period and rotational variables),
each with an established VSX literature period. The set combines an original 72-star
curated core with a 54-star extension (`benchmarks/dataset/download_extension.py`) added to
broaden coverage of harder classes. TLS is validated on 12 confirmed Kepler KOIs. The
curves and their literature periods ship with the suite, so §1–2 and §4 are fully
reproducible offline.

§1, §2, §4 and §5 below cover the seven methods that go through the single-band validation
suite. SuperSmoother, new in this release, is pinned in the unit tests against the reference
`supersmoother` package and `gatspy` ({doc}`guide/methods`), joins the performance sweep
in §3, and is validated on real data alongside every other multi-band method in §7.

## 1. Numerical validation

Every method runs the **same grid** through cuPeriod's CPU and GPU backends and through an
independent reference implementation. Two checks: CPU↔GPU **parity** (the backends must
agree to round-off) and cuPeriod↔**reference** (must match an established implementation).

```{list-table}
:header-rows: 1
:widths: 18 10 18 12 22 20

* - Method
  - N
  - CPU↔GPU parity
  - same P
  - reference
  - ref. agreement
* - GLS
  - 126
  - 2.1e-06
  - 100%
  - astropy LS
  - 8.0e-10
* - BLS
  - 126
  - 1.2e-11
  - 100%
  - astropy BLS
  - 1.1e-09
* - PDM
  - 126
  - 2.0e-10
  - 100%
  - PyAstronomy
  - r ≥ 0.929
* - CE
  - 126
  - 1.7e-15
  - 100%
  - Graham 2013
  - 3.1e-15
* - String-Length
  - 126
  - 6.2e-15
  - 100%
  - Dworetsky 1983
  - 4.3e+00 †
* - MHAOV
  - 126
  - 1.1e-05
  - 100%
  - Sch.-Czerny
  - 1.4e-04
* - TLS
  - 28
  - 9.3e-10
  - 100%
  - —
  - —
```

*parity* is the worst-case relative difference between the CPU and GPU statistic over all
stars (GLS/MHAOV GPU paths are single precision, hence ~1e-6 and ~1e-5; the rest — String-Length
now included, via a stable phase sort on every backend — are double). † String-Length's
*reference* agreement shows one isolated outlier: a heavily phase-tied star where the
textbook reference breaks ties with an unstable sort. Its correlation stays ≈ 1 (median
\|Δ\| ≈ 6e-12) and **the recovered period is unaffected** (*same P* = 100%).

**Torch backend.** The portable `backend="torch"` path was validated against cuPeriod's
own CPU backend on the same grid, on an NVIDIA device (`torch:cuda`):

```{list-table}
:header-rows: 1
:widths: 22 14 20 22

* - Method
  - N
  - max rel. diff vs CPU
  - identical peak vs CPU
* - GLS
  - 126
  - 1.4e-06
  - 100%
* - BLS
  - 126
  - 1.3e-11
  - 100%
* - PDM
  - 126
  - 2.5e-10
  - 100%
* - CE
  - 126
  - 1.8e-15
  - 100%
* - String-Length
  - 126
  - 6.0e-15
  - 100%
* - MHAOV
  - 126
  - 2.2e-06
  - 100%
* - TLS
  - 28
  - 1.1e-09
  - 100%
```

All seven methods in the validation suite pick the identical best period as the CPU backend
on every validated star. The same torch code path also runs on Apple (`mps`) and Intel (`xpu`) devices, but
those were not exercised in this report (see {doc}`guide/backends`).

## 2. Period recovery on real light curves

```{list-table}
:header-rows: 1
:widths: 30 24 24

* - Method
  - harmonic-aware
  - exact (≤ 2%)
* - GLS
  - 89%
  - 69%
* - BLS
  - 88%
  - 81%
* - PDM
  - 88%
  - 72%
* - CE
  - 90%
  - 72%
* - String-Length
  - 89%
  - 51%
* - MHAOV
  - 88%
  - 67%
* - TLS
  - 96%
  - 86%
```

*harmonic-aware* accepts the method-appropriate fold ambiguity (e.g. Fourier methods
recover P/2 for contact binaries); *exact* requires the VSX literature period itself within
2%. Wilson 95% confidence intervals per method (n=126, or n=28 for TLS) are given in the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md).

**Curated core vs. extension.** Pooled across all frequency methods, harmonic-aware
recovery is 99% [97–100%] on the original 72-star curated core but 75% [70–80%] on the
54-star extension. The extension deliberately adds harder classes — spot-evolving rotators
(BY Dra/RS CVn, whose starspot-driven period drifts between seasons) and semiregular/Mira
variables (whose pulsation cycle wanders relative to a single catalogue period) — so the
pooled 126-star rate is a more realistic field estimate than the curated core's rate alone.
This is a sampling effect, not a code regression: CPU, GPU, and torch backends agree to
round-off on every star in both groups (§1).

## 3. Performance

Single light curve (~900 points), NVIDIA RTX 5070 Ti vs. the `[fast]`-extra CPU backends,
on a 32-thread machine:

```{list-table}
:header-rows: 1
:widths: 15 14 11 11 13 17 16

* - Method
  - CPU backend
  - CPU
  - GPU
  - torch:cuda
  - vs reference
  - GPU speed-up
* - GLS
  - finufft
  - 0.013 s
  - 0.005 s
  - 0.009 s
  - 3× astropy
  - ~2×
* - BLS
  - numba
  - 0.171 s
  - 0.093 s
  - 0.396 s
  - **20× astropy**
  - ~2×
* - PDM
  - numba
  - 0.003 s
  - 0.005 s
  - 0.009 s
  - **>2,000× PyAstronomy**
  - ~0.6× (GPU slower)
* - CE
  - numba
  - 0.003 s
  - 0.004 s
  - 0.021 s
  - —
  - ~0.7× (GPU slower)
* - String-Length
  - numba
  - 0.043 s
  - 0.012 s
  - 0.009 s
  - —
  - ~4×
* - MHAOV
  - numba
  - 0.026 s
  - 0.038 s
  - 0.032 s
  - —
  - ~0.7× (GPU slower)
* - SuperSmoother
  - numba
  - 0.099 s
  - 0.284 s
  - 0.222 s
  - —
  - ~0.3× (GPU slower)
* - TLS
  - numba
  - 0.132 s
  - 0.072 s
  - 2.205 s
  - —
  - ~2×
```

Two takeaways:

- cuPeriod's **CPU** path already beats every reference tool it was checked against — most
  dramatically PDM (a multicore `numba` port over PyAstronomy's pure-Python `pyPDM`) and
  BLS, where the multicore `numba` box search is **20× faster than astropy's compiled
  `BoxLeastSquares`** while matching it to floating point.
- **The multicore numba CPU tier changes the GPU calculus for a single curve.** Now that
  PDM, CE, String-Length, MHAOV, SuperSmoother, and TLS all default to numba kernels rather
  than plain numpy, the GPU's single-curve margin is a modest win (String-Length ~4×,
  GLS ~2×), a near-wash (BLS and TLS, ~1.8×), or the GPU is actually a touch *slower* than
  the CPU (PDM/CE/MHAOV ~0.6-0.7×, SuperSmoother ~0.3×) — fixed dispatch and host↔device
  transfer overhead no longer amortizes once the CPU kernel itself runs in low single-digit
  milliseconds. The scaling sweep (up to 30k points, a 100k-frequency grid — see the full
  report) adds two nuances: GLS holds a ~4-5× GPU edge at every grid size since its CPU
  path is finufft, not numba, and v1.2's auto-sized batching lifted MHAOV's GPU from ~5×
  slower to a near-wash across the whole range, while SuperSmoother's GPU only approaches
  parity once curves reach several thousand points. The GPU's clear, reproducible win is
  now **catalog throughput**, not single-curve latency.

:::{note}
The **`torch:cuda`** column is the portable PyTorch backend on the *same* RTX 5070 Ti —
validated on NVIDIA hardware, where every method above matches the CPU reference to
round-off. It is competitive with the cupy fast paths on the frequency methods — here it
even edges them out on String-Length, MHAOV, and SuperSmoother — while staying well behind
on BLS/TLS, whose cupy `RawKernel`s are hand-tuned. Its real value is reaching **AMD,
Intel, and Apple** GPUs the CUDA paths can't (those share the same code and are
CPU-validated; see {doc}`guide/backends`). On the CPU it is correct but not the speed
champion — finufft (GLS) and the numba kernels win there.
:::

## 4. Injection–recovery sensitivity

§1–2 validate against real, bright, well-established stars — a favourable regime. A
complementary synthetic sweep (`benchmarks/injection_recovery.py`) injects a signal of
tunable amplitude onto real ASAS-SN observation cadences (so the irregular sampling and
seasonal gaps are realistic) and scores recovery with the same harmonic-aware 2% tolerance,
40 trials per method × signal × SNR cell:

```{list-table}
:header-rows: 1
:widths: 20 16 16 16

* - Signal
  - Method
  - SNR=1
  - SNR=6
* - Sinusoid
  - GLS / PDM / MHAOV
  - 95–100%
  - 98–100%
* - Sinusoid
  - CE / String-Length
  - 98–100%
  - 100%
* - Eclipse
  - BLS
  - 95%
  - 100%
* - Eclipse
  - PDM / CE / String-Length
  - 10–72%
  - 75–100%
* - Transit
  - BLS / TLS
  - 90–92%
  - 100%
```

Every method reaches ≥95% recovery by moderate-to-high SNR for the signal it targets;
isolated cells in the low-to-mid 90s are consistent with one or two alias near-misses at
40 trials per cell. The one method that plateaus well below 100% even at the highest
tested SNR is String-Length on the narrow eclipse model (~78%) — a method–signal mismatch,
not a bug: its rank-based statistic is comparatively insensitive to narrow, low duty-cycle
dips, and a box-fitting method (BLS) is the appropriate tool for narrow eclipses/transits.
Full per-SNR grid and figure in the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md#5--injection–recovery-sensitivity).

## 5. Batch throughput & transits

- **Batch:** up to 574 light curves/second on one GPU for GLS on short survey curves
  (>2 million/hour) — a *single-batch* rate that includes one-off worker-pool spin-up
  (process spawn + per-worker CUDA context); a warmed pool sustains a higher rate over many
  chunks. On the same 32-thread machine, the CPU process pool keeps pace with the GPU for
  the numba-tier methods at the batch sizes tested (PDM: ~1.0× at n=256 and n=1024); GLS is
  the one method with a consistent GPU edge at batch scale too (~1.3-1.4× up to n=1024).
  Expect a wider GPU margin on a narrower CPU, or at larger batch sizes than swept here.
- **Kepler transits (TLS):** on 12 confirmed KOIs with a blind 0.5–12 d search, cuPeriod
  recovers 10/12 within 2% (the `transitleastsquares` reference recovers 11/12); both
  struggle only on the shallowest transits, where a blind search aliases — a shared,
  honest failure mode, not a backend defect. On the CPU-timed subset the GPU is a median
  **~2×** faster at CPU↔GPU agreement ≤ 2.1e-14 (the numba CPU tier narrowed what used
  to be a much larger single-curve gap).

## 6. Multi-band recovery at survey cadence

`benchmarks/multiband_recovery.py` (fully synthetic, offline) measures what joint
multi-band fitting buys at sparse survey cadence: 300 faint RRab-like stars per cell
(0.2 mag noise), six bands with WFD-like epoch shares over a 3-year window, recovery =
top period within 1% of truth with no harmonic credit.

| strategy | 30 epochs | 60 epochs | 120 epochs |
|---|---|---|---|
| best single band (*r*) | 0.0% | 37.3% | 97.3% |
| any single band | 0.0% | 49.3% | 99.0% |
| multi-band `perband` (0,1) | 17.7% | 97.0% | 100.0% |
| multi-band `flex` (1,1) | 20.0% | 97.0% | 100.0% |
| multi-band `offsets` (1,0) | **81.7%** | **99.7%** | 100.0% |

At 30 total epochs — roughly Rubin's first year for one band's worth of visits spread
over six filters — single-band search recovers nothing and the shared-phase `offsets`
model recovers 82%. The models that grant each band its own phase (`perband`, `flex`)
sit far below it at this sparsity: pooling phase information is what buys the
recovery, which is why `offsets` is the default. Model definitions and guidance live
in {doc}`guide/multiband`; the native `offsets` path is also ~400× faster than
astropy's `LombScargleMultiband` on a 6-band, 200k-frequency search (55 s → 0.13 s,
CPU).

## 7. Multi-band validation on real data

§6 is simulated; this is the same question asked of **real** photometry with known answers.
`benchmarks/multiband_real.py` runs every multi-band method over 100 SDSS Stripe 82 RR Lyrae
from Sesar et al. (2010) — 80 RRab and 20 RRc, real SDSS *ugriz* cadence with ~55 epochs per
band over a ~3200 d baseline, each with a literature period from that paper's ~10-year
solution. These are the stars VanderPlas & Ivezić (2015) built the multiband periodogram on,
so the shared-phase model that ships as cuPeriod's default `offsets` is being checked on its
home ground. Every star gets one identical blind search: periods 0.15–1.2 d at 5 samples per
Rayleigh width (~97,000 trial frequencies), all methods at default settings. *Strict* means
the top period is within 1% of the literature value with no harmonic credit; *harmonic-aware*
accepts a small-integer harmonic within 2%.

```{list-table}
:header-rows: 1
:widths: 30 16 22 22

* - Multi-band model
  - strict
  - harmonic-aware
  - median CPU s/star
* - GLS `offsets` (1,0)
  - 76%
  - 80%
  - 0.070
* - GLS `perband` (0,1)
  - 78%
  - 81%
  - 0.112
* - GLS `flex` (1,1)
  - 78%
  - 81%
  - 0.367
* - PDM
  - **93%**
  - 94%
  - 0.011
* - CE
  - 85%
  - 90%
  - 0.023
* - String-Length
  - **93%**
  - **97%**
  - 0.036
* - MHAOV
  - 83%
  - 84%
  - 0.538
* - SuperSmoother
  - 85%
  - 96%
  - 0.296
* - BLS
  - 22%
  - 34%
  - 3.024
```

Single-band GLS, one filter at a time, is the baseline: 72–78% strict per band (*z* worst,
*r* best) and 92% for *any* single band — the optimistic bound that counts a star as recovered
if any of the five searches lands on the right period, which in practice you cannot know.

Two regimes, one conclusion. On curves this well sampled (~280 points across five bands) the
pooled fold statistics lead: PDM and String-Length reach 93% strict and String-Length 97%
harmonic-aware, because a dense fold exploits the whole non-sinusoidal RRab shape while the
single-harmonic GLS models stay alias-limited. The three GLS models are indistinguishable here
(76–78%) and no better than the best single band, the *opposite* of §6's sparse cadence where
`offsets` recovers 82% against ≤20% for the flexible models. Dense per-band data reward shape;
sparse data reward parsimony — the fold methods were not run at sparse cadence, so §6 remains
the guidance for a survey-cadence search. SuperSmoother's 85% → 96% gap is the documented
integer-multiples family: of 21 fold-family picks that are harmonic but not strict, 11 sit at
exactly 2P and 5 at 3P, and its strict rate is 55% on the near-sinusoidal RRc against 92.5% on
RRab, since a fold at twice the period stays coherent. Read the shortest member of a near-tied
family, or let {func}`~cuperiod.alias_diagnostics` arbitrate. Of the 163 non-harmonic misses
across all models, 54% fall on the ±1 or ±2 cycle/day loci — the ground-based window function,
not noise. BLS's 22% is expected and not a defect: it fits transit shapes, and is reported for
completeness rather than recommended for RR Lyrae. Model definitions and guidance live in
{doc}`guide/multiband`.

To reproduce: `benchmarks/dataset/download_s82_rrlyrae.py` builds the bundle from the
astroML-data mirror (needs network, one time only), then `benchmarks/multiband_real.py` runs
offline — the ~390 KB bundle is committed with the suite.

See the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md)
for the figures, per-band and per-subtype breakdowns, per-KOI detail, and reproduction
commands.
