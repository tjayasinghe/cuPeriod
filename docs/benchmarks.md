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
stars (GLS/MHAOV GPU paths are single precision, hence ~1e-6/1e-7; the rest — String-Length
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

All seven methods pick the identical best period as the CPU backend on every validated
star. The same torch code path also runs on Apple (`mps`) and Intel (`xpu`) devices, but
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
  - 0.015 s
  - 0.005 s
  - 0.008 s
  - 2× astropy
  - ~3×
* - BLS
  - numba
  - 0.188 s
  - 0.092 s
  - 0.392 s
  - **18× astropy**
  - ~2×
* - PDM
  - numba
  - 0.003 s
  - 0.005 s
  - 0.011 s
  - **>2,000× PyAstronomy**
  - ~0.6× (GPU slower)
* - CE
  - numba
  - 0.003 s
  - 0.004 s
  - 0.009 s
  - —
  - ~0.8× (GPU slower)
* - String-Length
  - numba
  - 0.043 s
  - 0.012 s
  - 0.009 s
  - —
  - ~4×
* - MHAOV
  - numba
  - 0.025 s
  - 0.135 s
  - 0.114 s
  - —
  - ~0.2× (GPU slower)
* - TLS
  - numba
  - 0.150 s
  - 0.043 s
  - 2.110 s
  - —
  - ~4×
```

Two takeaways:

- cuPeriod's **CPU** path already beats every reference tool it was checked against — most
  dramatically PDM (a multicore `numba` port over PyAstronomy's pure-Python `pyPDM`) and
  BLS, where the multicore `numba` box search is **18× faster than astropy's compiled
  `BoxLeastSquares`** while matching it to floating point.
- **The multicore numba CPU tier changes the GPU calculus for a single curve.** Now that
  PDM, CE, String-Length, MHAOV, and TLS all default to numba kernels rather than plain
  numpy, the GPU's single-curve margin is modest (BLS/String-Length/TLS, ~2-4×), a wash
  (CE), or the GPU is actually a touch *slower* than the CPU (PDM, MHAOV) — kernel-launch
  and host↔device transfer overhead no longer amortizes once the CPU kernel itself runs in
  low single-digit milliseconds. That holds across the benchmark's scaling sweep too (up to
  30k points, a 100k-frequency grid — see the full report). GLS is the exception, with a
  consistent ~3× GPU edge since its CPU path is finufft, not numba. The GPU's clear,
  reproducible win is now **catalog throughput**, not single-curve latency.

:::{note}
The **`torch:cuda`** column is the portable PyTorch backend on the *same* RTX 5070 Ti — now
validated on NVIDIA hardware, where all seven methods match the CPU reference to round-off.
It is competitive with the cupy fast paths on PDM/CE/String-Length/MHAOV and slower on
BLS/TLS, whose cupy `RawKernel`s are hand-tuned. Its real value is reaching **AMD, Intel,
and Apple** GPUs the CUDA paths can't (those share the same code and are CPU-validated; see
{doc}`guide/backends`). On the CPU it is correct but not the speed champion — finufft (GLS)
and the numba kernels win there.
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
  - 42–98%
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
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md#4--injection–recovery-sensitivity).

## 5. Batch throughput & transits

- **Batch:** up to 587 light curves/second on one GPU for GLS on short survey curves
  (>2.1 million/hour) — a *single-batch* rate that includes one-off worker-pool spin-up
  (process spawn + per-worker CUDA context); a warmed pool sustains ~490 lc/s over many
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

See the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md)
for the figures, per-KOI detail, and reproduction commands.
