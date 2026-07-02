# Validation & benchmarks

cuPeriod ships a reproducible validation + benchmark suite (`benchmarks/` in the
repository). This page summarizes the headline results; the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md),
with figures, lives in the repo and regenerates from bundled data with no network access.

**Validation data** — 72 real ASAS-SN *g*-band light curves across six variability classes
(eclipsing binaries, RR Lyrae, Cepheids, δ Scuti, long-period and rotational variables),
each with an established VSX literature period. TLS is validated on confirmed Kepler KOIs.
The curves and their literature periods ship with the suite, so §1–2 and §4 are fully
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
  - 72
  - 2.1e-06
  - 100%
  - astropy LS
  - 3.4e-10
* - BLS
  - 72
  - 1.2e-11
  - 100%
  - astropy BLS
  - 1.1e-09
* - PDM
  - 72
  - 3.5e-11
  - 100%
  - PyAstronomy
  - r ≥ 0.948
* - CE
  - 72
  - 1.7e-15
  - 100%
  - Graham 2013
  - 0.0e+00
* - String-Length
  - 72
  - 7.7e-16
  - 100%
  - Dworetsky 1983
  - 1.7e+00 †
* - MHAOV
  - 72
  - 1.5e-07
  - 100%
  - Sch.-Czerny
  - 1.4e-04
* - TLS
  - 22
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

## 2. Period recovery on real light curves

```{list-table}
:header-rows: 1
:widths: 30 24 24

* - Method
  - harmonic-aware
  - exact (≤ 2%)
* - GLS
  - 99%
  - 75%
* - BLS
  - 100%
  - 93%
* - PDM
  - 99%
  - 81%
* - CE
  - 99%
  - 76%
* - String-Length
  - 99%
  - 60%
* - MHAOV
  - 99%
  - 75%
* - TLS
  - 100%
  - 95%
```

*harmonic-aware* accepts the method-appropriate fold ambiguity (e.g. Fourier methods
recover P/2 for contact binaries); *exact* requires the VSX literature period itself within
2%.

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

## 4. Batch throughput & transits

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
  **~1.6×** faster at CPU↔GPU agreement ≤ 2.1e-14 (the numba CPU tier narrowed what used
  to be a much larger single-curve gap).

See the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md)
for the figures, per-KOI detail, and reproduction commands.
