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
  - 2.1e-11
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
  - 1.4e-02
  - 100%
  - Dworetsky 1983
  - 5.2e-11
* - MHAOV
  - 72
  - 3.6e-07
  - 100%
  - Sch.-Czerny
  - 8.0e-05
* - TLS
  - 22
  - 8.1e-10
  - 100%
  - —
  - —
```

*parity* is the worst-case relative difference between the CPU and GPU statistic over all
stars (GLS/MHAOV GPU paths are single precision, hence ~1e-6/1e-7; the rest are double).
String-Length's looser parity is isolated trial frequencies where near-equal phases sort
in a different order on the GPU — **the recovered period is unaffected** (*same P* = 100%).

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
  - 96%
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

Single light curve (~900 points), NVIDIA RTX 5070 Ti vs the CPU backends:

```{list-table}
:header-rows: 1
:widths: 16 18 14 14 16 22

* - Method
  - CPU backend
  - CPU
  - GPU
  - vs reference
  - GPU speed-up
* - GLS
  - finufft
  - 0.019 s
  - 0.007 s
  - 3× astropy
  - ~3×
* - BLS
  - numba
  - 0.187 s
  - 0.091 s
  - **20× astropy**
  - ~2×
* - PDM
  - numpy
  - 1.79 s
  - 0.010 s
  - 4× PyAstronomy
  - **178×**
* - CE
  - numpy
  - 0.53 s
  - 0.012 s
  - —
  - 45×
* - String-Length
  - numpy
  - 0.92 s
  - 0.023 s
  - —
  - 39×
* - MHAOV
  - numpy
  - 3.56 s
  - 0.111 s
  - —
  - 32×
* - TLS
  - numpy
  - 4.49 s
  - 0.042 s
  - —
  - 106×
```

Two takeaways:

- cuPeriod's **CPU** path already beats every reference tool it was checked against — most
  dramatically BLS, where the multicore `numba` box search is **20× faster than astropy's
  compiled `BoxLeastSquares`** while matching it to floating point.
- The **GPU** delivers 30–180× on the methods whose CPU path is plain numpy (PDM, CE,
  String-Length, MHAOV, TLS), and a smaller single-curve margin on GLS/BLS — whose CPU
  backends are already specialized. The GPU's decisive win for GLS/BLS is at **catalog
  scale**.

## 4. Batch throughput & transits

- **Batch:** ~620 light curves/second on one GPU for GLS on short survey curves —
  over 2.2 million light curves/hour.
- **Kepler transits (TLS):** on 12 confirmed KOIs with a blind 0.5–12 d search, cuPeriod
  recovers 10/12 within 2% (the `transitleastsquares` reference recovers 11/12); both miss
  only the shallowest, where a blind search aliases — a shared, honest failure mode, not a
  backend defect. On the CPU-timed subset the GPU is a median **109×** faster at
  CPU↔GPU agreement ≤ 2e-14.

See the
[full report](https://github.com/tjayasinghe/cuPeriod/blob/main/benchmarks/REPORT.md)
for the figures, per-KOI detail, and reproduction commands.
