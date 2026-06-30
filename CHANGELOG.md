# Changelog

All notable changes to cuPeriod are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multi-vendor GPU support via PyTorch and the Python array API.** All seven
  period-search methods (GLS, BLS, PDM, CE, String-Length, MHAOV, TLS) gain a portable
  `torch` backend that runs on AMD (ROCm), Intel (XPU), and Apple (MPS) GPUs as well as a
  real CPU path — so the accelerated code is no longer NVIDIA-only, and works even with no
  GPU at all. Select it with `backend="torch"` (or `"torch:cpu"`, `"torch:cuda"`,
  `"torch:mps"`, `"torch:xpu"`); `backend="auto"` now reaches a torch GPU on non-NVIDIA
  machines after the cufinufft/cupy fast paths.
  - GLS adds a NUFFT-free direct trig-sum path (the portable formulation; cufinufft
    remains the NVIDIA fast path).
  - BLS, PDM, CE, String-Length, MHAOV, and TLS run their vectorized kernels through the
    array-API namespace; the cupy `RawKernel`s (BLS/PDM/CE/TLS), numba (BLS), and finufft
    (GLS) remain the fast paths where present.
  - New `device` and `precision` settings: `precision="auto"` is float64 everywhere it is
    supported and float32 only where the device forces it (Apple MPS cannot do float64);
    an explicit `precision="float64"` on MPS raises rather than silently downgrading.
- `array-api-compat` is now a dependency; install the portable accelerator with the
  `[torch]` extra (`pip install 'cuperiod[torch]'`).

### Known limitations

- **Non-NVIDIA GPU numerics are written-to-spec and CPU-validated, not yet hardware-
  verified.** The torch CUDA/ROCm/MPS/XPU paths share the array-API body that is parity-
  tested on the CPU torch device; on-device parity self-skips (`requires_torch_gpu`) until
  such hardware is available.
- **No fp64 capability probe on Intel XPU.** `precision="auto"` resolves to float64 on an
  XPU; a device without native float64 will error at compute time rather than falling back
  to float32. Pass `precision="float32"` explicitly on such a device.
- **The torch GPU path does not auto-shrink to small VRAM.** A large period×bin grid on a
  small consumer GPU can raise an out-of-memory error; reduce `batch_periods`.
- **Tie-broken best-fit *extras* may differ across devices.** Where an `argmax` lands on an
  exact tie (e.g. BLS `transit_time`, TLS `t0`/`duration` at non-transit periods), the
  chosen index is device-dependent; the periodogram power and best period are unaffected.

## [1.0.0] — 2026-06-29

First public release.

### Added

- Seven period-search methods, each with an optimized CPU backend and a CUDA GPU backend:
  - **GLS** — generalized (floating-mean) Lomb–Scargle via non-uniform FFT
    (finufft / cufinufft).
  - **BLS** — box least squares, with a multicore `numba` CPU search (`[fast]` extra) and
    a CUDA kernel.
  - **PDM** — phase dispersion minimization (Stellingwerf).
  - **CE** — conditional entropy (Graham et al. 2013).
  - **String-Length** — Dworetsky / Lafler–Kinman.
  - **MHAOV** — multiharmonic analysis of variance (Schwarzenberg-Czerny).
  - **TLS** — transit least squares, a limb-darkened matched filter.
- Single-light-curve entry point `cuperiod.periodogram` and the batch sibling
  `cuperiod.batch_periodograms` (CPU process pool or GPU, resumable Parquet/CSV output).
- Multi-band joint modelling for GLS, BLS, and MHAOV.
- Frictionless inputs: arrays, dicts, pandas/astropy/pyarrow tables, and
  CSV/ECSV/FITS/Parquet files, with case-insensitive, **survey-aware** column
  auto-detection (`ColumnMap` — ASAS-SN, ASAS-3, ATLAS, CRTS/CSS, ZTF, Pan-STARRS, LSST,
  TESS, Kepler, Gaia, MACHO) and domain-aware error pairing, plus automatic
  magnitude/flux handling.
- N-best-period peak finding with alias- and harmonic-aware selection.
- `cuperiod` command-line interface: `run`, `batch`, `methods`, `gpu-info`, `grid-info`.
- Per-method settings models with `CUPERIOD_<METHOD>_<FIELD>` environment overrides, and
  GPU worker auto-sizing.
- Full Sphinx documentation (hosted on Read the Docs), a worked-example Jupyter notebook
  (`examples/cuperiod_tour.ipynb`), and a reproducible validation + benchmark suite under
  `benchmarks/`.

### Robustness

- Peak selection returns the true peak even when it sits at a frequency-grid edge (a
  signal whose period is comparable to the observing baseline), and never reports a
  non-finite period or NaN-power sample.
- Settings reject transposed frequency / period / duration-fraction bounds at
  construction; the CLI's `--out` JSON is always standard JSON (no `NaN`/`Infinity`).
- GPU kernels opt into larger shared memory where the device allows, and otherwise raise
  a clear error naming the setting to reduce — instead of a raw CUDA driver error.
- Batch sinks are correct across re-runs: a file sink is keyed by `(key, method)`, a
  directory sink refuses a mismatched `chunk_size` on resume, and a CSV sink asked to
  store raw spectra fails loudly rather than dropping the columns.
- The batch process pool uses the `spawn` start method on every platform, so a CPU/GPU
  pool no longer deadlocks on Linux (the default `fork` copies parent native thread pools
  / CUDA contexts into the workers).
- Method-name lookup ignores case and non-alphanumeric separators, so `"String-Length"`,
  `"StringLength"` and `"STRINGLENGTH"` all resolve (as documented).

### Validated

- Every method matches an established reference (astropy `LombScargle` / `BoxLeastSquares`,
  PyAstronomy, and textbook implementations) to floating-point round-off, with CPU↔GPU
  parity, on 72 real ASAS-SN light curves across six variability classes and on confirmed
  Kepler transits.

[1.0.0]: https://github.com/tjayasinghe/cuPeriod/releases/tag/v1.0.0
