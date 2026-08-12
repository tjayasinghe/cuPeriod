# Changelog

All notable changes to cuPeriod are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Automated, uncertainty-aware pre-whitening for classical pulsators**
  ({func}`cuperiod.prewhiten`). Frequency analysis of δ Scuti, γ Doradus and SPB stars
  has funnelled through interactive Period04-style sessions one star at a time; this
  runs the whole loop — amplitude spectrum, peak selection, simultaneous non-linear
  re-fit of every component, significance test — end to end, and makes every judgement
  call an explicit, recorded setting. A `PreWhitenResult` carries the ranked components
  with their uncertainties, the residuals and their spectrum, the fit statistics, and
  the reason the run stopped.
- **A batch-capable GPU/NUFFT amplitude spectrum** (`cuperiod.amplitude_spectrum`,
  `SpectrumEngine`) on the cufinufft / finufft / torch / numpy backends. The terms of
  the least-squares normal equations that depend only on the observation times are
  computed once and cached, so each pre-whitening iteration costs a *single* transform:
  a 50-frequency solution needs ~52 transforms rather than ~150. A 20 000-point TESS
  sector with 15 modes takes about a second on a laptop CPU.
- **Principled stopping criteria**, combinable and always reported in `stop_reason`:
  the Breger et al. (1993) signal-to-noise ratio, the Baluev false-alarm probability,
  a ΔBIC improvement threshold, and an absolute amplitude floor. After the final
  simultaneous polish the solution is re-checked and components that no longer pass are
  pruned, so every reported frequency satisfies the criterion it was admitted by.
- **Error propagation** with three estimators (`uncertainty=`): the linearised
  least-squares covariance of the joint fit (the default, and the only one that accounts
  for correlations between close frequencies), the classical Montgomery & O'Donoghue
  (1999) formulae, and a residual bootstrap. All optionally inflated by the
  Schwarzenberg-Czerny (1991) correlation factor. Phases are referenced to the weighted
  mean epoch, at which a phase is uncorrelated with its own frequency — validated
  against Monte Carlo: reported 1-sigma errors match the realised scatter to within
  ~5% in frequency, amplitude, and phase.
- **Combination-frequency identification**
  ({func}`cuperiod.identify_combinations`) with uncertainty-aware tolerances — a match
  must fall within `max(3σ, 0.25/T)` of the *propagated* prediction — and a
  chance-coincidence rate reported per identification, so a spurious match is visible as
  such. `PreWhitenResult.independent()` returns the candidate independent-mode list.
- **g-mode period-spacing tools**: a comb scan over trial spacings
  ({func}`cuperiod.spacing_spectrum`, unaffected by missing radial orders, and immune to
  the sub-multiple ambiguity that makes a naive scan report ΔΠ/2), extraction of the
  longest *tilted* series `ΔP(P) = a + bP` bridging missing orders
  ({func}`cuperiod.find_period_spacing`), échelle coordinates
  ({func}`cuperiod.echelle`), and the buoyancy radius Π₀
  ({func}`cuperiod.buoyancy_radius`).
- **Batch pre-whitening** ({func}`cuperiod.batch_prewhiten`) over the existing CPU/GPU
  worker pools, writing one row per extracted component to Parquet/CSV with resumable
  directory sinks.
- **CLI**: `cuperiod prewhiten` (with `--spacing`, JSON/CSV/npz output) and
  `cuperiod batch-prewhiten`.
- **The spectral window as a first-class diagnostic**: `SpectrumEngine.window()`,
  {func}`cuperiod.spectral_window`, and `PreWhitenResult.window` expose `|W(f)|` of the
  sampling — the alias-lobe pattern every real peak is convolved with — at zero extra
  transforms (its sums are already part of the cached normal equations). The GUI
  spectrum view gains a *window* overlay toggle (scaled to the tallest peak,
  Period04-style) and `cuperiod prewhiten --save-spectrum` writes it into the `.npz`.
- **A native Baluev (2008) false-alarm probability** ({func}`cuperiod.baluev_fap`)
  matching astropy's `false_alarm_probability(method="baluev")` to machine precision on
  centred times — and staying accurate on raw Julian dates, where the one-pass time
  variance loses ~11 digits. Pre-whitening no longer imports `astropy.timeseries`,
  which was ~1 s of first-solution latency in the CLI/GUI and per batch worker.
- **GUI**: an **Analysis** picker switches the desktop app between *Periodogram* and
  *Pre-whitening* without disturbing anything else — same inputs, same spectrum, phased
  and raw views, same source browser, same off-thread compute and result caching. In
  pre-whitening mode the spectrum shows the amplitude spectrum with the residual
  spectrum overlaid and the components marked, a **Frequencies** dock lists them with
  uncertainties and S/N (click to fold, right-click to export), and a **Period spacing**
  dock scans for a regular spacing and draws the échelle diagram. The settings form is
  generated from `PreWhitenSettings` by the existing machinery, so every knob is exposed
  with no bespoke widgets.

### Fixed

- `cuperiod.gui.models.ResultCache` is now generic over its value type, so the app keeps
  one cache per analysis and switching back and forth is instant.
- **The automatic pre-whitening band could sit entirely below a δ Scuti star.** The
  default topped out at the median-gap pseudo-Nyquist (with `nyquist_factor=1`), which
  for nightly ground-based sampling is ~0.5–2.5 cycles/day — so on the bundled ASAS-SN
  HADS demo (P = 0.0898 d, f = 11.14 c/d) the extraction fitted the *daily aliases* of
  the real signal (P = 0.123 d, residual rms 0.245). The auto band is now
  `max(pseudo-Nyquist × nyquist_factor, 50 c/d)` with `nyquist_factor=5` (matching the
  periodogram methods), exposed as
  {func}`cuperiod.prewhiten.default_maximum_frequency`, and the GUI's auto value uses
  the same helper. The demo star now yields P = 0.089757 d — the VSX period to the
  last digit — with its 2f, 3f, 4f harmonics extracted and combination-labelled
  (residual rms 0.073).
- **Bootstrap frequency errors were exactly zero for every component but the newest.**
  The engine handed its per-iteration refinement policy (default `"last"`, which pins
  all established frequencies) to the bootstrap's replicate fits, so their scatter
  collapsed. Replicates now always sweep every frequency, boxed by the same
  per-frequency bounds as the fit they characterise.
- `batch_prewhiten` forces `store_spectra=False`: catalogue rows never carry spectra,
  so keeping them only made each worker hold megabytes of grid arrays per star.

## [1.1.0] - 2026-07-08

### Performance

- **Multicore `numba` CPU kernels for PDM, CE, String-Length, MHAOV, and TLS** (BLS
  already had one). With the `[fast]` extra installed, `backend="cpu"`/`"auto"` now
  resolve to them; warm-kernel speedups over the vectorized numpy paths on a 3k-point
  curve: PDM ~300x, CE ~135x, MHAOV ~57x, TLS ~53x, String-Length ~23x, with parity to
  the numpy results at or below ~1e-11.
- **MHAOV is rewritten around harmonic trig sums.** Every entry of the normal equations
  is analytically a trig sum, so the Gram matrix is now assembled from `C_m`/`S_m`
  sums computed with the Chebyshev recurrence (one `cos`/`sin` evaluation regardless of
  the harmonic order) instead of materializing the `(F, N, 2H+1)` design tensor:
  ~2.7-3.2x faster on numpy/cupy/torch alike, an order of magnitude less transient
  memory (no more multi-GB tensors at 1e5 points), and gemm-free by construction —
  the Blackwell cuBLAS workaround branch is gone because nothing calls gemm anymore.
- **The portable GLS direct path evaluates `cos`/`sin` once instead of six times** —
  the base-grid sums for the weights and the weighted data share one angle matrix and
  the doubled-frequency sums follow from the double-angle identities: ~1.8x on
  torch:cpu and torch:cuda; the frequency chunk is auto-capped by the light-curve
  length so device memory stays bounded regardless of `N`.
- **PDM bins each point once** into `n_bins*n_covers` fine bins and regroups every
  cover exactly from that histogram (vectorized, CUDA, and numba paths): ~2.8x on the
  numpy path, one third of the shared-memory atomics in the CUDA kernel.
- **Opt-in `precision="float32"` now reaches the CUDA `RawKernel`s** (BLS, PDM, CE,
  TLS): on consumer GPUs, whose float64 throughput is 1/64 of float32, the
  FLOP-bound searches speed up ~8.6x (BLS full segmented run 610 -> 71 ms; TLS
  34 -> 4 ms on an RTX Blackwell card). float64 stays the default; PDM keeps float64
  accumulators and CE counts are exact integers at any precision. The float32 BLS
  guards empty box windows with an `ivar` floor scaled to the total inverse variance
  (float32 cumsum noise would otherwise fabricate boxes).
- **String-Length gains a real CUDA kernel**: one block per period bitonic-sorts the
  (phase, index) pairs in shared memory — stable, matching the CPU tie order — with
  no `(P, N)` intermediates (~75x over numpy; curves longer than the shared-memory
  capacity fall back to the vectorized path). The torch path fuses its sort+gather
  via `torch.sort(stable=True)`.
- **Faster scatter shims everywhere**: numpy binning goes through buffered
  `bincount` instead of unbuffered `np.add.at`, cupy through `cupyx.scatter_add`,
  and the batch kernels scatter row-wise so torch reads a stride-0 view instead of
  materializing `(P, N)` broadcast copies and flat indices.
- **BLS uploads each light curve to the GPU once per run** (cupy and torch) via a
  per-curve device cache shared by the period segments — previously every segment
  re-uploaded the arrays and synced on a device-side `t.min()` — and returns all
  seven per-period outputs in one stacked device-to-host copy. The GLS NUFFT paths
  batch the base-grid pair (`w`, `w*y`) into a single `n_trans=2` transform, and the
  one-shot cufinufft path assembles the power on the GPU (194 -> 290 curves/s).

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
- **`cuperiod doctor` CLI command.** A one-stop environment diagnosis: which backends are
  installed, the NVIDIA CUDA fast paths, the portable `torch` backend and every device it
  sees (CUDA/ROCm/MPS/XPU/CPU) with the precision each would use, and what `backend="auto"`
  resolves to for every method. Reach for it first when a GPU isn't being picked up or
  you're unsure which PyTorch build you have.
- **Interactive desktop GUI (`cuperiod-gui`).** A PySide6 + pyqtgraph periodogram explorer
  over the existing API: load a light curve (or a whole folder in batch mode), run any
  method with all of its options, and explore the full-resolution spectrum with a live
  phased light curve — pick peaks and watch the fold update, overlay multi-band curves,
  and toggle a dark/light theme (remembered across launches). Install with the `[gui]`
  extra (`pip install 'cuperiod[gui]'`) and launch with `cuperiod-gui` or
  `python -m cuperiod.gui`. It is a pure presentation layer — the compute core is unchanged.
- **Torch-backend numerical validation in the benchmark suite.** The validation report
  (`benchmarks/`) now checks the portable `backend="torch"` path against cuPeriod's CPU
  backend on every method and validation star, in addition to the existing CPU↔GPU parity
  and reference-implementation checks.
- **Synthetic injection–recovery sensitivity suite** (`benchmarks/injection_recovery.py`).
  Sinusoid, eclipse, and box-transit signals of known period and amplitude are injected onto
  real ASAS-SN observation cadences and scored across a range of signal-to-noise ratios, per
  method, with the same harmonic-aware 2% tolerance used elsewhere in the report.
- **Validation dataset expanded from 72 to 126 ASAS-SN light curves**, via a reproducible
  download script (`benchmarks/dataset/download_extension.py`) that adds 54 less-curated
  stars — spot-evolving rotators and long-period semiregular/Mira variables — to give a more
  realistic, heterogeneous field sample alongside the original curated core.
- **Wilson 95% confidence intervals and a data-driven "notable failures" breakdown** in the
  benchmark report, replacing point-estimate recovery rates and a single blanket explanation
  with per-star-group failure-mode categorization (near-miss, period-wandering, alias/
  harmonic).

### Fixed

- **torch GPU device placement (all seven methods).** The portable kernels built their
  trial grids and accumulators (`xp.zeros`/`arange`/…) on the host, so a run on a torch
  *GPU* device raised "Expected all tensors to be on the same device". The array-API bodies
  had only ever been exercised on the torch **CPU** device (where the bug is invisible);
  they now create arrays on the input's device. Surfaced by first running the torch CUDA
  path on real GPU hardware.
- **MHAOV GPU crash on some GPUs.** MHAOV's batched normal-equations solve went through
  cuBLAS gemm, which intermittently raised `CUBLAS_STATUS_INVALID_VALUE` on a cold handle
  (observed on NVIDIA Blackwell / sm_120, in **both** cupy and torch). Its GPU paths now use
  gemm-free reductions — numerically identical to the einsum, and here no slower — while the
  numpy CPU path keeps the einsum. (MHAOV's cupy path is now routed through the array-API
  namespace, like the other vectorized methods, so it can build device-matched arrays.)
- **`cuperiod.__version__`** returned a stale hard-coded `"1.0.0"`; it now derives from the
  installed package metadata so it always matches the real version.

### Known limitations

- **AMD/Intel/Apple torch GPUs are written-to-spec and CPU-validated, not yet on-device-
  verified.** The torch **CUDA** path is now validated on NVIDIA hardware (RTX 5070 Ti,
  Blackwell / sm_120): all seven methods match the CPU reference to round-off. The ROCm,
  MPS, and XPU paths share that same array-API body, but their on-device parity self-skips
  (`requires_torch_gpu`) until such hardware is available.
- **No fp64 capability probe on Intel XPU.** `precision="auto"` resolves to float64 on an
  XPU; a device without native float64 will error at compute time rather than falling back
  to float32. Pass `precision="float32"` explicitly on such a device.
- **The torch GPU path does not auto-shrink to small VRAM.** A large period×bin grid on a
  small consumer GPU can raise an out-of-memory error; reduce `batch_periods`.
- **Tie-broken best-fit *extras* may differ across devices.** Where an `argmax` lands on an
  exact tie (e.g. BLS `transit_time`, TLS `t0`/`duration` at non-transit periods), the
  chosen index is device-dependent; the periodogram power and best period are unaffected.

## [1.0.0] — 2026-06-30

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

[1.1.0]: https://github.com/tjayasinghe/cuPeriod/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/tjayasinghe/cuPeriod/releases/tag/v1.0.0
