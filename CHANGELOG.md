# Changelog

All notable changes to cuPeriod are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-08-14

### Added

- **SuperSmoother — a fully non-parametric period search** (`"SuperSmoother"`,
  `cuperiod.SuperSmootherSettings`): Friedman's (1984) variable-span smoother
  applied to every phase-fold — three local-linear smooths over span fractions
  `(0.05, 0.2, 0.5)`, leave-one-out cross-validation picking the best span at each phase
  point, and a final pass over the blended curve. The statistic follows gatspy,
  `1 - mean|y - model|/dy / mean|y - mu|/dy`, the fractional reduction in mean absolute
  (error-standardized) deviation about the inverse-variance weighted mean, *maximized* at
  the true period (`1` a perfect fit, `0` no better than a constant, slightly negative
  when the fold fits worse than the mean). Fitting the fold itself instead of a harmonic
  model is the point: no harmonic budget to choose, and an RR Lyrae sawtooth, an eclipse,
  or a shape nothing analytic describes are all captured — this was the period finder
  behind the Stripe 82 RR Lyrae work (Sesar et al.) and one of the methods compared by
  VanderPlas & Ivezić (2015). The price is cost and a soft spectrum, plus one caveat worth
  stating plainly: a fold at an integer *multiple* of the true period is still a coherent
  repeating curve, so `2P`, `3P`, … score nearly as high as `P` — read the shortest period
  of a high-scoring family as the candidate, bound the trial periods from above by raising
  `minimum_frequency` (the longest period searched is `1/minimum_frequency`), or let
  `cuperiod.alias_diagnostics` arbitrate the family.
- **SuperSmoother on every backend.** One vectorized array-API kernel serves `numpy` on the
  CPU, `cupy` on NVIDIA and `torch` on any device, alongside a numba-parallel CPU tier that
  becomes the default with the `[fast]` extra: 20 000 trial frequencies on a 600-point
  curve take **0.05 s** there against 6.6 s for plain numpy on the same 32-thread machine.
  Every window sum comes from prefix sums over circularly padded folds, so the periodic
  path is exact and the plain smoother's edge pathologies cannot arise, and the
  leave-one-out span selection subtracts each point's own contribution rather than
  refitting. float32 is available on cupy/torch, automatic only where the device forces it
  (MPS).
- **Multi-band SuperSmoother** — gatspy's `SuperSmootherMultiband` exactly. Being
  non-parametric there is no shared-phase model to pool into, so each band is smoothed
  independently on the shared trial grid and the per-band scores are combined with
  baseline-error weights `B_k = mean|y - mu_k|/dy`. `B_k` is the denominator of that band's
  own score, which makes the combination the *total* fractional reduction in mean absolute
  deviation across all bands: a flat or noisy band contributes little weight, and with one
  band it collapses exactly to the single-band score. A band participates with at least 3
  finite points. Since nothing ties the bands' phases together, the GLS `"offsets"` model
  remains the right tool for a sparse Rubin-cadence *search*; this one is for
  characterizing an arbitrary fold shape when the bands are individually decent.
- **SuperSmoother is pinned against the reference implementations** — the `supersmoother`
  package (VanderPlas) and `gatspy.periodic.SuperSmoother` / `SuperSmootherMultiband`, to
  ~1e-9 in `tests/test_supersmoother.py` on the point counts where the window conventions
  coincide; both join the `dev` extra as test-only pins (BSD-2-Clause). Four deviations
  from the reference are deliberate and documented in the module: span windows are forced
  to odd point counts (upstream master's fix of the released 0.4 truncation), folding is
  always periodic, degenerate duplicate-phase windows fall back to the weighted mean
  instead of raising, and the bass-enhancement factor is clamped to close the reference's
  `alpha` ∈ (9, 10) NaN bug. Cross-backend parity (numba, torch, cupy against numpy) is
  asserted at ~1e-11.
- **A native multi-band GLS, replacing the astropy delegation** on every backend
  (finufft on the CPU, cufinufft on CUDA, torch on any device). Three joint models are
  selected with `GLSSettings.mb_model`. The default `"offsets"` is the shared-phase
  `(1, 0)` model of VanderPlas & Ivezić (2015) — one sinusoid on a phase shared by every
  band plus an independent constant offset per band, their recommended search model for
  sparse multi-band data. The offsets are profiled out in closed form, which leaves the
  Zechmeister-Kürster assembly with every trigonometric sum replaced by its band-centered
  counterpart and costs `K + 2` NUFFTs for `K` bands: on a six-band, 100-point star over
  200 000 frequencies, **55 s through astropy becomes 0.13 s** (~400×). With one band it
  reduces exactly to single-band GLS.
- **`mb_model="perband"`** — the multi-phase `(0, 1)` model: independent per-band
  floating-mean sinusoids combined with the paper's reference-chi-squared weights
  (eq. 23), `P = sum_k chi2_0k P_k / sum_k chi2_0k`. astropy's `method="fast"` intends
  this model but weighs the bands by the summed *squared periodogram* instead of
  `chi2_0k`, which makes its output depend on the frequency grid it was evaluated on;
  that differs from the published weighting. gatspy uses `chi2_0k`, and so does cuPeriod.
- **`mb_model="flex"`** — astropy's flexible regularized model (`mb_nterms_base` shared
  harmonics plus `mb_nterms_band` harmonics-with-offset per band, trace-scaled ridge
  `mb_reg_band=1e-6` on the band columns), reproduced from per-band harmonic trig sums
  and batched normal-equation solves. Parity with astropy is ~2e-10 across term counts
  and under both ridge conventions (`tests/test_multiband_gls.py`).
- **Multi-band false-alarm probabilities** (`cuperiod.multiband_fap`), which
  astropy's `LombScargleMultiband` does not offer at all — its FAP methods raise
  `NotImplementedError`, because the single-band analytic formulas assume one sinusoid fit
  to one band. cuPeriod calibrates the joint periodogram by within-band bootstrap: each
  band's `(value, error)` pairs are resampled with replacement while every observation
  *time* stays fixed, which preserves the window function, the per-band sample sizes and
  the heteroskedastic errors while destroying phase coherence. `MultibandFAP` carries the
  null sample with `.fap(power)` and `.level(fap)`; `GLSSettings(mb_fap_bootstrap=N)`
  attaches `extras["fap"]` at the spectrum's peaks plus `meta["fap_level_10pct"]` /
  `fap_level_1pct`. On the NUFFT backends the whole bootstrap runs as the *same* `K + 2`
  transforms as one power evaluation, with the resamples stacked along `n_trans`; the
  perband/flex models and torch fall back to a per-resample loop. The smallest resolvable
  false-alarm probability is `1/(n_bootstrap + 1)`, and asking for less raises.
- **Every method is now multi-band** — pooled PDM, conditional entropy, and string length
  join the existing MHAOV (pooled `F`) and BLS (shared ephemeris, stacked depth-SNR); only
  TLS remains single-band. Each band keeps its own mean curve, histogram, and
  normalization, and only the resulting statistics are pooled: PDM by within-bin degrees
  of freedom `max(n_k - n_bins, 1)`, CE and string length by point count. Forcing the
  filters onto one common fold would smear it by the band offsets alone and look
  disordered at *every* trial period.
- **Multi-band ingestion everywhere a light curve loads.**
  `MultiBandLightCurve.from_file` reads a long-format CSV/ECSV/FITS/Parquet table and
  splits it on an auto-detected or named band column, with no pandas dependency. The CLI's
  `cuperiod run FILE --band COL` now performs a true joint fit — it silently dropped to
  single-band before — and `batch_periodograms` honors `band_column` for file, glob, and
  directory inputs, so a directory of survey tables runs multi-band end to end.
- **Alias diagnostics for any periodogram** (`cuperiod.alias_diagnostics`). A peak
  quoted without an alias check is a period a referee will ask about, so this measures the
  spectral window of *this* light curve's sampling, predicts the alias family it implies
  (`f0 ± m·f_w` off the window's own peaks, plus harmonics and subharmonics; the classic
  sidereal-day/solar-day/synodic-month/year suspects when no light curve is supplied),
  matches each prediction to a local optimum of the periodogram within a Rayleigh
  tolerance, and scores the competitors on one scale where `1.0` means "as good as the
  peak being diagnosed". Harmonics are reported but never make a result `ambiguous`, since
  `2f` is expected structure. Works for maximized and minimized statistics alike and for
  multi-band input; `AliasReport.summary()` prints the competitor table a period-search
  paper is expected to show.
- **LINCC Frameworks interoperability** (`cuperiod.interop`, new `[nested]` and `[lsdb]`
  extras): run a period search directly on nested-pandas / lsdb light curves — one row per
  object, the epochs in a nested column — with no flattening, no `groupby`, and no
  per-object DataFrames. `cuperiod.interop.nested_periodogram` is the row-wise tier
  (`map_rows`, composable, right for a CPU backend or a quick look);
  `cuperiod.interop.partition_periodogram` is the throughput tier, reading a
  partition's flat Arrow buffers and list offsets once and evaluating every object in it
  against **one** GPU engine, so plan/kernel setup is amortized over thousands of stars
  instead of paid per star. Both accept an in-memory `NestedFrame` or a lazy lsdb
  `Catalog`, resolve their columns from the nest's schema without computing, and turn a
  failed object into NaN result columns rather than an aborted run. `COLUMN_PRESETS`
  carries verified layouts for `"ztf_dr22"`, `"ztf_alerts"`, `"rubin_dp1_object"` and
  `"rubin_dp1_dia"`; the Rubin presets search in the **flux** domain because DP1 fluxes
  are nJy and can legitimately be negative. One code path supports both ecosystem worlds —
  nested-pandas 0.6.10 (the lsdb / pandas-2 pin) and 0.7.x (pandas 3) — using only
  `map_rows` / `map_partitions` / `join_nested`, never the `reduce` that 0.7.0 removed.
- **A multi-band recovery benchmark** (`benchmarks/multiband_recovery.py`): faint
  RRab-like stars on a simulated Rubin-like six-band cadence (3-year span, WFD epoch
  shares, 0.20 mag per-point noise ≈ an *r* ≈ 23 halo RR Lyrae), 300 stars per cell,
  recovery = top period within 1% with no harmonic credit. At 30 total epochs across all
  six bands the best single band recovers 0.0% and any-single-band 0.0%, while multi-band
  `perband` reaches 17.7%, `flex` 20.0%, and the shared-phase `offsets` model **81.7%**;
  at 60 epochs, 37.3% / 49.3% versus 97.0% / 97.0% / **99.7%**; by 120 epochs everything
  converges. The cadence is deliberately simplified (random nights, no rolling cadence),
  so the result to read is the *ordering*: sharing the phase is what buys sparse-cadence
  recovery, consistent with Rubin's own alert-production study and with VanderPlas &
  Ivezić (2015) — and it is why `"offsets"` is the default.
- **A bundled real multi-band validation set** (`benchmarks/dataset/s82_rrlyrae.parquet`,
  built by the re-runnable `benchmarks/dataset/download_s82_rrlyrae.py`): 100 SDSS Stripe 82
  RR Lyrae from Sesar et al. 2010 (ApJ 708, 717) — real *ugriz* photometry on the real
  ground-based cadence, ~55 epochs per band over a ~3200 d baseline, 80 RRab and 20 RRc with
  periods 0.26–0.91 d, each carrying the discovery paper's literature period. It is the
  canonical real multi-band test set: VanderPlas & Ivezić (2015) developed the multiband
  periodogram on these very stars. The script fetches the paper's tables from the astroML-data
  mirror (the original MPIA host is dead) and keeps the first 100 stars by ascending Sesar ID,
  a deterministic cut with no quality selection and no filtering on how any method performs.
  At ~390 KB the bundle is committed, so the validation reproduces offline.
- **Every multi-band method validated on that real data** (`benchmarks/multiband_real.py`,
  written up as a new REPORT.md §4): one identical blind search per star — periods 0.15–1.2 d
  at 5 samples per Rayleigh width (~97 000 trial frequencies), every method at its default
  settings — scored **strict** (top period within 1% of the literature value, no harmonic
  credit) and **harmonic-aware** (a small-integer harmonic within 2%). The pooled fold
  statistics lead on curves this well sampled: PDM and string length reach **93%** strict,
  with string length at **97%** harmonic-aware and SuperSmoother at **96%**, against 76–78%
  for the three GLS models — about what the best single band manages (78%, or 92% if any of
  the five bands is allowed to be the right one). That is the mirror image of the simulated
  sparse-cadence benchmark above, where `"offsets"` leads: dense per-band data reward fold
  shape, sparse data reward parsimony. SuperSmoother's strict-vs-harmonic gap is exactly the
  documented integer-multiples family (11 of 21 fold-family harmonic picks land on precisely
  2P; 55% strict on the near-sinusoidal RRc against 92.5% on RRab), and 54% of the 163
  non-harmonic misses across all models sit on the ±1 or ±2 cycle/day window-alias loci — the
  ground-based window function, not noise. BLS is reported for completeness (22%) and stays
  the wrong tool for a pulsator.
- **Automated, uncertainty-aware pre-whitening for classical pulsators**
  (`cuperiod.prewhiten`). Frequency analysis of δ Scuti, γ Doradus and SPB stars
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
  pruned — the re-check is the S/N test, so it runs when `"snr"` is among the criteria,
  and every reported frequency then satisfies it.
- **Error propagation** with three estimators (`uncertainty=`): the linearised
  least-squares covariance of the joint fit (the default, and the only one that accounts
  for correlations between close frequencies), the classical Montgomery & O'Donoghue
  (1999) formulae, and a residual bootstrap. All optionally inflated by the
  Schwarzenberg-Czerny (1991) correlation factor. Phases are referenced to the weighted
  mean epoch, at which a phase is uncorrelated with its own frequency — validated
  against Monte Carlo: reported 1-sigma errors match the realised scatter to within
  ~5% in frequency, amplitude, and phase.
- **Combination-frequency identification**
  (`cuperiod.identify_combinations`) with uncertainty-aware tolerances — a match
  must fall within `max(3σ, 0.25/T)` of the *propagated* prediction — and a
  chance-coincidence rate reported per identification, so a spurious match is visible as
  such. `PreWhitenResult.independent()` returns the candidate independent-mode list.
- **g-mode period-spacing tools**: a comb scan over trial spacings
  (`cuperiod.spacing_spectrum`, unaffected by missing radial orders, and immune to
  the sub-multiple ambiguity that makes a naive scan report ΔΠ/2), extraction of the
  longest *tilted* series `ΔP(P) = a + bP` bridging missing orders
  (`cuperiod.find_period_spacing`), échelle coordinates
  (`cuperiod.echelle`), and the buoyancy radius Π₀
  (`cuperiod.buoyancy_radius`).
- **Batch pre-whitening** (`cuperiod.batch_prewhiten`) over the existing CPU/GPU
  worker pools, writing one row per extracted component to Parquet/CSV with resumable
  directory sinks.
- **CLI**: `cuperiod prewhiten` (with `--spacing`, JSON/CSV/npz output) and
  `cuperiod batch-prewhiten`.
- **The spectral window as a first-class diagnostic**: `SpectrumEngine.window()`,
  `cuperiod.spectral_window`, and `PreWhitenResult.window` expose `|W(f)|` of the
  sampling — the alias-lobe pattern every real peak is convolved with — at zero extra
  transforms (its sums are already part of the cached normal equations). The GUI
  spectrum view gains a *window* overlay toggle (scaled to the tallest peak,
  Period04-style) and `cuperiod prewhiten --save-spectrum` writes it into the `.npz`.
- **GUI: the data curve has its own toggle**, so the residual and window traces it
  draws over can be read on their own, and **double-clicking the spectrum restores the
  default view** (what the *Reset* button does). The *peaks* toggle now also hides the
  shaded selection band — it marks a peak, so it belongs to the same layer — while
  keeping the selection itself, so the folded period does not change underneath you.
- **An amplitude-reliability flag.** Every component now records
  `spectrum_amplitude` — the amplitude spectrum read directly at its frequency, a
  single-frequency measurement independent of the joint fit — alongside the derived
  `amplitude_ratio` and a `blended` flag (`blend_tolerance`, default 2×). A ratio far
  from 1 says the amplitude is entangled with a component it is correlated with, which
  in ground-based data usually means the mode's own alias sidelobe: on the bundled
  ASAS-SN HADS demo the 3f harmonic is fitted at nearly twice what the data holds
  there. It is explicitly *not* a significance test — a blended component can be real —
  and nothing is dropped because of it. `PreWhitenResult.n_blended` counts them,
  `summary()` gains an `A/Asp` column, the batch catalogue gains both floats, and the
  GUI's Frequencies dock marks the affected rows.
- **A native Baluev (2008) false-alarm probability** (`cuperiod.baluev_fap`)
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

### Performance

- **The GPU plan cache is now reused for multi-band periodograms too.** The batch runner
  and the interop partition kernel built a `CufinufftGLS` engine per worker/partition but
  dropped it on the multi-band branch, so every star paid full plan setup again.
  `multiband_power` accepts the same engine the single-band path uses, and the `"offsets"`
  model's `K + 2` transforms, the flex harmonic sums, and the perband per-band powers all
  route through its bucketed plan cache — a plan is fixed by mode count and `n_trans`
  alone, so one pair serves every band, harmonic, and star in a partition. The
  bootstrap-FAP pass deliberately stays planless: its `n_trans` varies with the grid, and
  caching a plan per value would balloon device memory. The fold methods accept and ignore
  the parameter; their kernels are already module-cached.
- **Single-shot GPU calls of MHAOV and SuperSmoother no longer pay per-chunk dispatch
  overhead.** Both kernels walked the trial grid in small fixed chunks (512/1024), and on
  a ~100k-frequency grid the hundreds of chunk iterations — each a burst of kernel
  launches, for SuperSmoother plus a device→host copy that synchronized the stream every
  chunk — dominated the wall clock: on the Stripe 82 validation stars a single multi-band
  call took ~24 s on GPU against ~0.3–0.5 s on the numba CPU tier. `batch_periods` now
  defaults to `0` = auto-sized from a transient-memory budget (`~512 MiB` of workspaces
  on device backends, the previous chunk sizes on host numpy, always adapted to the
  light-curve length so long curves cannot blow memory), and SuperSmoother accumulates
  scores on the device and crosses to the host once. Chunking never affects the result —
  the same stars now run at CPU-tier speed on the GPU (MHAOV ~0.5 s, SuperSmoother
  ~0.8 s single-shot; identical spectra). An explicit `batch_periods` value is honored
  as before. The single-curve benchmark sweep was re-measured under the new defaults
  (single-band MHAOV GPU 0.135 s → 0.038 s, from ~5× slower than the numba CPU tier to a
  near-wash across the whole grid-size sweep), and SuperSmoother joined the sweep with
  its first recorded single-curve and scaling numbers.

- `cuperiod.gui.models.ResultCache` is now generic over its value type, so the app keeps
  one cache per analysis and switching back and forth is instant.
- `batch_prewhiten` forces `store_spectra=False`: catalogue rows never carry spectra,
  so keeping them only made each worker hold megabytes of grid arrays per star.

### Fixed

- **The automatic pre-whitening band could sit entirely below a δ Scuti star.** The
  default topped out at the median-gap pseudo-Nyquist (with `nyquist_factor=1`), which
  for nightly ground-based sampling is ~0.5–2.5 cycles/day — so on the bundled ASAS-SN
  HADS demo (P = 0.0898 d, f = 11.14 c/d) the extraction fitted a spurious low-frequency
  solution instead (P = 9.00 d, residual rms 0.245), while the GUI — whose own ceiling
  was 10 c/d — fitted the signal's daily alias (P = 0.123 d). The auto band is now
  `max(pseudo-Nyquist × nyquist_factor, 50 c/d)` with `nyquist_factor=5` (matching the
  periodogram methods), exposed as
  `cuperiod.prewhiten.default_maximum_frequency`, and the GUI's auto value uses
  the same helper. The demo star now yields P = 0.089757 d — the VSX period to the
  last digit — with its 2f, 3f, 4f harmonics extracted and combination-labelled
  (residual rms 0.073). The GUI applies the same floor to the **GLS and MHAOV**
  periodograms — a trial frequency costs them a trig sum, so the wider band is free —
  and GLS now also recovers the demo star's period; the fold-based methods (PDM, CE,
  string-length), which pay a full fold per trial frequency, keep their 10 c/d auto
  ceiling.
- **`MultiBandLightCurve.from_dataframe` ignored `band_column` when `columns=` was also
  given.** An explicit `ColumnMap` replaced the map built from `band_column` outright, so
  `from_dataframe(df, band_column="filter", columns=ColumnMap(time=..., value=...))`
  raised `ColumnResolutionError` telling the caller to pass `band_column` — which they
  had. The two are now merged, matching `from_file`.
- **GUI: hiding the peaks left their hover label behind.** The label is anchored to a
  marker, but nothing dismissed it when the markers went away — so unchecking *peaks*
  right after hovering one to read it (the natural order) stranded the numbers over an
  empty plot, and the same label survived a new result and axis/log switches that moved
  its anchor. It is now dismissed whenever the markers are redrawn, and reappears on
  the next hover.
- **GUI: the Frequencies table ignored its own number formats.** The sort key was
  written to `EditRole`, which `QTableWidgetItem` stores in the same slot as
  `DisplayRole`, so every numeric column silently rendered Qt's six-significant-digit
  default — too few digits for a frequency (`11.1412` where the solution knows
  `11.1412066`) and too many for an uncertainty. The sort key now lives on the item.
- **GUI: component markers floated above the amplitude spectrum.** They were drawn at
  each component's *fitted* amplitude while the curve shows the single-frequency
  amplitude spectrum. Those agree for a well-separated mode but diverge as soon as
  components are correlated — on the bundled HADS demo, whose harmonics each carry
  yearly alias sidelobes (Δf = 1/365.25 d), the two differ by up to a factor of 4.6 —
  so markers hung in empty space claiming peaks the spectrum does not have. Markers
  now sit at the height of the curve they annotate; the fitted amplitude and S/N moved
  to the hover readout, alongside the Frequencies dock that already reported them.
- **Bootstrap frequency errors were exactly zero for every component but the newest.**
  The engine handed its per-iteration refinement policy (default `"last"`, which pins
  all established frequencies) to the bootstrap's replicate fits, so their scatter
  collapsed. Replicates now always sweep every frequency, boxed by the same
  per-frequency bounds as the fit they characterise.

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

[1.2.0]: https://github.com/tjayasinghe/cuPeriod/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/tjayasinghe/cuPeriod/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/tjayasinghe/cuPeriod/releases/tag/v1.0.0
