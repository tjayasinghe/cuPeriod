# Changelog

All notable changes to cuPeriod are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Interactive, Period04-style pre-whitening** (`cuperiod.PreWhitenSession`): the
  extraction loop one action at a time. `step()` is exactly one iteration of the
  automatic loop; `advance()`, `run()`, `polish()`, `prune()`, `renumber()` and
  `finalize()` complete the automatic path. Manual edits: `add` (snapped to the residual
  peak), `add_peak` (tallest peak, no acceptance test), `remove` (`cascade=True`
  dissolves the ties of a parent's children), `set_active`, `fix` / `set_values`, `tie`,
  `set_variation`, `calculate`, `improve` (all, or selected rows with the rest held) and
  `set_note`. Data edits: `delete_points` / `restore_points`, `exclude_subsets`,
  `define_partition` / `drop_partition`, `use_offsets` (a fitted zero point per night,
  sector or band), `set_subset_weights` / `use_weights` (with the 1/rms² helper
  `subset_rms_weights`), `use_variation` and `configure`. Every action is logged with
  the arguments it really used and undoable in groups (`undo`, `redo`, `goto`,
  `group()`, `history`, `log_entries`, `history_cursor`); an edit no fit could honour is
  refused before it commits. One entry point, `do(op, **args)`, never raises for bad
  input: it returns a `refused` `StepReport` and changes nothing.
- **Errors from Calculate and Improve.** Calculate (F8, `PreWhitenSession.calculate()`)
  and Improve (all or selected) solve with the covariance, so the frequency table shows
  frequency, amplitude and phase errors straight away (0 for a fixed quantity); the
  automatic re-calculation after each edit stays fast and shows "–" until Calculate or
  Improve. Each action's history line says it once ("Restored 22 points", not "Restore
  22 point(s): restored 22 points").
- **`cup.prewhiten` is now a session run to its stopping rule and finalized.** Its
  results are unchanged, held bitwise by a 47-case golden corpus and frozen copies of
  the v1.3.0 numerics.
- **Bitwise replay on any backend.** Steps log their peak seed and grid index, so
  `run(seeds=...)`, `PreWhitenSession.replay()` and `load(verify=True)` repeat a session
  exactly on any spectrum backend (a finufft session verifies on numpy).
- **Pre-whitening projects** (`*.cupw.json`, schema version 1): `save()` / `load()`
  write and read the creation arguments, the current solution and the seeded action
  log, with the light curve embedded or referenced next to the project and pinned by a
  SHA-256. Projects are safe to open from anyone (see Security); `snapshot()` /
  `restore()` keep and re-attach a session in memory, undo states included, without a
  re-run.
- **Reproducing scripts**: `to_script()` / `export_script()` write a Python script that
  uses only the public API, reloads and hash-checks the data, repeats every action
  (consecutive steps as one `run(seeds=[...])`) and reports whether it reproduced the
  session. `PreWhitenSession.from_csv` reads the data file it writes when the source
  cannot be re-read exactly.
- **Amplitude/phase variation tests**: `PreWhitenSession.variations()` fits per-subset
  amplitudes and/or phases of selected components and returns Wald constancy tests (χ²
  with S−1 degrees of freedom), p-values and the BIC change (`VariationResult`), with
  the frequencies held; `set_variation` puts the variation into the model.
- **Constrained multi-sine fits** (`fit_multisine(constraints=...)`, `FitConstraints`,
  `ComponentConstraint`, `FrequencyTie`, `FitStart`, `ConstraintError`): fixed
  frequencies, amplitudes and phases; integer frequency ties (`"F3 = F1 + F2"`, `"2F1"`,
  parsed, never evaluated); one fitted zero point per subset; per-subset amplitudes
  and/or phases on an independent partition; and relative point weights, solved by a
  general variable-projection kernel. Unconstrained calls never reach it.
  `MultiSineFit.detail` (`FitDetail`) carries per-subset values with errors, flags
  (`fixed_f`, `fixed_a`, `fixed_phi`, `tied`, `negative_amplitude`, `degenerate`,
  `at_bound`, …), the covariance, rank and condition; the uncertainty estimators honour
  the constraints and `bootstrap_uncertainties` / `component_uncertainties` accept
  `should_stop=`.
- **Partitions** (`cuperiod.Partition`, `cuperiod.prewhiten.suggest_partition`): nights,
  TESS sectors, chunks, seasons, label columns, time ranges and crossings of the points,
  with the recipe recorded and hand edits kept run by run.
- **Result additions**: `Sinusoid` gains `fixed`, `tied`, `variation`, `amplitude_rms`
  and `flags`; `PreWhitenResult` gains `subsets` (`SubsetSolution`: zero points and
  per-subset amplitudes and phases with errors) and `n_deleted`; `to_table(constraints=)`
  adds the constraint columns only when a component carries a constraint, so automatic
  results keep the v1.3 layout. `AmplitudeSpectrum.noise_curve()` gives the local noise
  at every grid frequency; `SpectrumEngine.release()` drops GPU plans.
- `cup.SessionSettings` (`CUPERIOD_SESSION_*`): undo depth, caches, implicit
  recalculation, prune protection and project limits. `PreWhitenSession.from_bands()`
  pre-whitens all bands merged (each centred, as the GUI does) with a root `band`
  partition, via `cuperiod.core.lightcurve.stack_bands()`.
- **CLI**: `cuperiod prewhiten --save-project FILE --export-script FILE`, and a
  `cuperiod session` group — `verify` (replay a project's log against its stored
  solution), `script` (export a reproducing script) and `replay` (re-create a project,
  optionally saving it with its full history).
- **GUI: interactive pre-whitening.** Compute in pre-whitening runs a session (the
  numbers of `cup.prewhiten`) that stays open for Step (F7), Add peak (F6), edits and
  undo; a cached solution comes back with its exact session, without a re-run. The data
  spectrum appears shortly after a curve is loaded, computed on the CPU; when it cannot
  be computed the spectrum pane says why and the session commands stay off (Run and
  Step still try, and report the error).
- **GUI: the spectrum as a workbench.** A pick cursor separate from the fold; Shift+click
  (Ctrl+Shift: exactly) or A / Insert adds a peak; N jumps to the next candidate, drawn
  as a hollow "next" triangle; component markers show their state by shape (circle free,
  square fixed, diamond tied, hollow not used); the S/N acceptance limit is drawn; edits
  keep the zoom.
- **GUI: editable Frequencies table and session bar.** Use / fix f / fix A / fix ph check
  boxes, typed frequencies, amplitudes, phases and periods, ties typed as `=F1+F2` or
  `2f1` (or by double-clicking an identified combination), per-subset variation and
  notes; Space, L, Shift+L, T, Del and Shift+I act on every selected row. The session bar
  holds Auto, Step, Add peak, Calculate, Improve all / selected, Delete, Undo, Redo and
  Stop, a chip with the candidate Step would try next, and an Add-frequency field that
  takes c/d, µHz, periods and combinations. A session's CSV export lists every component
  with its constraints.
- **GUI: point editing in the raw light curve.** Data / Residuals / Data + model; points
  coloured and shaped by subset; Pan / Rectangle / Lasso selection (Shift-drag in Pan,
  Ctrl adds); Delete / Restore / Clear; *Select beyond N σ* per zero-point subset;
  deleted points stay visible as grey crosses and can be restored. A click (or Left /
  Right on the focused plot) puts a time cursor where Subsets ▸ *Split at* cuts.
- **GUI: Subsets, Variations and History panes.** Split by nights, TESS sectors, Kepler
  quarters, seasons, chunks, bands, a label column or time ranges; zero points, weights
  (by hand, equal or 1/rms²) and per-subset amplitude/phase per partition; leave out,
  rename, merge, split and cross subsets. Variation tests plotted per subset with an
  *Apply to model*. The History pane shows every action (groups collapsible and titled
  in words), returns to any state, and copies an action as Python.
- **GUI: projects.** Open / Save / Save as `*.cupw.json`, Export script, Verify project;
  the title bar reads `cuPeriod — star.csv — star.cupw.json*`; a project whose data file
  moved asks for it. Unsaved sessions survive switching source, band or analysis
  (File ▸ Unsaved sessions), and closing, New session or opening other data asks Save /
  Discard / Cancel; both name a demo or batch source by its label, not its key.
- **GUI: menu bar and command registry.** File, Edit, Pre-whitening, Data, View and Help,
  with Alt mnemonics on every menu and item, built from one registry that also drives
  the buttons, the keyboard shortcuts and the F1 sheet; File gains Open folder
  (Ctrl+Shift+F), Open recent, Load demo, Unsaved sessions, Verify project and Quit
  (Ctrl+Q); View gains Analysis, Panes, Show raw light curve (Ctrl+R), theme (Ctrl+T),
  Zoom and Reset layout; Help has Keyboard shortcuts (F1), What's This? (Shift+F1),
  Documentation and About.
- **GUI: Stop (Esc).** Ends Auto after its current step and keeps what it found; releases
  a running periodogram at once (the run finishes in the background and is cached, not
  shown). The info bar says what runs and for how long.
- **GUI: import dialog** for Open, drops, Open recent and a batch file that failed to
  load (*Choose columns…*): time / brightness / error / band pickers, magnitudes or
  fluxes, the delimiter and header row of text tables, an *As loaded* tab at full
  precision, a note on the time scale the numbers look like, and Load disabled with the
  reason while the choice cannot work. Headerless JD/mag/err tables (OGLE, Period04),
  semicolon-separated files and unrecognised headers such as `HJD, V, e_V` now load.
- **GUI: keyboard-operable spectrum.** A click folds at the nearest peak within 12 px
  (Alt: exactly there), also with the markers hidden; Left / Right move between local
  maxima (Ctrl: marked peaks, Shift: one sample), Home, Enter, + / − / 0 and the Menu
  key work on the focused plot, which screen readers describe. The phased view gains ×2,
  ÷2 and a typed period.
- **GUI: accessibility and zoom.** A visible 2 px focus ring on every control; WCAG 2.1
  AA contrast in both themes; drawn arrows, ticks and grips and larger pointer targets;
  Zoom in / out / reset (Ctrl++ or Ctrl+=, Ctrl+−, Ctrl+0) for text and plot fonts,
  remembered (an application font above 13 px also enlarges the interface); band colours
  that stay distinct under colour-vision deficiencies plus a marker shape per band
  (`ThemePalette.band_symbol()`, `ThemePalette.residual`); accessible names, status tips
  and What's This? help on every option, and a help line under the Options form; 24 px
  pane Float / Close buttons; panes hidden behind a tab leave the Tab order; Ctrl+1 … 5
  focus the panes. New theme API: `apply_theme(..., font_scale=)`, `stylesheet()`,
  `font_scale()`, `step_font_scale()`, `contrast_ratio()`.
- **Grid policy, public** (`cup.GridTooLargeError`, `cup.GridSizeWarning`,
  `cup.default_frequency_grid`, `cup.pseudo_nyquist_frequency`,
  `cup.regular_cadence_nyquist`, `cup.check_grid_size`, `cup.uniform_grid_size`; also in
  `cuperiod.core` with `coerce_domain` and `raise_if_unusable`): every default grid is
  sized before allocation — above 5 million automatic samples it warns (naming outlying
  epochs when they inflate the baseline), above `MAX_GRID_SIZE` (250 million) it raises
  (`max_size=` for a deliberate larger grid). `regular_cadence_nyquist(time)` returns
  `0.5/cadence` for lattice sampling, or `None`.
- **Peak defaults that travel with a result**: `Periodogram.peak_defaults`,
  `with_peak_defaults()` and `cuperiod.core.result.peak_defaults_from_settings`;
  `best_periods(alias_tolerance=)`; `MultiResult` gains `in`, `get()` and `values()`.
- **Calendar times** in the time column: `datetime64` (pandas, tz-aware in UTC, pyarrow
  timestamps) becomes MJD, `NaT` becomes NaN, `timedelta64` / `TimeDelta` become days,
  and astropy `Time` columns keep jd / mjd / btjd / bkjd values and become MJD otherwise.
  Units on `Quantity` columns are not applied: the number is taken as days.
- `LightCurve.unusable_reason()`, `MultiBandLightCurve.unusable_reason()` and
  `cuperiod.core.lightcurve.raise_if_unusable()`, which name a placeholder error column
  as the reason a light curve is empty.
- Backend helpers: `cuperiod.core.backend.refresh_backends()` (re-probe after installing
  an extra mid-session), `cpu_backends()`, `cpu_pool_backend()`, `requested_backend()`,
  `normalize_backend()`; `PeriodogramMethod.resolve_backend_for()`;
  `cuperiod.methods.base.select_settings()`; batch sizing helpers in
  `cuperiod.batch.sizing`; and `cuperiod.methods.gls.FinufftGLS`, a plan-reusing CPU GLS
  engine (`make_engine("finufft")`, one per thread or process).
- `cuperiod.wwz(..., backend="auto" | "cpu" | "numba" | "numpy")` with a multicore numba
  kernel; multi-band BLS reports `extras["band_index"]`; `EphemerisFit.pddot_err_scaled`;
  `PeriodLawFit.bic_scaled` and `.weighted`; `SlidingPeriodogram.n_window_points`;
  `batch_pdot`'s `law_bic_scaled` column; `BatchSummary.n_workers`; a `source` column in
  batch rows; `--chunk-size` on `cuperiod batch` and `batch-prewhiten`.
- `cuperiod.core` re-exports `PdotSettings`, `PreWhitenSettings`, `SpacingSettings`,
  `SuperSmootherSettings` and `SessionSettings`.
- **Joint (period, dP/dt) searching for every method** (`cuperiod.pdot_search`,
  `cuperiod.PdotSettings`): a drifting period defocuses a constant-period search, and a
  constant dP/dt has exactly the phase `f * ln(1 + (Pdot/P) * dt) / (Pdot/P)` (Mikulášek
  et al. 2012 — the same logarithmic phase PHOEBE integrates for its `dpdt`), so
  evaluating an *ordinary* periodogram on warped times scans a dP/dt axis with no new
  kernels: GLS, PDM, CE, String-Length, MHAOV, and SuperSmoother — single- and
  multi-band, every backend — all become period-derivative searches, with the cached
  cufinufft plans serving every row of a GPU scan (cupy and CPU rows agree to 1e-7).
  Both axes are stepped to equal phase smear, the dP/dt step additionally refines by a
  per-method harmonic scale (sharp phase structure smears sooner — the criterion of the
  one prior art, Katz et al. 2021's conditional-entropy GPU search), and the default
  range is set in accumulated drift *cycles*, making the row count baseline-independent
  (201 at the defaults). On a planted 1.1-cycle drift the constant-period row keeps GLS
  power 0.19 where the correct cell shows 0.95; the refined best cell recovers the
  planted (P, dP/dt) to grid precision (`tests/test_pdot_scan.py`, including sign
  conventions, the 160-cycle exact-warp regression, and CPU/GPU/torch parity).
- **Automated O−C analysis** (`cuperiod.phase_track`, `cuperiod.fit_ephemeris`): the
  classical observed-minus-calculated method (Sterken 2005) without hand-collected
  minima — one BIC-order multi-harmonic template at a fixed period, per-segment rigid
  template time shifts with Fisher-information uncertainties (timing precision comes
  from the light curve's *slopes*, so eclipses and sawtooths time better than sines),
  sequential cycle-count unwrapping with a `wrap_risk` flag, and weighted linear-vs-
  quadratic ephemeris fits reporting `Pdot = 2 P c2` with its formal error, the Pringle
  (1975) F-test, and AIC/BIC (convention documented). The template is re-derived on
  drift-corrected times until self-consistent, so an uncorrected drift cannot smear its
  own measurement, and results are always referenced to the constant-period ephemeris.
  A 25-realization Monte Carlo calibrates the reported sigma (pull scatter 0.72–0.93,
  i.e. mildly conservative); `fit_ephemeris` accepts plain event-time lists (nearest-
  integer cycles, `worst_phase_residual` ambiguity diagnostic) and its unit conversions
  are pinned against WASP-12b's published −29 ms/yr — guarding the exact
  multiply-vs-divide mistake that reached a refereed decay claim
  (`tests/test_pdot_track.py`).
- **The Eddington–Plakidis wander test** (`cuperiod.eddington_plakidis`): a smooth O−C
  is not evidence of a smooth period change — random cycle-to-cycle fluctuations
  accumulate a random walk that fits a parabola beautifully (Sterken's dice), and most
  Miras plus a third of Cepheids wander non-secularly. The 1929 test separates the two
  through the growth of accumulated delays with cycle lag, here with two corrections:
  the secular quadratic is removed first, and — because polynomial detrending absorbs
  part of the walk itself — the fit uses *exact* expectations (the detrending projector
  propagated through the walk covariance) rather than the classical line, recovering a
  planted per-cycle fluctuation at 93% ± 20% where the naive recipe returns half
  (`tests/test_pdot_stability.py`).
- **Sliding-window periodograms** (`cuperiod.sliding_periodogram`): any registered
  method on stepped windows over a shared frequency grid — a (time × frequency) map
  with per-window refined peaks, `window_periodogram(i)` slices that are ordinary
  `Periodogram` objects, and a `period_trend()` diagnostic — for the period *wander*
  regime (Miras, spotted binaries) where a single quadratic is the wrong model, and as
  a visual sanity check on any `pdot_search` detection (`tests/test_pdot_sliding.py`).
- **The Weighted Wavelet Z-transform** (`cuperiod.wwz`): Foster's (1996) WWZ, the
  long-period-variable community's standard for period *wander* — the date-compensated
  three-function projection under Gaussian weights whose window is a constant number of
  *periods* wide at every trial frequency (`sigma_t = P` exactly at the default
  `c = 1/(8*pi^2)`; the Templeton et al. 2005 Mira convention `c = 0.001` is
  documented). Returns the Z and semi-amplitude maps with the per-cell effective point
  count, a parabolically refined ridge, and a `period_trend()` diagnostic; cells below
  `min_neff` are masked rather than reported. The docstrings carry Foster's own health
  warning verbatim in spirit: `N_eff` is not an integer, the nominal F-distribution
  does not hold, and no p-value should ever be read off a WWZ map. The implementation
  is pinned cell-by-cell against a transparent loop-and-lstsq reference in
  `tests/test_pdot_wwz.py`.
- **CLI: `cuperiod pdot`** — the search + track chain from the command line: window
  and drift-axis flags mapping onto `PdotSettings`, `--track/--no-track`, JSON output
  (`--out`) and the 2-D map as `.npz` (`--save-map`), with the same column-override
  and domain options as `run` (`tests/test_cli_pdot.py`).
- **Period-evolution models for the search** (`model=` on `pdot_search` and
  `phase_track`; `cuperiod.pdot.models`): the frequency-independence of the time warp
  is a property of a *family* of period laws, not of one law, so the same machinery
  scans either. `"constant-pdot"` is the classical quadratic ephemeris
  (`tau = ln(1 + a*dt)/a`); `"exponential"` holds `d(ln P)/dt` constant
  (`tau = (1 - exp(-a*dt))/a`), the accelerating decay of an inspiralling or
  mass-transferring binary, and has no singular epoch so it can be scanned over a wider
  range. Both are parameterized by the same `a = (dP/dt)/P` at the reference epoch, so
  their `period` and `pdot` are directly comparable and they agree to first order,
  differing in the *cubic* term of the O−C (`-a²/3` vs `-a²/6`) — which is exactly what
  the residual ladder below measures (`tests/test_pdot_models.py`).
- **Model-free P(t) tracks** (`cuperiod.period_profile`, `PeriodProfile`,
  `PeriodPoint`): the period measured independently in consecutive intervals, by
  minimizing a fixed-order harmonic chi-squared over frequency (multi-harmonic AOV —
  Schwarzenberg-Czerny 1996) with the uncertainty read off its curvature. This is the
  data product the period-change literature actually publishes when the rate is not
  constant, and it assumes nothing about the shape of the evolution. The harmonic order
  is chosen once and held fixed so intervals stay comparable, and each interval's search
  is confined to a narrow window around the reference period because an interval holding
  a few tens of cycles has a rich alias comb — periods pushed to that window's edge are
  flagged (`at_window_edge`) rather than reported as measurements. Interval period
  errors are calibrated on synthetic data (pull rms 0.98) and are honest about what they
  are not on real data (`tests/test_pdot_profile.py`).
- **Parametric period laws** (`cuperiod.fit_period_law`, `compare_period_laws`,
  `PeriodLawFit`): `constant`, `linear` (constant dP/dt), `exponential`, `singular`
  — `P = P0 exp(b/(t - t0))`, a finite-time singularity, which is Tylenda et al.
  (2011) Eq. (1) for V1309 Scorpii — and `pejcha`, which lets that singularity's
  exponent float (Pejcha 2014 Eq. 12; `singular` is its `delta = 2` case). Fitted by
  weighted least squares on the periods, with `t0` constrained outside the observed span
  and fitted as a log offset so the optimizer cannot walk into the singularity, and both
  branches (`t0` after the data, and before) tried. BIC ranks them; `law="auto"` skips
  `pejcha`, which needs an exceptionally well-sampled track and can otherwise win with
  its exponent pinned at a bound and wholly unconstrained. `fractional_change` and
  `timescale_yr_at` integrate the law rather than multiplying a rate by a baseline
  (`tests/test_pdot_laws.py`).
- **Higher-order ephemerides and the curvature ladder**: `EphemerisFit` now fits any
  degree and reports `pddot` (the second period derivative, from
  `P*(6*c3 + 8*c2²)`) with its error, plus `implied_model` — whether the fitted
  curvature favours a constant rate (`pddot = 0`) or an exponential one
  (`pddot = pdot²/P`). `pdot` from a cubic fit is still the instantaneous rate at the
  reference epoch but is de-biased against a rate that is itself changing.
  `pdot_err_scaled` exposes the `sqrt(chi2_reduced)` inflation the docs already
  recommended, instead of leaving every caller to apply it by hand.
- **Trend removal** (`cuperiod.detrend`, `cuperiod.running_median`): a running median —
  not a mean, since eclipses are one-sided — with an `O(N log N)` binned variant for
  very long light curves. Survey photometry of a period-changing star routinely carries
  a slow brightness trend that the sampling window scatters onto the 1 c/d alias comb;
  on V1309 Sco an undetrended search returns a one-day alias rather than the orbital
  period. Nothing detrends automatically: the width is a physical judgement about a
  particular star.
- **`PdotBest.ridge_correlation`**: the frequency/rate correlation at the search peak,
  from the local curvature — a number for the tilted ridge the v1.2→v1.3 audit found
  the docs denying. It is the peak's own error ellipse and can differ in sign from the
  visually dominant tilt of the alias comb crossing the whole map, which is documented
  because the figures invite exactly that confusion.
- **Batch period-change measurement** (`cuperiod.batch_pdot`, `pdot_to_rows`): the whole
  chain over many light curves on the same spawned worker pool as
  `batch_periodograms`, writing three tables — one row per light curve (search, O−C,
  curvature ladder, winning law and its parameters), one row per O−C segment, and one
  row per P(t) interval — to Parquet/CSV files or resumable directory sinks. Every stage
  is wrapped so one pathological star cannot kill a survey run: a failure still
  contributes a summary row carrying the reason in `status`, and every law's parameter
  columns are present and NaN-filled in every part so a directory sink reads back with a
  single Arrow schema (`tests/test_pdot_batch.py`).
- **Diagnostic figures** (`cuperiod.pdot.plots`, new `cuperiod[plot]` extra): a
  six-panel per-star figure — light curve, drift-corrected fold coloured by epoch,
  search map, O−C, P(t) track with the fitted law, and the decay timescale — plus the
  panel helpers to compose your own. Built on `Figure` + `FigureCanvasAgg` rather than
  pyplot, so it is safe to call from batch workers, and matplotlib is imported lazily so
  everything else works without it.

### Changed

Results that move when you upgrade — read these first:

- **Default bands on regular cadences.** On strictly regular sampling (constant cadence;
  gaps, dropped cadences and barycentric timing allowed — TESS, Kepler) the GLS default
  grid and the automatic pre-whitening band stop at the true Nyquist frequency
  `0.5/cadence` instead of running past it into mirror images. Set `maximum_frequency`
  (or, for GLS, `nyquist_factor`) for super-Nyquist work. Irregular and nightly sampling
  are unchanged.
- **Multi-exposure ceiling.** When most gaps fall inside visits of at most ten
  back-to-back exposures (dithered triplets, filter sequences, stacked near-simultaneous
  bands), the automatic frequency ceiling of PDM, CE, MHAOV, String-Length,
  SuperSmoother, pre-whitening, WWZ and the GUI is 1e5 Rayleigh widths — about 55 c/d on
  a 5-year baseline and 27 c/d on a 10-year survey (ASAS-SN-like triplets: 20.5M → 500k
  trial frequencies). The capped grid is a prefix of the old one; a `nyquist_factor`
  above 5 widens it in proportion, `maximum_frequency` restores any band.
- **GLS FAP.** A best peak on the grid edge now gets a FAP (a conservative upper bound;
  it was NaN), FAPs are computed on relative times (absolute JDs gave wrong or NaN FAPs
  for runs shorter than a day), the data are centred in float64 before the transform,
  power is clipped to [0, 1], and a constant light curve has zero power.
- **MHAOV centring.** Every backend fits mean-centred values: float64 values move by up
  to ~1e-5 relative on magnitude-like data, towards the exact least-squares value.
  Multi-band MHAOV pools each band in its own variance units, so every power with two or
  more bands changes.
- **BLS likelihood floor.** Boxes holding all or none of the inverse-variance weight are
  rejected by a relative floor instead of astropy's absolute `DBL_EPSILON`, which removes
  fabricated likelihood peaks of ~1e6 on clustered sampling and makes results
  independent of the flux and error scale. `power()` now searches exactly the grid it is
  given (custom grids in log-period segments with period-scaled durations). Multi-band
  BLS shares only the period (statistic Σ max(SNR_k, 0)²), and a band set with any band
  lacking errors uses each band's robust flux scatter as its error instead of raw flux
  weighting (power, `depth_snr` and `sde` change for such input).
- **TLS grid density, depths and epochs.** The default period grid step is
  `min(grid_duration_frac, 2 × duration_min_frac) / (oversample × baseline)`, 2.5×
  denser at the defaults, so the shortest searched transit is resolved; an explicit
  `grid_duration_frac` above `2 × duration_min_frac` is capped. The out-of-transit level
  is fitted jointly with the depth (depths unbiased; SR/SDE move by ~1%, up to ~14% for
  long durations), and `extras["t0"]` (and the batch `t0_N` columns) is an absolute
  mid-transit time like BLS's — code that added `t.min()` must stop.
- **SuperSmoother** scores a constant light curve 0 (it scored 1.0, the maximum), and
  breaks tied phases identically on every backend and row order; numba scores at
  tied-phase frequencies change to match numpy/torch.
- **Pre-whitening uncertainties.** The bootstrap is a wild (sign-flip) residual bootstrap
  with a `sqrt(N/(N−M))` correction and replicates refined like the final polish, so
  seeded bootstrap values differ. `offset_error` goes through the selected estimator and
  the `sqrt(D)` inflation (it grows by `sqrt(D)` at the defaults). The covariance is the
  pseudo-inverse of the Jacobi-scaled normal matrix about the weighted-mean epoch, and a
  parameter of a zero-amplitude component is NaN rather than 0. Two stops are always on:
  the degrees-of-freedom guard (`2(3K+1) > N`) and residuals at numerical round-off.
- **Period-change error calibration.** `fit_ephemeris(errors=None)` estimates the timing
  error from the scatter about the fitted quadratic (`pdot_err` was 10–40× too large and
  depended on the trial period; `chi2_reduced` is now 1 by construction).
  `phase_track` / `fit_template` choose the template order with the scale-free BIC
  (`weighted` no longer affects it). `period_profile` converges each interval on the χ²
  minimum, so interval periods and errors change. `compare_period_laws` and
  `law="auto"` rank on `PeriodLawFit.bic_scaled`, and without `period_err` the laws are
  compared on the residuals' own scale. `eddington_plakidis` errors are the estimator's
  exact covariance (they were 13–26× too small) and `slope_significance` is a calibrated
  z-score. `ltte_search` without `errors=` uses the concentrated likelihood for
  `aic` / `bic` / `delta_bic`. `WWZResult.period_trend()` counts independent windows
  (its error grows up to ~6× on overlapping tau grids). `ladder_pddot_err` is inflated
  by `sqrt(chi2_reduced)` like `ladder_pdot_cubic_err`. An explicit
  `pdot_over_p_range` spanning zero is laid on a zero-anchored lattice (an exact
  dP/dt = 0 row; values differ and may gain a row or two). `running_median` /
  `detrend(method="binned")` honour `min_points`, so binned trends change where a bin
  holds one or two points.
- **Batch resume semantics.** File and directory sinks resume **by key**: a re-run
  computes only the (light curve, method) pairs not yet in the sink and retries failures
  — adding, removing or reordering inputs no longer drops or duplicates light curves, and
  a different `chunk_size` is fine. A directory records its request: adding a method is
  allowed, while changed settings, `n_best`, `store_raw`, columns or grid raise
  `ValueError` once it holds part files. `resume=False` first removes the directory's
  `part-*.parquet` files. `batch_prewhiten` resumes the same way, and `batch_pdot`
  resumes per light curve across its three tables, retrying failed stars and replacing a
  recomputed star's rows in every table (one summary row per key).
- **CPU pools never touch the GPU.** `device="cpu"` maps `"auto"` to CPU backends in
  `batch_periodograms`, `batch_prewhiten` and `batch_pdot` and rejects GPU-only backends
  (`"gpu"`, `"cupy"`, `"cufinufft"`); every CPU-pool worker used to open its own CUDA
  context on CUDA machines. Such batches now run float64 CPU kernels: values can move by
  ~1e-6 and the rows' `backend` column changes. `device="gpu"` keeps a concrete GPU
  backend (torch, cupy, cufinufft) and rejects CPU-only ones.
- **`settings.backend` is the default backend.** `cup.periodogram(..., backend="auto")`
  (or `None`) defers to the method settings' `backend` and `CUPERIOD_<METHOD>_BACKEND`;
  an explicit backend wins. The batch drivers, LINCC adapters, `pdot_search`,
  `sliding_periodogram`, `multiband_fap`, `batch_prewhiten(backend=None)` (the new
  default) and the CLI follow it. A settings pin of `"gpu"` on a machine without one
  raises `BackendUnavailableError` instead of silently running on the CPU.
- **Peak settings are live.** `n_peaks`, `peak_separation_rayleigh`, `harmonic_max`,
  `alias_freq_tolerance` and `downsample_points` now default `best_periods()`,
  `downsample()` and `to_dict()` on results of `cup.periodogram`, the batch drivers, the
  LINCC adapters and the GUI (explicit arguments win). `best_periods(alias_diverse=True)`
  uses an alias/harmonic tolerance separate from the duplicate-peak separation — the
  BLS/TLS `alias_freq_tolerance` (0.0035 c/d), else one Rayleigh width — and treats
  solar-day offsets as aliases: rank 1 is unchanged, lower ranks can differ (a 27-day
  two-planet BLS search now reports the second planet at rank 2).

Other changes:

- **Settings selection is strict.** A settings object of the wrong class raises
  `TypeError` (it was replaced by defaults); mapping keys match like method names
  (`"String-Length"`, `"string_length"`), a key naming no method raises
  `UnknownMethodError`, and two keys for one method raise `ValueError`. Settings validate
  their values: finite frequency bounds with `minimum_frequency >= 0` (pre-whitening
  `> 0`), BLS/TLS duration fractions below 1, TLS `u1 + u2 <= 1`, finite non-negative
  thresholds, positive spacing bounds, `PdotSettings.n_segments >= 4`.
- **Domains and columns.** `domain="flux"` as a string behaves exactly like
  `Domain.FLUX` everywhere (case and spaces ignored; an unknown string raises); an
  explicit `domain=` also chooses the auto-detected column on tables carrying both kinds
  (with a warning when the requested kind is missing); a bare `f` column is flux;
  `domain=` applies to every band of multi-band input, and bands in different domains
  raise unless it is given. BLS/TLS warn when a bare `(t, y, dy)` tuple looks like a
  normalised flux. `ColumnMap(band=...)` on a table given to `cup.periodogram` /
  `to_input` (and on `(DataFrame, group)` batch inputs) gives a multi-band light curve; a
  table with a multi-label band column read as one curve warns that the bands are
  pooled.
- **Band labels** are normalised identically by `MultiBandLightCurve.from_dataframe` and
  `from_file` (`1`, `"1"` and `1.0` are one band `"1"`); rows without a label are dropped
  with a warning instead of forming phantom bands. `from_dataframe` accepts astropy and
  pyarrow tables and `{column: array}` mappings; a `MultiBandLightCurve` refuses
  non-`LightCurve` bands and mixed domains. `meta["bands"]` lists only the bands that
  contributed, and multi-band methods use errors only when every contributing band has
  them.
- **Identity equality.** `Periodogram`, `GridSpec`, `MultiResult`, `PreWhitenResult`,
  `MultiSineFit`, `AmplitudeSpectrum`, `Uncertainties`, `SpacingSpectrum` and
  `PeriodSpacingSeries` compare and hash by identity (`==` used to raise on their
  arrays); `Peak` keeps value equality and is hashable.
- **Validation of grids and spectra.** `GridSpec` rejects unknown kinds, non-finite,
  descending, negative-frequency or non-positive-period values and `uniform=True` on a
  non-arithmetic progression (rounding-level steps back are accepted); the grid builders
  name the bad argument; `Periodogram.from_spectrum` checks shapes and
  `objective_sense`; `alias_diagnostics` refuses a non-finite or out-of-range
  `frequency=`.
- **`WWZResult` API.** `peak_frequency`, `peak_period` and `peak_wwa` are arrays
  (properties) instead of methods, and `to_dict` takes `include_map=` (was
  `include_maps=`), as on `SlidingPeriodogram`. `SlidingPeriodogram.window_periodogram(i)`
  reports the window's own baseline and point count (the run's move to
  `meta["run_baseline"]` / `meta["run_n_samples"]`). `wwz` raises for a non-finite `c` or
  `min_neff` below 3, and `sliding_periodogram` for a non-finite window, step or period
  or a period longer than the window.
- **Pre-whitening details.** `fit_mean=False` fits the weighted-mean-subtracted values
  and reports that mean as `offset`; combination identifications have order ≥ 2 and
  `expected_false` counts only positive in-band predictions (it overstated chance matches
  ~6×); the default period-spacing trial range comes from the bulk of the values; an
  explicit `minimum_frequency` above the automatic maximum names `maximum_frequency` in
  its error.
- **Batch details.** `chunk_size` defaults to automatic (about four tasks per worker, at
  most 256); `grid=` is honoured; rows, errors and file sinks are in input order; every
  row carries `{extra}_{k}` for each per-peak extra (NaN where absent), so directory
  parts share one schema; colliding file stems become path-qualified keys (with a
  warning and a `source` column); a glob matching nothing raises `FileNotFoundError` and
  an existing path is always literal; a batch with no inputs leaves its sink untouched;
  the default worker count follows the CPUs the process may use, and pools are capped at
  61 on Windows. `batch_pdot` validates `method`, `model`, `detrend_method`, `domain` and
  law names up front, matches catalogue keys as strings (a missing, NaN or non-positive
  entry means a blind search; a catalogue matching no key warns), counts a bare string
  for `laws=` / `plot_keys=` as one name, and measures `law_delta_bic` on `bic_scaled`.
- **CLI.** User errors (missing file, unknown method / backend / device, bad sink suffix,
  non-positive `--period`, empty `-m`, `--n-best` < 1, `--segments` < 4, a GPU-only
  backend with `--device cpu`) are usage errors (exit 2) without a traceback; batch
  commands exit 1 when every light curve that ran failed, list every failure in
  `<sink>.errors.csv` (`_errors.csv` in a directory sink) and print the worker count;
  `prewhiten` / `batch-prewhiten` `--backend` default to the settings
  (`CUPERIOD_PREWHITEN_BACKEND`); `pdot --profile` prints `bic_scaled` and its gap to
  the best law and marks fits at a bound; `run --out` saves the same alias-diverse BLS/TLS
  peaks it prints. The `--resume/--no-resume` help of `batch-prewhiten` and `pdot-batch`
  says what resume does now: skip the light curves already in the sink and retry
  failures, matched by key.
- **LINCC interop.** `nested=None` is the new default (the preset's nest, else `"lc"`);
  an explicit `nested=` wins over a preset; a `value=` naming another column than the
  preset's drops the preset's error column and domain; `append_columns=True` onto
  existing result columns raises, pointing at `prefix=`.
- **GUI.** The Options form is grouped into collapsible sections (search range first,
  Advanced collapsed, Multiband only when a joint fit runs) with units, a Reset to
  defaults button, stop-criterion check boxes and editable SuperSmoother spans; it shows
  the values that run (samples per peak 10, explicit values honoured,
  `CUPERIOD_<METHOD>_<FIELD>` overrides applied) and no longer offers `n_peaks`,
  `store_spectra` or `downsample_points`. Results are cached by a fingerprint of the
  data, not the file name, in caches bounded at 512 MiB. A loaded curve is drawn at once;
  errors are explained in plain language with the full text on demand; the Backend and
  Band choices persist; the spectrum remembers its x-axis and log choice per kind of
  result; clicking a marker selects the whole peak (BLS/TLS t0 included); the Peaks
  table shows frequency and names each method's statistic; both tables have Copy /
  Export CSV buttons and a column chooser; the Frequencies CSV has one fixed header and a
  pre-whitening spectrum CSV names its column `amplitude`; plot axes no longer rescale
  by SI prefixes; the window sizes itself to the screen and keeps its layout, analysis
  and options; plot toolbars wrap, a pane wider than its dock scrolls instead of
  squeezing the plots, and the raw view stacks under the fold when there is no room
  beside it.
- **Release process.** The release workflow refuses to publish when the pushed `v*` tag
  differs from the built version, and a manually started run publishes to TestPyPI only.
- **Development.** GPU tests carry the registered `gpu` marker: `pytest -m "not gpu"`
  deselects them and hides the GPUs from the rest of the run (default-backend tests run
  on the CPU); with the GPU stack but no device, GPU tests skip.
- **Docs.** New guide page on interactive pre-whitening, with the GUI's shortcut tables
  generated from its command registry at build time; `BatchSettings` is documented as
  reserved (nothing reads it yet; `CUPERIOD_BATCH_*` has no effect).

### Deprecated

- `PreWhitenSettings.downsample_points` never had an effect; setting it now warns
  (`DeprecationWarning`), and it will be removed.

### Fixed

- **The (period, dP/dt) axes are not orthogonal for skewed sampling**, contrary to what
  the module and guide claimed. Referencing to the weighted mean epoch zeroes the first
  moment of the times — which is what decouples phase from frequency — but the residual
  frequency–drift correlation is set by the sampling's *third* central moment and
  vanishes only for observations symmetric about that epoch. V1309 Sco (skew −1.29, its
  cadence densifying steeply toward the merger) shows the resulting tilted ridge plainly
  in its search map. The claim is now stated correctly: the tilt widens the joint
  uncertainty without biasing the best cell, and `PdotBest.ridge_correlation` now
  measures it.
- **`backend="auto"` raised `ModuleNotFoundError` when the CUDA fast path was
  half-installed** — cupy present but cufinufft unimportable. Resolution now probes the
  GPU path before selecting it and falls back to the CPU backend, instead of choosing a
  backend it cannot load. This made 46 tests fail on any CPU-only machine, and would
  have made `batch_pdot` fail on every star at its own defaults.
- **A finite-time singularity would chase the worst point on a flat track.** Keeping
  `singular`'s `t0` merely *outside* the observed span is not enough: nothing stopped it
  converging to within a hair of the nearest point, where `b/(t - t0)` becomes a spike
  on that one measurement — a three-parameter outlier absorber that beat `constant` on
  BIC on a track with no period change in it at all. Found on a 12.7-year ASAS-SN EW
  binary whose period is constant to 5 parts in 10^8 per year, where `singular` was
  buying Δχ² = 21 from a single 3.8σ point. Two guards now: `t0` is held at least one
  median point-spacing from the data (closer than that the divergence is unresolvable by
  construction), and any fit whose variation is still dominated by a single end step is
  flagged `PeriodLawFit.at_bound` and ranked *below* every unpinned fit by
  `compare_period_laws` and `law="auto"`, whatever its BIC. The same check retires the
  manual caveat about `pejcha` winning with its exponent pinned at a bound — it is now
  detected rather than described. V1309 Sco's genuine singular fit is untouched, and the
  batch summary carries the flag as `law_at_bound`.
- **Harmonic-order selection over-fitted whenever the quoted errors were optimistic.**
  Weighted BIC (`chi2 + k*ln(n)`) is right when the errors are right and catastrophic
  when they are not — every harmonic looks like a bargain and the selection runs to
  `max_harmonics`. On OGLE's V1309 Sco photometry (chi2_reduced ≈ 115 against the stated
  errors) that produced a 10-harmonic template on 50-point intervals, whose jagged
  chi-squared then locked onto sampling aliases and scattered the P(t) track beyond use.
  `period_profile` now selects on the scale-free criterion `n*ln(chi2/n) + k*ln(n)`,
  against a frequency refined inside the interval, and takes the median over five
  intervals — which is stable at 2 harmonics across every interval size tried.
- **Inputs.** Masked entries are NaN on every input route — numpy masked arrays,
  astropy `MaskedColumn` / `Masked`, pandas NA, pyarrow nulls — so tuples and
  `LightCurve.from_arrays` no longer analyse the fill values under a mask; masked integer,
  boolean and short text columns load (they raised `Cannot convert fill_value nan` or
  `could not convert string to float: 'n'`). Compressed FITS (`.fits.gz`, `.bz2`, `.xz`)
  and `.fts` files are read as FITS; integer-labelled frames (`pd.read_csv(header=None)`)
  load with `ColumnMap(time=0, value=1, error=2)`; Parquet categorical columns with nulls
  no longer take a neighbouring category; conversion errors name the column. A
  placeholder error column (all 0, −1 or NaN) is reported as such by every method, the
  multi-band path, `sliding_periodogram`, `pdot_search`, `phase_track` and pre-whitening,
  instead of "no usable time baseline".
- **Backends.** Selectors ignore case and surrounding spaces and the unknown-backend
  error lists `auto` / `cpu` / `gpu` / `torch[:<device>]`; `is_gpu_backend("torch")` is
  true only when torch has a GPU, so a CPU-only torch run is no longer labelled GPU; a
  one-sample trial grid of the opposite kind no longer crashes the torch backends.
- **Peaks and spectra.** `downsample()` (and `to_dict(include_raw=True)`, batch
  `store_raw=True`) kept each bin's maximum for minimised statistics, erasing the PDM /
  CE / String-Length detection — it now keeps the minimum (spectra stored by earlier
  versions for these methods are affected); a NaN next to the true maximum no longer
  hides that peak or wins its bin; `best_periods(0)` returns no peak.
  `alias_diagnostics` no longer raises for runs shorter than ~26 minutes and never
  matches a predicted alias to the diagnosed peak itself (a false *ambiguous*).
- **GLS.** A period grid flagged uniform is evaluated by astropy instead of raising.
- **BLS.** The flux is median-centred in float64 (ppm-level float32 / MPS errors);
  durations round half away from zero as in astropy; the default grid no longer holds a
  period below `min_period_days` from an `np.arange` overshoot; a frequency grid
  containing 0 raises a clear `ValueError`; multi-band BLS resolves the torch device and
  records `torch:<device>`.
- **TLS.** An exactly constant flux gives an all-zero spectrum (rounding of the mean
  produced SDE values up to ~11); float32 backends no longer return a silent all-zero
  spectrum for fluxes in small or large physical units (rescaled internally; all outputs
  float64).
- **PDM, CE, String-Length, MHAOV.** MHAOV in float32 (torch; automatic on Apple MPS)
  returned an all-zero spectrum once the mean exceeded ~90× the scatter, and PDM's
  float32 path lost the within-bin variance (negative θ) — values are now centred on
  every backend. The CUDA String-Length kernel no longer raises for curves just above a
  shared-memory power of two (8193–8277 points on 99 KB devices, 4097–5290 on 64 KB). A
  NaN/inf trial frequency or a 0/NaN trial period gave the numba MHAOV kernel
  uninitialised values and could make the CUDA String-Length sort read out of bounds;
  the low-level kernels return NaN there. PDM, CE and String-Length on numpy/torch no
  longer use memory linear in the light-curve length (~4 GiB at 65k points). MHAOV's
  numpy backend no longer requires NumPy ≥ 2.0, and it reports `min_detections` and the
  harmonic-order limit separately.
- **Multi-band.** Multi-band BLS searches a user `grid=` instead of rebuilding its
  default. `min_detections` is enforced by multi-band PDM, CE, String-Length, MHAOV and
  BLS (batch runs now skip such objects). `GLSSettings(mb_fap_bootstrap=N)` is no longer
  ignored on the astropy backend or a non-uniform grid. Exactly constant bands give zero
  multi-band GLS power in every model (the offsets model reported up to 0.44, flex 1.0),
  and near-constant bands are left out of SuperSmoother and MHAOV. `MultibandFAP.fap(nan)`
  is NaN, `level(p)` is the order statistic `fap()` implies and no longer raises at the
  documented floor, `fap_level_10pct` is written only from 9 resamples, and a best peak
  on the grid edge gets a finite FAP; `n_bootstrap < 1` raises, and
  `multiband_fap(backend="auto")` honours `settings.backend`.
- **Pre-whitening.** No more giant spurious components where the sampling barely
  separates a sinusoid from the constant (near multiples of a regular cadence, near 1,
  2, … c/d for nightly data, f → 0): the spectrum reports the bounded power-equivalent
  amplitude there and the criteria judge such candidates on the variance they explain.
  Covariance errors no longer collapse to ~0 on long baselines, in flux units or with a
  distant `t_ref`. Sparse light curves no longer run away to `max_frequencies` with an
  interpolating solution, and constant or noise-free ones no longer yield round-off
  components. `fit_mean=False` no longer fits un-centred data. The correlation factor
  `D` uses the residuals in time order (row order changed it); analytic errors use the
  weighted rms of weighted fits. `SpectrumEngine`, `amplitude_spectrum`,
  `spectral_window` and `fit_multisine` raise `ValueError` for non-finite times, values
  or errors and non-positive errors (a NaN time crashed the interpreter inside finufft),
  and `fit_multisine` rejects non-finite or non-positive frequencies and unknown refine
  policies. Float32 direct-path spectra keep their phase accuracy at large f·T, and the
  default grid is size-checked before allocation. `identify_combinations` no longer
  reports order-1 `Fk = Fj` identities.
- **Period change: search and tracking.** `fit_ephemeris` pairs each timing error with
  its own event (unsorted lists with unequal errors gave a wrong `pdot` and an
  understated error) and drops a non-finite time with its error. `f_test_p` is NaN, not
  0, when both ephemerides fit exactly. `pdot_search` on fewer than ~80 cycles with the
  weighted epoch near one end no longer fails at its defaults (the axis is trimmed to
  the model's domain; under ~1 cycle it raises `InsufficientDataError`), and the default
  axis has exactly 201 rows (rounding gave 203 for ~30% of inputs). `pdot_search`,
  `phase_track` and `fit_ephemeris` reject a NaN or infinite period with a clear
  `ValueError`; `phase_track` explains when it cannot form the four segments it needs;
  `pdot_search(backend="auto")` honours the method's settings backend. Binned
  detrending visits only occupied bins (one stray epoch allocated gigabytes or
  overflowed), and `running_median`, `detrend` and `fit_ephemeris` treat masked entries
  as missing.
- **Period change: P(t) tracks and laws.** `period_profile` intervals no longer scatter
  2–4× beyond `period_err` for many-harmonic or high-S/N curves (which made `compare()`
  prefer a period change on constant stars); `PeriodPoint.amplitude` is the
  fundamental's semi-amplitude; unweighted errors no longer depend on the units; bad
  arguments raise `ValueError` and a constant light curve `InsufficientDataError`.
  `fit_period_law` / `compare_period_laws` without `period_err` always picked `constant`
  and quoted errors for σ = 1 d. `at_bound` measures the law against the track's
  resolution. `pejcha` errors include the exponent's dependence; unconstrained or
  bound-held parameters report NaN, not 0; unknown law names and mismatched lengths
  raise; `PeriodLawFit == PeriodLawFit` no longer raises.
- **Period change: wander, light time, sensitivity.** `eddington_plakidis` reports 0
  (not inf) for an exactly periodic series, rejects a NaN period or `t0`, forms no n×n
  matrices and compares by value. `ltte_search`'s default scan floor uses distinct
  epochs (repeated or clustered timings collapsed it), masked entries are dropped,
  multi-column input is refused, NaN periods, empty input and `minimum_third_mass(0)` are
  handled, and Kepler's equation is solved up to e → 1. `PdotSensitivity.sigma_amplitude`
  is inf for outer periods the epochs cannot constrain; `detectable_pdot` drops a NaN
  epoch together with its error and needs degree + 1 distinct epochs.
- **Period change: time-resolved maps.** `wwz` centres the values (a constant curve gave
  Z values of several hundred; large offsets corrupted cells), keeps rows whose unused
  error is blank or zero (AAVSO visual estimates were dropped), refines the ridge in the
  right direction on a descending frequency axis and copies explicit axes.
  `sliding_periodogram` honours the method's settings backend, keeps its frequency axis
  ascending (a grid stored in another order mirrored every row) and no longer drops the
  last observation when the final window ends one ulp short (~2% of light curves).
- **Period change: batch and figures.** `batch_pdot(period=...)` honours NumPy scalars,
  0-d arrays, pandas Series and integer-keyed mappings (they were ignored, so every star
  ran a blind search and could lock onto an alias with status `ok`); failed rows record
  the light curve's `n_samples` and `baseline`; figures of flux light curves are no
  longer drawn inverted and labelled magnitude; distinct keys that sanitise to the same
  file name (`fieldA/lc_0000` vs `fieldA_lc_0000`) get a digest suffix instead of
  overwriting each other's figures. `cuperiod.pdot.plots` honours masks, its
  drift-corrected fold uses the track's fitted ephemeris, `plot_phase_fold` raises for an
  unknown model and says *(unwarped)* when the rate is outside the model's domain.
  `cuperiod pdot --profile` no longer prints negative ΔBIC for lower-ranked laws.
- **Batch.** An existing path containing `[` is no longer read as a glob, and a glob
  matching nothing is no longer a silent success (with `--no-resume` it emptied the
  sink). `(key, light curve)` pairs with NumPy integer keys are accepted. An unguarded
  script running a pooled batch fails with a message explaining
  `if __name__ == "__main__":` and `workers=1` instead of pages of child tracebacks and a
  bare `BrokenProcessPool`. A file-sink resume recomputed every light curve and then
  discarded the results (`batch_prewhiten` still counted them as done). An existing
  directory or uncreatable location given as a file sink is refused before the batch
  instead of after it, and a failed final rename names the temporary file holding the
  result. A directory whose earlier run failed entirely is not locked to that request.
  `cuperiod run --out` saved different BLS peaks from the ones it printed.
- **LINCC interop.** `nested_periodogram` on a zero-row frame returns typed float64
  result columns; `backend="auto"` defers to the settings' backend and the `period_i`
  columns follow the settings' peak fields, as in `cup.periodogram`.
- **GUI: data and caching.** Opening another curve while a computation runs no longer
  applies the old run's spectrum, fold or error to the new one, and returning to a
  source whose run is in flight no longer computes it twice. A band column splits the
  curve in every format (it was ignored for ECSV, FITS and ASCII), each file is parsed
  once, rows without a label are reported, and a band column with more than 32 labels no
  longer freezes the window. NaN timestamps (gap cadences in Kepler/TESS files) no longer
  blank the fold or turn the grid hint into `nan`. A band named `combined` or `stacked`
  no longer takes over the all-bands choice. Same-named files and files rewritten on
  disk no longer share cached results. Number fields keep full precision. The file
  preview shows full-precision times, counts rows correctly and flags header-only and
  wide one-row-per-object files. A batch source with no data rows is reported as such.
- **GUI: navigation and window.** Prev/Next and the arrow keys skip a batch source that
  failed to load; a new batch whose first source fails no longer shows the previous
  batch; loading a batch clears the source filter. The Sources pane no longer comes back
  empty on the next launch, and a corrupt saved window state no longer stops the window
  from opening. Dropped files load after the drop returns (Explorer froze behind the
  preview), and mixed drops load what they can. Closing the window during a computation
  no longer freezes it for 2 s or leaves the process running. The *runs on* hint and the
  device badge say CPU for a torch CPU run (and *GPU (MPS)* / *GPU (XPU)* for other
  devices); the grid hint describes the band that will be analysed; guidance text names
  the button that is actually there; the Options fields follow the pickers in Tab order;
  right-click menus no longer block the window.
- **GUI: plots and tables.** The spectrum is drawn end to end on a period or log axis
  (index-based decimation dropped the long-period end and its peaks) and re-binned on
  zoom; the phased and raw views draw around NaN values and accept empty curves; stop
  reasons containing `<` show verbatim; a BLS/TLS marker click keeps the peak's t0 and
  the tables follow selections made in the spectrum; PNG exports append `.png` and report
  failures, as CSV exports do; the Peaks table keeps locale-independent formats (a t0
  near JD 2459001.9 read `2.459e+06`) and exports full precision; the Frequencies ID
  column sorts F1…F12 in order; Ctrl+C in a table acts only while it has the focus and
  Tab leaves it; échelle points follow a theme switch; strokes keep their width on HiDPI
  screens; the crosshair readout no longer resizes the panes; log-axis tick labels no
  longer overprint, the top peak's label is not clipped, and legends have a background.
- **GUI: keyboard and feedback.** Tab no longer gets stuck in the Options after Run /
  Compute, the Options scroll to the field it lands on, and tab-bar scroll arrows stay
  out of the Tab order. Step says so in the status bar when it adds nothing (rejected,
  blocked or at the stopping rule), and so do Renumber, Restore all and an Improve with
  nothing to do; refusals name the window's own controls (*Add exactly here*, untie
  first) instead of keyword arguments.
- **GUI: layout.** The Subsets, Variations and Period-spacing panes and the session bar
  wrap their controls instead of hiding them in a narrow dock; zooming keeps the status
  chips, the Run button and the Frequencies ID column whole, and the window's minimum
  size follows what its panes need; the spectrum, fold, Peaks and Frequencies *Export…*
  buttons use the window's file dialog. Smaller: the spectrum readout keeps its start
  and reads c/d, no empty period badge, "no components yet", automatic option values to
  six significant digits, checked boxes in disabled tables keep their tick, and the
  import dialog no longer claims a header row a file does not have.

### Security

- **Pre-whitening projects are untrusted input.** A `*.cupw.json` file is read as JSON
  only (no pickle, no `eval`), size-checked before it is read (64 MiB by default), with
  `NaN` / `Infinity`, overflowing numbers and duplicate keys refused, every object
  validated by a strict model (`extra="forbid"`, no coercion), every cross reference
  checked, the log length and point count capped, and data references confined to the
  project's folder; anything wrong raises `ProjectError`. Tie expressions are parsed,
  never evaluated, and exported scripts carry every value as a Python literal and free
  text only as sanitised comments.
- The core dependency floor is `pyarrow>=14.0.1`: earlier releases unpickle
  extension-type metadata while reading Parquet / IPC files (CVE-2023-47248), and
  cuPeriod opens user-supplied `.parquet` light curves in the CLI, the batch runner and
  the GUI.
- **Batch sinks.** CSV sinks escape text cells a spreadsheet would run as formulas (`=`,
  `+`, `-`, `@`, tab, CR; plain numbers untouched); an unreadable existing sink or part
  is moved aside to `<name>.corrupt-<timestamp>` instead of being overwritten; sinks are
  written through a unique, exclusively created, dot-prefixed temporary file, never a
  fixed `<sink>.tmp`.
- **GUI.** File-derived text (column headers, band names, file names, errors quoting
  them) is shown as plain text in banners, dialogs, previews, tooltips and plot legends,
  never as Qt rich text, so a header such as `<img src=//host/...>` is neither rendered
  nor fetched. The header revision is read from cuPeriod's own checkout without running
  git (on Windows a `git.exe` in the launch folder was executed), and demo data are
  looked up only in the checkout or `CUPERIOD_EXAMPLE_DATA`, never in ancestor folders.
- The Windows CUDA DLL shim adds only the directories of installed nvidia wheels; a
  folder named `nvidia` next to a script can no longer supply DLLs.
- **CI and releases.** Every GitHub Action is pinned to a full commit SHA and kept current
  by Dependabot; both workflows default to a read-only token and checkout no longer
  leaves the token in `.git/config`; the release build pins uv (0.11.20, tarball SHA-256
  verified), twine (7.0.0, dependencies frozen with `--exclude-newer`) and the hatchling
  build backend by hash (`.github/build-constraints.txt`), and restores no cache.
- **Benchmark downloads.** `download_ogle_triples.py` and `download_v1309sco.py` fetch
  the OGLE archive over HTTPS instead of plaintext FTP, `download_s82_rrlyrae.py` reads
  astroML-data at a pinned commit with a 120 s timeout, and all three verify a pinned
  SHA-256 for every file, on download and for cached copies.

### Performance

- **Pre-whitening** runs about 1.7× faster end to end (amplitude-only in-loop spectra
  with cached normal-equation terms, guarded Cholesky solves with factor reuse, reused
  designs across cyclic sweeps, window-only engines, no CUDA probing for explicit CPU
  backends, a vectorised combination search). CPU spectra of small transforms (points +
  frequencies ≤ 150 000) run on one thread, up to ~7× faster per spectrum. Constrained
  fits project zero points per group out of every solve: 3.5–6.4× faster with 100
  nights (N = 2e4–1e5), 18× with 300 groups, identical to round-off.
- **GLS**: small finufft transforms run on one thread (4–11× faster single-curve GLS
  and `pdot_search` on many-core CPUs); `FinufftGLS` reuses its plan across scans,
  sliding windows and batches; the multi-band offsets model computes its per-band sums in
  one batched transform (1.6–2.5×).
- **BLS**: the numba kernel screens out boxes that provably cannot win — 3.5–6× faster on
  sparse survey light curves, bit-identical — and the numpy/torch batch is capped by a
  memory budget (512 MiB host, 1 GiB device).
- **Fold methods**: numpy PDM 2.4–2.8×, CE 2.2–3.4× and String-Length 1.2–1.3× faster
  (bit-identical folds); numba String-Length 2.3–4.6× (an O(N) stable bucket sort);
  MHAOV on uniform grids sums its harmonics by type-1 NUFFT (numpy 2–90×, numba 4–6×
  above 64 points per thread; agrees to ~1e-11). Trial-period chunks are capped at
  ~128 MiB host / 512 MiB device.
- **TLS**: the numba kernel is 2.2–4.5× faster (bit-identical to numpy) and the
  numpy/torch batch is memory-capped (N = 80 000: 629 → 117 MiB peak). **SuperSmoother**:
  the numba kernel is 1.5–2.1× faster, and its on-disk cache now works across processes
  — every CLI run, GUI session and batch worker spent 4–9 s recompiling it.
- **Period change**: `phase_track` 6–28× faster (Gram-form shift scan, shared
  trigonometry, one QR for the whole harmonic ladder); `running_median` 15–20× (sliding
  sorted window); singular and Pejcha laws fitted in milliseconds instead of 1–4 s;
  `period_profile`'s interval scan 1.4–2.7×; `ltte_search`'s scan 3–5×; `wwz` 50–85×
  with numba and 2.7–7× on the NumPy path.
- **Core**: backend discovery is memoised per process (~1 ms → ~1 µs; small PDM/CE
  periodograms up to 3× faster), explicit CPU requests no longer import cupy (~0.25 s per
  process or batch worker), default settings are no longer rebuilt from the environment
  on every call, and peak-preserving downsampling is vectorised (12–23×; batch rows with
  `store_raw=True` ~12×).
- **Batch**: CPU pools spread small batches over all workers (16 light curves on 4
  workers: `batch_pdot` 13.9 s → 2.2 s), pin BLAS/OpenMP threads only while the pool
  exists and give numba `cores // workers` threads per worker. The nested-pandas
  partition tier no longer imports cupy for a CPU GLS partition.
- **GUI**: light curves above 20 000 points are drawn as a fixed random subset and the
  hidden raw panel builds only when shown (refolds of 1e5–5e5-point curves: 0.2–2 s →
  ~30 ms); the Load demo menu fills when first opened (~0.8 s off start-up); the lasso
  hit test stays well under a second on million-point curves.

### Validated

- **End-to-end injection-recovery of the dP/dt chain**
  (`benchmarks/pdot_recovery.py`): 60 two-harmonic stars over 1500 days with drifts
  spanning 0.5–8 accumulated cycles, plus 60 drift-free nulls. The search localized
  the planted (P, dP/dt) in **60/60** stars, the tracker's F-test detected **60/60**
  at p < 1e-3 with **0/60** false detections on the nulls, and the pull distributions
  are calibrated (null rms 1.07, injected 0.8–1.3). Median cost per star on one CPU
  core: 4.1 s for the 201×601 search, 0.06 s for the tracking measurement.
- **Reproducing a published period-decay law on real data**
  (`benchmarks/pdot_v1309sco.py`, `download_v1309sco.py`): V1309 Sco, the contact binary
  OGLE watched spiral in until it merged as a red nova in 2008 — the one merger with
  dense photometry of the *progenitor*, and a deliberately hostile test (the decay is
  singular, so a constant rate is knowingly wrong; the star brightens 0.8 mag as it
  inspirals; OGLE's formal errors understate the scatter about tenfold; and the
  light-curve shape itself evolves). Tylenda et al. (2011) fitted
  `P = 1.4456 exp(15.29/(t - 2455233.5))` d to periods measured in subsamples of ~50
  observations, and state (their Sect. 6.2) that the period decreased by 1.2% over six
  years; Pejcha (2014) — not Tylenda et al., who quote no rate — is the source of the
  widely repeated decay timescale falling from ~1000 yr to ~170 yr.

  Running the same two-step recipe on the authors' own photometry (detrend,
  `period_profile` at 50 points per interval, `compare_period_laws`): BIC picks
  `singular` over a constant rate by **ΔBIC = 198**, the measured P(t) points scatter
  about the published Eq. (1) by **0.21% of P** with a +0.055% mean offset, and the two
  fitted P(t) curves differ by at most **0.33% of P** anywhere in the observed span. The
  total change is **−1.47%** from the law and −1.06% from the endpoint intervals against
  the published −1.2% (and −1.14% from integrating Eq. (1) over this exact window); the
  decay timescale runs **869 → 185 yr** against Eq. (1)'s 1730 → 156 and the quoted
  ~1000 → ~170. Repeating at 35/40/50/60 points per interval spans −1.06% to −1.47%,
  which is the method's own systematic and is reported rather than hidden. Forcing a
  single constant rate instead returns −1.91%, and the tools say why: the F-test is
  decisive (p = 3e-8) while the O−C residuals fall 194 → 89 → 16 minutes from a
  quadratic to a quartic ephemeris.

  The `singular` law's `b` and `t0` are individually degenerate here (they trade off
  strongly, and the fit reports errors that say so) while the P(t) they predict over the
  observed span is not — which is why the comparison above is made on the predictions.
- **The cyclic O−C: light-travel-time orbits** (`cuperiod.ltte_search`,
  `LTTESearchResult`, `LTTEFit`, `PhaseTrackResult.ltte_search`): a distant third
  body makes the O−C *periodic* through the light-travel time across the outer
  orbit (Irwin 1952), and when that outer period is comparable to or shorter than
  the baseline the O−C is not monotonic — the rate a parabola reports is then a
  baseline-weighted average of a wobble whose value and sign depend on which part
  of the cycle the data cover. Hong et al. (2022) found 307 such systems (plus 104
  with a parabola as well) among 14,127 OGLE bulge contact binaries, at outer
  periods of 5–14 yr on a 15-yr baseline. `ltte_search` fits the cycle *jointly*
  with the ephemeris polynomial: at every trial outer period the model is linear in
  `A cos ω`, `A sin ω` and the polynomial (a floating-polynomial periodogram of the
  O−C), the best trial is polished by nonlinear least squares — with Irwin's
  eccentric elements freed on request — and the result carries the outer period,
  amplitude, `a sin i`, mass function and minimum third mass, the *secular* `dP/dt`
  with the cycle removed next to the parabola-only one (`pdot_change`), ΔBIC and an
  F-test against the polynomial alone, and the `n_cycles` / `at_grid_edge` flags
  that mark the degenerate regime. It accepts a `phase_track` result, a timing list,
  or explicit O−C values; the batch chain runs it per star (`ltte=`, `ltte_*`
  columns), the CLI exposes `--ltte/--no-ltte`, and the O−C panel of the diagnostic
  figure overlays the orbit. Planted circular and eccentric orbits are recovered
  within their errors against an independent Kepler solver, and a planted 800-day
  cycle that the parabola alone turned into a 2σ "drift" comes out null in the joint
  fit (`tests/test_pdot_ltte.py`).
- **What a rate means physically, and how small a one is measurable**
  (`cuperiod.mass_loss_rate`, `mass_transfer_rate`, `pdot_from_mass_loss`,
  `pdot_from_mass_transfer`, `detectable_pdot`, `PdotSensitivity`, `convert_pdot`;
  `pdot_d_per_yr` on every result): the Jeans-mode wind relation
  `(dP/dt)/P = 2 mdot/M` — `dP/dt = 2e-6 (P/yr)(mdot/1e-6 Msun/yr)(Msun/M)` — and
  Kwee's (1958) conservative-transfer relation `(dP/dt)/P = 3 mdot (1/M_d − 1/M_a)`,
  both directions, with the sign checks that Schaefer (2024) found failing for 44% of
  cataclysmic variables built in (a negative implied loss or transfer rate is
  reported as such, not silently absolute-valued). `detectable_pdot` is the formal
  floor of the quadratic-ephemeris method, `σ(dP/dt) ≈ 27 P σ_t / (T² √n)` for uniform
  sampling or from the actual epochs, plus the light-time amplitude floor at a given
  outer period — the 0.1-cycle rule of thumb overestimates the floor by two orders
  of magnitude because eclipse timings reach 1e-3 P and there are dozens of them,
  which is why survey detections sit at ~1e-6 d/yr (3e-9 s/s). Pinned against a Monte
  Carlo of the fit and the closed form (`tests/test_pdot_physics.py`).
- **Symmetry-aware cycle unwrapping in `phase_track`** (`PhaseTrackResult.symmetry_order`):
  a contact binary with equal minima has a template that is almost perfectly
  symmetric under a half-period shift, so a segment's best shift lands on either
  branch by noise and the O−C jumped by P/2 between segments (on four of the six
  OGLE stars below, for up to two thirds of the segments, which the template
  re-derivation then turned into hour-scale garbage). The
  tracker now measures the template's fold symmetry (harmonics outside the multiples
  of `m` below 5% of the power) and unwraps consecutive delays by `period / m`;
  `wrap_risk` is scaled the same way. Found on real data by the OGLE benchmark below.
- **Light-time orbits on real OGLE triples** (`benchmarks/pdot_ogle_triples.py`,
  `download_ogle_triples.py`; write-up with figures in
  `benchmarks/REPORT_ogle_triples.md`): six of Hong et al. (2022)'s sinusoidal contact
  binaries, 14.3 yr of OGLE-III + IV photometry each, run through detrend →
  `phase_track` (40 segments) → `ltte_search` with no per-season binary modelling.
  **6/6** recovered outer periods agree with the published ones within 2σ (median
  offset 4%; e.g. 6.43 ± 0.09 yr vs 6.2 ± 0.2 for OGLE-BLG-ECL-289126, amplitude
  660 ± 17 s, ΔBIC +2085), while the parabola alone claimed a ≥ 3σ dP/dt on **3/6**
  and the joint fit's secular rate on 1/6.
- **Injection-recovery of the cyclic O−C** (`benchmarks/pdot_ltte_recovery.py`): 80
  planted 120–600 s orbits at outer periods of 0.25–1.5 baselines plus 40 nulls. With
  the cycle covered (≤ 1 baseline), the outer period is recovered within 10% in 81%
  and the amplitude within 3σ in 98%; the parabola alone reports a ≥ 3σ "dP/dt" on
  41% of stars that have no secular rate at all (64% when the cycle is 0.75–1
  baselines long) against 20% for the joint fit, and there are no false cycles and
  no false rates on the nulls. Beyond one baseline the cycle is detected in 12%: the
  degenerate regime, stated as such.

### Known limitations

- The (P, dP/dt) map is a localization tool, not an inference tool: the wrapped
  statistic's false-alarm calibration on a 2-D grid over irregular sampling is an
  open problem (the independent-trials count is unknown), so significance always
  comes from `phase_track`'s F-test. The GUI does not yet expose the period-change
  tools. The *search* scans one rate parameter at a time (constant dP/dt or constant
  d ln P/dt); laws with a characteristic epoch, such as `singular`, are fitted to a
  measured P(t) track rather than searched, and for wander the right tool is still a
  time-resolved map. Period-law parameters can be strongly degenerate on short
  baselines even where their predictions are not — read the reported errors.
- **Interactive pre-whitening**: Period04 frequency-list import and export are not
  supported; after `load` only the stored state is in memory (the rest of a project's
  history is re-created by replaying it, as *Verify project* does); and a session's
  spectrum backend is fixed when it is created (`load` / `restore` / `replay` take
  `backend=`).
- Several published benchmark figures predate this release's fixes and are being
  re-measured: the batch throughput (it ignored `grid=`, and its "CPU pool" arm ran on
  the GPU on CUDA hosts), the TLS and SuperSmoother timings, and the multi-band MHAOV
  results.
- The fold methods' default grids are not exactly invariant to the time origin: the last
  sample sits on a phase-bin edge, so JD and MJD inputs differ at the O(1/N) level.
- `BatchSettings` is reserved: no batch driver reads it yet, so `CUPERIOD_BATCH_*` has no
  effect.

## [1.2.0] - 2026-08-30

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
