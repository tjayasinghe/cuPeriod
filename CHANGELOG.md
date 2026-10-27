# Changelog

All notable changes to cuPeriod are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Everything below is measured against 1.2.0. Interactive pre-whitening (sessions,
projects, constrained fits, partitions), the period-change tools (`cuperiod.pdot`), the
GUI's import dialog, menu bar and session panes, and the `cuperiod pdot`, `pdot-batch`
and `session` commands are new in this release and are described under Added as they
now behave; Changed, Fixed, Performance and Deprecated list only what moves for code
and data that worked with 1.2.0.

### Added

#### Interactive pre-whitening

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
  refused before it commits, and `goto` says why a state is out of reach (no such log
  entry, evicted by `SessionSettings.undo_depth`, or older than the project file it was
  opened from). One entry point, `do(op, **args)`, never raises for bad input (an
  unknown setting name or an invalid value included): it returns a `refused`
  `StepReport` and changes nothing. An explicit `grid=(f0, df, nf)` is size-checked
  (`GridTooLargeError`) before anything is allocated.
- **Errors from Calculate and Improve.** Calculate (F8, `PreWhitenSession.calculate()`)
  and Improve (all or selected) solve with the covariance, so the frequency table shows
  frequency, amplitude and phase errors straight away (0 for a fixed quantity); the
  automatic re-calculation after each edit stays fast and shows "–" until Calculate or
  Improve. Each action's history line says it once ("Restored 22 points", not "Restore
  22 point(s): restored 22 points").
- **`cup.prewhiten` now runs on `PreWhitenSession`**: a session run to its stopping rule
  and finalized. Moving it onto the session changed no numbers (a 47-case golden corpus
  holds the automatic loop's results bitwise), but other changes in this release do
  move pre-whitening results: see *Default bands on regular cadences*, *Multi-exposure
  ceiling*, *Pre-whitening uncertainties* and *Pre-whitening details* under Changed and
  *Pre-whitening* under Fixed.
- **Bitwise replay on any backend.** Steps log their peak seed and grid index, so
  `run(seeds=...)`, `PreWhitenSession.replay()` and `load(verify=True)` repeat a session
  exactly on any spectrum backend (a finufft session verifies on numpy).
- **Pre-whitening projects** (`*.cupw.json`, schema version 1): `save()` / `load()`
  write and read the creation arguments, the current solution and the seeded action
  log, with the light curve embedded or referenced next to the project and pinned by a
  SHA-256. A project stores the log, not the undo states: after `load` only the stored
  state is in memory, and the earlier ones are re-created by replaying the log.
  `snapshot()` / `restore()` keep and re-attach a session in memory, undo states
  included, without a re-run.
- **Projects are read as untrusted input.** A `*.cupw.json` file is read as JSON only
  (no pickle, no `eval`), size-checked before it is read (64 MiB by default), with
  `NaN` / `Infinity`, overflowing numbers and duplicate keys refused, every object
  validated by a strict model (`extra="forbid"`, no coercion), every cross reference
  checked, the log length, point count and frequency grid capped (64 grid frequencies
  per data point, but never fewer than 20 million or more than 100 million), and data
  references confined to the project's folder; anything wrong raises `ProjectError`,
  and `save()` refuses a session that `load` would refuse. The source path an embedded
  project records is kept as `meta["recorded_source"]`, a record that is never re-read;
  a URL, a network (UNC) path or a drive-relative path is refused before any filesystem
  call. A project recorded on the direct `numpy` backend whose spectrum would cost more
  than 2e9 point-frequency evaluations opens on the NUFFT CPU backend, with a
  `SessionWarning`, unless `backend=` is given. Tie expressions are parsed, never
  evaluated, and exported scripts carry every value as a Python literal and free text
  only as sanitised comments.
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
  `should_stop=`. A parameter the data leave undetermined (say, the common phase of a
  component whose per-subset amplitudes solve to round-off) is NaN with the
  `degenerate` flag and is left out of `FitDetail.rank`, in the fit and in the analytic
  and bootstrap errors alike.
- **Partitions** (`cuperiod.Partition`, `cuperiod.prewhiten.suggest_partition`): nights,
  TESS sectors, chunks, seasons, label columns, time ranges and crossings of the points,
  with the recipe recorded and hand edits kept run by run.
- **Result additions**: `Sinusoid` gains `fixed`, `tied`, `variation`, `amplitude_rms`
  and `flags`; `PreWhitenResult` gains `subsets` (`SubsetSolution`: zero points and
  per-subset amplitudes and phases with errors) and `n_deleted`; `to_table(constraints=)`
  adds the constraint columns only when a component carries a constraint, so the
  automatic loop's tables keep the columns they had in 1.2.0.
  `AmplitudeSpectrum.noise_curve()` gives the local noise at every grid frequency;
  `SpectrumEngine.release()` drops GPU plans.
- `cup.SessionSettings` (`CUPERIOD_SESSION_*`): undo depth, caches, implicit
  recalculation, prune protection and project limits. `PreWhitenSession.from_bands()`
  pre-whitens all bands merged (each centred, as the GUI does) with a root `band`
  partition, via `cuperiod.core.lightcurve.stack_bands()`.
- **Refusal hints.** A refused `StepReport` whose message names a remedy in keyword
  arguments also carries it as a code in `StepReport.hint` (`"allow_close"`,
  `"cascade"`, `"exact"` for `refine="none"`, `"fit_mean"`; `REFUSAL_HINTS`,
  `SessionError(..., hint=...)`), so a front end can phrase its own advice without
  parsing the message, which is unchanged.
- **CLI**: `cuperiod prewhiten --save-project FILE --export-script FILE`, and a
  `cuperiod session` group — `verify` (replay a project's log against its stored
  solution), `script` (export a reproducing script) and `replay` (re-create a project,
  optionally saving the replayed log and solution as a new project with `-o`). Their
  `--data FILE` reads the file with the project's recorded column mapping and domain
  (the content hash is still checked). `--save-project` embeds a data file kept outside
  the project's folder even above `embed_data_max_points` (200 000), with a note
  saying so, and `--export-script` is written even when the project save fails.

#### GUI

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
  with its constraints. Step, Renumber, Restore all and Improve say so in the status bar
  when they have nothing to do.
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
  the title bar reads `star.csv — star.cupw.json* — cuPeriod` (the `*` marks unsaved
  work; macOS shows its own dot instead); a project whose data file moved asks for it.
  Warnings raised while opening a project (a recorded backend that is not available
  here, values re-fitted) appear in the status bar and beside the frequency table, and a
  project that cannot be opened or verified says so without marking the run failed.
  Unsaved sessions survive switching source, band or analysis (File ▸ Unsaved
  sessions), and closing, New session or opening other data asks Save / Discard /
  Cancel; both name a demo or batch source by its label, not its key.
- **GUI: refused edits offer their remedy.** When the session refuses an edit the window
  can fix itself, it asks without blocking and does it on a yes, as one undoable action:
  removing a component others are tied to (*Dissolve their ties and remove F1?*), a
  frequency or tie inside another component's collision guard (*Add anyway* / *Tie
  anyway*), a frequency outside the grid or one the refinement would move onto another
  component (*Add it exactly, no refinement?*). The wording comes from
  `StepReport.hint` (documented in the interactive pre-whitening guide); a yes that
  arrives after the session changed does nothing and says so.
- **GUI: menu bar and command registry.** File, Edit, Pre-whitening, Data, View and Help,
  with Alt mnemonics on every menu and item, built from one registry that also drives
  the buttons, the keyboard shortcuts and the F1 sheet; File gains Open folder
  (Ctrl+Shift+F), Open recent, Load demo, Unsaved sessions, Verify project and Quit
  (Ctrl+Q); View gains Analysis, Panes, Show raw light curve (Ctrl+R), theme (Ctrl+T),
  Zoom and Reset layout; Help has Keyboard shortcuts (F1), What's This? (Shift+F1),
  Documentation and About. The F1 sheet lists every shortcut, the spectrum's,
  light curve's and tables' own keys included.
- **GUI: Stop (Esc).** Ends Auto after its current step and keeps what it found; releases
  a running periodogram at once (the run finishes in the background and is cached, not
  shown). The info bar says what runs and for how long.
- **GUI: import dialog** for Open, drops, Open recent and a batch file that failed to
  load (*Choose columns…*), replacing the read-only preview: time / brightness / error /
  band pickers, magnitudes or fluxes, the delimiter and header row of text tables, an
  *As loaded* tab at full precision, a note on the time scale the numbers look like, and
  Load disabled with the reason while the choice cannot work (a header row read as data
  included). The file's own detection comes first: the read options and the
  magnitude/flux choice of the previous import are only fallbacks, and a column named
  as flux or magnitude always decides the domain. Headerless JD/mag/err tables (OGLE,
  Period04), semicolon-separated files and unrecognised headers such as `HJD, V, e_V`
  now load.
- **GUI: keyboard-operable spectrum.** A click folds at the nearest peak within 12 px
  (Alt: exactly there), also with the markers hidden; Left / Right move between local
  maxima (Ctrl: marked peaks, Shift: one sample), Home, Enter, + / − / 0 and the Menu
  key work on the focused plot, which screen readers describe. The phased view gains ×2,
  ÷2 and a typed period.
- **GUI: accessibility and zoom.** A visible 2 px focus ring on every control; WCAG 2.1
  AA contrast in both themes; drawn arrows, ticks and grips and larger pointer targets;
  Zoom in / out / reset (Ctrl++ or Ctrl+=, Ctrl+−, Ctrl+0) for text and plot fonts,
  remembered (an application font above 13 px also enlarges the interface; zooming
  keeps the status chips, the Run button and the Frequencies ID column whole); band
  colours that stay distinct under colour-vision deficiencies plus a marker shape per
  band (`ThemePalette.band_symbol()`, `ThemePalette.residual`); accessible names, status
  tips and What's This? help on every option, and a help line under the Options form;
  24 px pane Float / Close buttons; panes hidden behind a tab leave the Tab order;
  Ctrl+1 … 5 focus the panes. New theme API: `apply_theme(..., font_scale=)`,
  `stylesheet()`, `font_scale()`, `step_font_scale()`, `contrast_ratio()`.

#### Periodograms, light curves and batches

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
- **Multi-band BLS with a shared ephemeris.** `BLSSettings(mb_ephemeris="shared")` places
  one box (duration and mid-transit time on a common phase grid) in every band, fits
  each band's own depth, and maximises `Σ_k max(SNR_k, 0)²` over that one box, so noise
  maxima at unrelated phases in different bands no longer add up. On 40 simulated
  six-band light curves with a 1.5% transit in 0.5% noise it recovers 39 periods where
  the independent stack recovers 24. numba, numpy and torch agree bit for bit, cupy to
  1e-15; astropy is refused. The default (`"independent"`) is unchanged, and every
  multi-band BLS result records which one ran in `meta["mb_ephemeris"]`
  (`"independent"` or `"shared"`) and, in `extras["band_index"]`, the band with the
  largest SNR, whose depth it reports (an index into `meta["bands"]`).
- `BatchSummary.n_workers`; `--chunk-size` on `cuperiod batch` and `batch-prewhiten`.
- `cuperiod.core` re-exports `PdotSettings`, `PreWhitenSettings`, `SpacingSettings`,
  `SuperSmootherSettings` and `SessionSettings`.

#### Period change (`cuperiod.pdot`)

- **Joint (period, dP/dt) searching for every method** (`cuperiod.pdot_search`,
  `cuperiod.PdotSettings`): a drifting period defocuses a constant-period search, and a
  constant dP/dt has exactly the phase `f * ln(1 + (Pdot/P) * dt) / (Pdot/P)` (Mikulášek
  et al. 2012 — the same logarithmic phase PHOEBE integrates for its `dpdt`), so
  evaluating an *ordinary* periodogram on warped times scans a dP/dt axis with no new
  kernels: GLS, PDM, CE, String-Length, MHAOV, and SuperSmoother — single- and
  multi-band, every backend — all become period-derivative searches. On the GPU every
  warped row is computed on the device for GLS (cached cufinufft plans) and PDM (cupy),
  and cupy and CPU rows agree to 1e-7. Both axes are stepped to equal phase smear, the
  dP/dt step additionally refines by a per-method harmonic scale (sharp phase structure
  smears sooner — the criterion of the one prior art, Katz et al. 2021's
  conditional-entropy GPU search), and the default range is set in accumulated drift
  *cycles*, making the row count baseline-independent (201 rows for GLS at the
  defaults). The axis is trimmed to the period model's domain (so a short run with its
  weighted epoch near one end still searches; under about one cycle of baseline it
  raises `InsufficientDataError`), and an explicit `pdot_over_p_range` spanning zero is
  laid on a zero-anchored lattice with an exact dP/dt = 0 row. `backend="auto"` honours
  the method's settings backend. On a planted 1.1-cycle drift the constant-period row
  keeps GLS power 0.19 where the correct cell shows 0.95; the refined best cell recovers
  the planted (P, dP/dt) to grid precision (`tests/test_pdot_scan.py`, including sign
  conventions, the 160-cycle exact-warp regression, and CPU/GPU/torch parity).
- **The search map's tilted ridge, measured** (`PdotBest.ridge_correlation`): the
  (period, dP/dt) axes are not orthogonal for skewed sampling. Referencing to the
  weighted mean epoch zeroes the first moment of the times, which decouples phase from
  frequency, but the residual frequency–drift correlation is set by the sampling's
  *third* central moment and vanishes only for observations symmetric about that epoch
  (V1309 Sco, skew −1.29, shows the tilt plainly). The tilt widens the joint uncertainty
  without biasing the best cell; `ridge_correlation` is the peak's own error ellipse
  from the local curvature, and can differ in sign from the visually dominant tilt of
  the alias comb crossing the whole map, which is documented because the figures invite
  exactly that confusion.
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
  the ephemeris ladder below measures (`tests/test_pdot_models.py`).
- **Automated O−C analysis** (`cuperiod.phase_track`, `cuperiod.fit_ephemeris`): the
  classical observed-minus-calculated method (Sterken 2005) without hand-collected
  minima — one multi-harmonic template at a fixed period, its order chosen by the
  scale-free BIC, per-segment rigid template time shifts with Fisher-information
  uncertainties (timing precision comes from the light curve's *slopes*, so eclipses and
  sawtooths time better than sines), sequential cycle-count unwrapping with a
  `wrap_risk` flag, and weighted linear-vs-quadratic ephemeris fits reporting
  `Pdot = 2 P c2` with its formal error, the Pringle (1975) F-test, and AIC/BIC
  (convention documented). The template is re-derived on drift-corrected times until
  self-consistent, so an uncorrected drift cannot smear its own measurement, and results
  are always referenced to the constant-period ephemeris. A template that is almost
  symmetric under a fraction of the period (a contact binary with equal minima looks
  the same half a period later) is detected — harmonics outside the multiples of `m`
  below 5% of the power — and consecutive delays are unwrapped by `period / m`
  (`PhaseTrackResult.symmetry_order`; `wrap_risk` scales the same way), so the O−C does
  not jump by P/2 between segments. `phase_track` says why when it cannot form the four
  segments it needs. A 25-realization Monte Carlo calibrates the reported sigma (pull
  scatter 0.72–0.93, i.e. mildly conservative). `fit_ephemeris` accepts plain event-time
  lists (nearest-integer cycles, `worst_phase_residual` ambiguity diagnostic; each
  timing error stays with its own event, and a non-finite time is dropped with its
  error); without `errors=` it estimates the timing error from the scatter about the
  fitted quadratic, so `chi2_reduced` is 1 by construction. `f_test_p` is NaN when no
  test is possible. Its unit conversions are pinned against WASP-12b's published
  −29 ms/yr — guarding the exact multiply-vs-divide mistake that reached a refereed
  decay claim (`tests/test_pdot_track.py`).
- **Second period derivatives**: `EphemerisFit` describes a polynomial ephemeris of any
  degree; from degree 3 it reports `pddot` (the second period derivative, from
  `P*(6*c3 + 8*c2²)`) with its error and `pddot_err_scaled`, plus `implied_model` —
  whether the fitted curvature favours a constant rate (`pddot = 0`) or an exponential
  one (`pddot = pdot²/P`). `pdot` from a cubic fit is still the instantaneous rate at
  the reference epoch but is de-biased against a rate that is itself changing.
  `fit_ephemeris` and `phase_track` fit degrees 1 and 2; a cubic comes from
  `ltte_search(..., degree=3)` (or `PdotSettings.ltte_degree`), whose
  `LTTESearchResult.null` is the cubic `EphemerisFit`, and from `batch_pdot`'s
  `ladder_*` columns (`ladder_pddot`, `ladder_pdot_cubic` and their errors, inflated by
  `sqrt(chi2_reduced)`, and the O−C rms per degree). `pdot_err_scaled` applies the
  `sqrt(chi2_reduced)` inflation for you.
- **The Eddington–Plakidis wander test** (`cuperiod.eddington_plakidis`): a smooth O−C
  is not evidence of a smooth period change — random cycle-to-cycle fluctuations
  accumulate a random walk that fits a parabola beautifully (Sterken's dice), and most
  Miras plus a third of Cepheids wander non-secularly. The 1929 test separates the two
  through the growth of accumulated delays with cycle lag, here with two corrections:
  the secular quadratic is removed first, and — because polynomial detrending absorbs
  part of the walk itself — the fit uses *exact* expectations (the detrending projector
  propagated through the walk covariance) rather than the classical line, recovering a
  planted per-cycle fluctuation at 93% ± 20% where the naive recipe returns half
  (`tests/test_pdot_stability.py`). Its errors are the estimator's exact covariance,
  `slope_significance` is a calibrated z-score, and an exactly periodic series reports
  0.
- **Sliding-window periodograms** (`cuperiod.sliding_periodogram`): any registered
  method on stepped windows over a shared, ascending frequency grid — a (time ×
  frequency) map with per-window refined peaks, `window_periodogram(i)` slices that are
  ordinary `Periodogram` objects reporting the window's own baseline and point count
  (the run's are in `meta["run_baseline"]` / `meta["run_n_samples"]`;
  `SlidingPeriodogram.n_window_points` counts each window), and a `period_trend()`
  diagnostic — for the period *wander* regime (Miras, spotted binaries) where a single
  quadratic is the wrong model, and as a visual sanity check on any `pdot_search`
  detection (`tests/test_pdot_sliding.py`). `backend="auto"` honours the method's
  settings backend.
- **The Weighted Wavelet Z-transform** (`cuperiod.wwz`, `WWZResult`): Foster's (1996)
  WWZ, the long-period-variable community's standard for period *wander* — the
  date-compensated three-function projection under Gaussian weights whose window is a
  constant number of *periods* wide at every trial frequency (`sigma_t = P` exactly at
  the default `c = 1/(8*pi^2)`; the Templeton et al. 2005 Mira convention `c = 0.001` is
  documented). Returns the Z and semi-amplitude maps with the per-cell effective point
  count, a parabolically refined ridge (`peak_frequency`, `peak_period` and `peak_wwa`
  are array properties), and a `period_trend()` diagnostic that counts independent
  windows; cells below `min_neff` are masked rather than reported, and `to_dict` takes
  `include_map=` as on `SlidingPeriodogram`. The values are centred first, rows with a
  blank or zero error are kept (WWZ does not use the errors; AAVSO visual estimates
  have none), and the automatic frequency axis is geometric, from 2/baseline to the
  pseudo-Nyquist frequency. It takes `backend="auto" | "cpu" | "numba" | "numpy"`,
  with a multicore numba kernel. The
  docstrings carry Foster's own health warning verbatim in spirit: `N_eff` is not an
  integer, the nominal F-distribution does not hold, and no p-value should ever be read
  off a WWZ map. The implementation is pinned cell-by-cell against a transparent
  loop-and-lstsq reference in `tests/test_pdot_wwz.py`.
- **Model-free P(t) tracks** (`cuperiod.period_profile`, `PeriodProfile`,
  `PeriodPoint`): the period measured independently in consecutive intervals, by
  minimizing a fixed-order harmonic chi-squared over frequency (multi-harmonic AOV —
  Schwarzenberg-Czerny 1996), converged on the chi-squared minimum, with the uncertainty
  read off its curvature. This is the data product the period-change literature
  actually publishes when the rate is not constant, and it assumes nothing about the
  shape of the evolution. The harmonic order is chosen once, by the scale-free criterion
  `n*ln(chi2/n) + k*ln(n)` (so optimistic errors cannot inflate it), as the median of the
  orders picked on up to five intervals spread across the baseline, and held fixed so
  intervals stay comparable; each interval's search is confined to a narrow window
  around the reference period because an interval holding a few tens of cycles has a
  rich alias comb — periods pushed to that window's edge are flagged (`at_window_edge`)
  rather than reported as measurements. `PeriodPoint.amplitude` is the fundamental's
  semi-amplitude, and unweighted errors do not depend on the units. Interval period
  errors are calibrated on synthetic data (pull rms 0.98) and are honest about what
  they are not on real data (`tests/test_pdot_profile.py`).
- **Parametric period laws** (`cuperiod.fit_period_law`, `compare_period_laws`,
  `PeriodLawFit`, `available_laws`): `constant`, `linear` (constant dP/dt),
  `exponential`, `singular` — `P = P0 exp(b/(t - t0))`, a finite-time singularity, which
  is Tylenda et al. (2011) Eq. (1) for V1309 Scorpii — and `pejcha`, which lets that
  singularity's exponent float (Pejcha 2014 Eq. 12; `singular` is its `delta = 2`
  case). Fitted by weighted least squares on the periods. The two singular laws are
  fitted in a regular coordinate about the data's mid-epoch `t_c`, `u = 1/(t0 - t_c)`,
  so a `t0` after the data, one before it and the exponential limit (`t0` at infinity)
  are one bounded parameter, with `t0` held outside the observed span by at least one
  median point spacing. `compare_period_laws` and `law="auto"` (`constant`, `linear`,
  `exponential` and `singular`; `pejcha` on request) rank on `PeriodLawFit.bic_scaled`
  — the BIC with every chi-squared divided by the excess scatter of the most flexible
  free law, so errors that understate the track's real scatter cannot buy a period
  change; without `period_err` the laws are compared on the residuals' own scale. A fit
  whose variation is dominated by one end of the track (a singular epoch parked against
  the data, absorbing a single point) or whose `pejcha` exponent reaches its guard range
  is flagged `at_bound` and ranked below every free fit, whatever its BIC. Parameter
  errors are inflated by `sqrt(chi2_reduced)` (`scale_errors=True`), and a parameter
  the points do not constrain, or one held at a bound, reports NaN. `fractional_change`
  and `timescale_yr_at` integrate the law rather than multiplying a rate by a baseline
  (`tests/test_pdot_laws.py`).
- **The cyclic O−C: light-travel-time orbits** (`cuperiod.ltte_search`,
  `LTTESearchResult`, `LTTEFit`, `PhaseTrackResult.ltte_search`): a distant third body
  makes the O−C *periodic* through the light-travel time across the outer orbit (Irwin
  1952), and when that outer period is comparable to or shorter than the baseline the
  O−C is not monotonic — the rate a parabola reports is then a baseline-weighted average
  of a wobble whose value and sign depend on which part of the cycle the data cover.
  Hong et al. (2022) found 307 such systems (plus 104 with a parabola as well) among
  14,127 OGLE bulge contact binaries, at outer periods of 5–14 yr on a 15-yr baseline.
  `ltte_search` fits the cycle *jointly* with the ephemeris polynomial (`degree=`,
  default quadratic): at every trial outer period the model is linear in `A cos ω`,
  `A sin ω` and the polynomial (a floating-polynomial periodogram of the O−C), the best
  trial is polished by nonlinear least squares — with Irwin's eccentric elements freed
  on request (Kepler's equation solved up to e → 1) — and the result carries the outer
  period, amplitude, `a sin i`, mass function and `minimum_third_mass()`, the *secular*
  `dP/dt` with the cycle removed next to the parabola-only one (`pdot_change`), ΔBIC and
  an F-test against the polynomial alone (without `errors=`, from the concentrated
  likelihood), `error_inflation` (`sqrt(chi2_reduced)`, floored at 1), and the
  `n_cycles` / `at_grid_edge` flags that mark the degenerate regime. The default scan
  floor follows the distinct epochs, so repeated or clustered timings do not collapse
  it. It accepts a `phase_track` result, a timing list, or explicit O−C values.
- **What a rate means physically, and how small a one is measurable**
  (`cuperiod.mass_loss_rate`, `mass_transfer_rate`, `pdot_from_mass_loss`,
  `pdot_from_mass_transfer`, `detectable_pdot`, `PdotSensitivity`, `convert_pdot`): the
  Jeans-mode wind relation `(dP/dt)/P = 2 mdot/M` —
  `dP/dt = 2e-6 (P/yr)(mdot/1e-6 Msun/yr)(Msun/M)` — and Kwee's (1958)
  conservative-transfer relation `(dP/dt)/P = 3 mdot (1/M_d − 1/M_a)`, both directions,
  with the sign checks that Schaefer (2024) found failing for 44% of cataclysmic
  variables built in (a negative implied loss or transfer rate is reported as such, not
  silently absolute-valued). `detectable_pdot` is the formal floor of the
  quadratic-ephemeris method, `σ(dP/dt) ≈ 27 P σ_t / (T² √n)` for uniform sampling or
  from the actual epochs (it needs degree + 1 distinct epochs), plus the light-time
  amplitude floor at a given outer period (`PdotSensitivity.sigma_amplitude`, inf for
  outer periods the epochs cannot constrain) — the 0.1-cycle rule of thumb overestimates
  the floor by two orders of magnitude because eclipse timings reach 1e-3 P and there
  are dozens of them, which is why survey detections sit at ~1e-6 d/yr (3e-9 s/s).
  `convert_pdot` converts between the conventions, and every result that carries a rate
  (`PdotBest`, `PhaseTrackResult`, `EphemerisFit`, and `LTTEFit` for its secular rate)
  reports it as `pdot` (d/d), `pdot_s_per_yr`, `pdot_ms_per_yr`, `pdot_d_per_myr`,
  `pdot_d_per_yr`, `pdot_over_p` and `dlnp_dt_per_yr`, in `to_dict()` too.
- **Trend removal** (`cuperiod.detrend`, `cuperiod.running_median`): a running median —
  not a mean, since eclipses are one-sided — with an `O(N log N)` binned variant for
  very long light curves that visits only occupied bins; both honour `min_points`. Survey
  photometry of a period-changing star routinely carries a slow brightness trend that
  the sampling window scatters onto the 1 c/d alias comb; on V1309 Sco an undetrended
  search returns a one-day alias rather than the orbital period. Nothing detrends
  automatically: the width is a physical judgement about a particular star.
- **Batch period-change measurement** (`cuperiod.batch_pdot`,
  `cuperiod.pdot.pdot_to_rows`): the whole chain over many light curves on the same
  spawned worker pool and device rules as `batch_periodograms`, writing three tables —
  one row per light curve (search, O−C, ephemeris ladder, light-time orbit (`ltte=`,
  `ltte_*` columns), winning law with `law_bic_scaled`, `law_delta_bic` on `bic_scaled`
  and `law_at_bound`, and its parameters), one row per O−C segment, and one row per P(t)
  interval — to Parquet/CSV files or resumable directory sinks. Every stage is wrapped
  so one pathological star cannot kill a survey run: a failure still contributes a
  summary row carrying the reason in `status` (and the light curve's `n_samples` and
  `baseline`), and every law's parameter columns are present and NaN-filled in every
  part so a directory sink reads back with a single Arrow schema. Resuming works per
  light curve across the three tables: failed stars are retried and a recomputed star's
  rows are replaced in every table. `method`, `model`, `detrend_method`, `domain` and
  law names are validated up front; `period=` is one seed period (a NumPy scalar or 0-d
  array included) or a catalogue (a mapping or a pandas Series) matched to keys as
  strings (a missing, NaN or non-positive entry means a blind search; a catalogue
  matching no key warns); a bare string for `laws=` / `plot_keys=` counts as one name;
  summary rows carry a `source` column (`tests/test_pdot_batch.py`).
- **Diagnostic figures** (`cuperiod.pdot.plots`, new `cuperiod[plot]` extra,
  `matplotlib>=3.6`): a six-panel per-star figure — light curve, drift-corrected fold
  coloured by epoch (with the track's fitted ephemeris), search map, O−C, P(t) track with
  the fitted law, and the decay timescale — plus the panel helpers to compose your own.
  Built on `Figure` + `FigureCanvasAgg` rather than pyplot, so it is safe to call from
  batch workers, and matplotlib is imported lazily so everything else works without it.
  Flux light curves are drawn the right way up, and the O−C panel draws a light-time
  orbit only when the O−C supports it (ΔBIC > 10, outer period not at the edge of the
  scan). `batch_pdot(plots=...)` raises `ImportError` naming
  `pip install 'cuperiod[plot]'` before any work when matplotlib is missing; figures of
  keys that sanitise to the same file name, or differ only in letter case, get distinct
  names.
- **CLI: `cuperiod pdot` and `cuperiod pdot-batch`** — the search + track chain for one
  light curve, and `batch_pdot` from the command line: window and drift-axis flags
  mapping onto `PdotSettings` (an option not given defers to `CUPERIOD_PDOT_<FIELD>`, and
  the help names the effective default), `--track/--no-track`, `--ltte/--no-ltte`,
  `--profile` (the P(t) track and the laws ranked on it, printing `bic_scaled`, its gap
  to the best law and fits at a bound), JSON output (`--out`) and the 2-D map as `.npz`
  (`--save-map`), with the same column-override and domain options as `run`. The
  light-time block quotes P2, A and the secular dP/dt with their formal errors times
  `LTTEFit.error_inflation`, and says so; an option the search cannot honour (`-m BLS`
  or `TLS`, a `--pdot-range` outside the model's domain, too large a grid) is a usage
  error (exit 2) without a traceback. `pdot-batch` writes the summary, O−C and P(t)
  tables (`--sink`, `--oc-sink`, `--profile-sink`) and per-star figures (`--plots`),
  resumes by key, and reports each star's failure in the summary's `status` column
  (`tests/test_cli_pdot.py`).
- **Input handling of the period-change tools.** Masked entries are missing values; a
  placeholder error column is reported as such by `pdot_search`, `phase_track` and
  `sliding_periodogram`; and the tools raise a `ValueError` naming the argument for a
  NaN or infinite period, a non-finite or non-integer `f0`, harmonic count, `n_tau`,
  `min_neff`, `min_window_points`, `points_per_interval`, `n_intervals` or
  `min_points`, a non-finite warp rate, window, step or WWZ `c`, a `min_neff` below 3,
  a period longer than the sliding window, an unknown law name or mismatched lengths.
  `PdotSettings` validates its fields (`n_segments >= 4`, finite positive bounds), and a
  constant light curve given to `period_profile` raises `InsufficientDataError`.

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
  SuperSmoother, pre-whitening and the GUI is 1e5 Rayleigh widths — about 55 c/d on a
  5-year baseline and 27 c/d on a 10-year survey (ASAS-SN-like triplets: 20.5M → 500k
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
  seeded bootstrap values differ (`cuperiod.prewhiten.bootstrap_uncertainties` and
  `component_uncertainties` now default to `refine="simultaneous"`; it was `"cyclic"`).
  `offset_error` goes through the selected estimator and the `sqrt(D)` inflation (it
  grows by `sqrt(D)` at the defaults). The covariance is the pseudo-inverse of the
  Jacobi-scaled normal matrix about the weighted-mean epoch, and a parameter of a
  zero-amplitude component is NaN rather than 0. Two stops are always on: the
  degrees-of-freedom guard (`2(3K+1) > N`) and residuals at numerical round-off.
- **Batch resume semantics.** File and directory sinks resume **by key**: a re-run
  computes only the (light curve, method) pairs not yet in the sink and retries failures
  — adding, removing or reordering inputs no longer drops or duplicates light curves, and
  a different `chunk_size` is fine. A directory records its request: adding a method is
  allowed, while changed settings, `n_best`, `store_raw`, columns or grid raise
  `ValueError` once it holds part files. `resume=False` first removes the directory's
  `part-*.parquet` files. `batch_prewhiten` resumes the same way. A directory sink
  written by 1.2.0 can be resumed, but its old part files have no `source` column (see
  *Provenance in every batch sink*), and `pandas.read_parquet(<directory>)` takes the
  columns of the first part, so it silently drops `source` from every row: read such a
  directory with `pd.concat` over its `part-*.parquet` files, or re-run it with
  `resume=False`.
- **CPU pools never touch the GPU.** `device="cpu"` maps `"auto"` to CPU backends in
  `batch_periodograms` and `batch_prewhiten` and rejects GPU-only backends (`"gpu"`,
  `"cupy"`, `"cufinufft"`); every CPU-pool worker used to open its own CUDA context on
  CUDA machines. Such batches now run float64 CPU kernels: values can move by ~1e-6 and
  the rows' `backend` column changes. `device="gpu"` keeps a concrete GPU backend
  (torch, cupy, cufinufft) and rejects CPU-only ones.
- **`settings.backend` is the default backend.** `cup.periodogram(..., backend="auto")`
  (or `None`) defers to the method settings' `backend` and `CUPERIOD_<METHOD>_BACKEND`;
  an explicit backend wins. The batch drivers, the LINCC adapters, `multiband_fap`,
  `batch_prewhiten(backend=None)` (the new default) and the CLI follow it. A settings
  pin of `"gpu"` on a machine without one raises `BackendUnavailableError` instead of
  silently running on the CPU.
- **Peak settings are live.** `n_peaks`, `peak_separation_rayleigh`, `harmonic_max`,
  `alias_freq_tolerance` and `downsample_points` now default `best_periods()`,
  `downsample()` and `to_dict()` on results of `cup.periodogram`, the batch drivers, the
  LINCC adapters and the GUI (explicit arguments win). `best_periods(alias_diverse=True)`
  uses an alias/harmonic tolerance separate from the duplicate-peak separation — the
  BLS/TLS `alias_freq_tolerance` (0.0035 c/d), else one Rayleigh width — and treats
  solar-day offsets as aliases: rank 1 is unchanged, lower ranks can differ (a 27-day
  two-planet BLS search now reports the second planet at rank 2).

Other changes:

- **Dependency floors.** `typer>=0.16` (was 0.12; older typer breaks on click 8.2 and
  later). The `[fast]` extra adds `scipy>=1.13`: with numba's thread pool, the OpenBLAS
  bundled with older scipy can crash the process at exit on many-core machines. The
  `[gui]` extra needs `PySide6>=6.7` (was 6.6, which lacks a `QTimer.singleShot`
  overload the GUI uses), below 6.12 on Python 3.11 (see Known limitations), and
  `pyqtgraph>=0.13.7` (was 0.13; 0.13.0 fails under PySide6 6.7); `[dev]` needs the same
  Qt bounds and `pytest-qt>=4.2` (4.0 and 4.1 do not load under pytest 8). The new
  `[plot]` extra is `matplotlib>=3.6`; the pyarrow floor is under Security.
- **Settings selection is strict.** A settings object of the wrong class raises
  `TypeError` (it was replaced by defaults); mapping keys match like method names
  (`"String-Length"`, `"string_length"`), a key naming no method raises
  `UnknownMethodError`, and two keys for one method raise `ValueError`. Settings validate
  their values: finite frequency bounds with `minimum_frequency >= 0` (pre-whitening
  `> 0`), BLS/TLS duration fractions below 1, TLS `u1 + u2 <= 1`, finite non-negative
  thresholds, positive spacing bounds.
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
- **Pre-whitening details.** `fit_mean=False` fits the weighted-mean-subtracted values
  and reports that mean as `offset`; combination identifications have order ≥ 2 and
  `expected_false` counts only positive in-band predictions (it overstated chance matches
  ~6×); the default period-spacing trial range comes from the bulk of the values; an
  explicit `minimum_frequency` above the automatic maximum names `maximum_frequency` in
  its error.
- **Batch details.** `chunk_size` defaults to automatic (about four tasks per worker, at
  most 256); `grid=` is honoured; rows, errors and file sinks are in input order; every
  row carries `{extra}_{k}` for each per-peak extra (NaN where absent), so the parts of a
  directory created by this release share one schema; colliding file stems become
  path-qualified keys (with a warning); a glob matching nothing raises
  `FileNotFoundError` and an existing path is always literal; a batch with no inputs
  leaves its sink untouched; the default worker count follows the CPUs the process may
  use, and pools are capped at 61 on Windows.
- **Batch keys are always distinct.** A key given twice, two light curves with one
  `meta["id"]`, or two keys a CSV sink would store identically (`"=x"` and `"'=x"`)
  keep the first occurrence and rename the repeats `key#2`, `key#3`, … with a warning;
  they used to share result rows and a resume entry, so a resumed sink skipped the second
  light curve as done.
- **Provenance in every batch sink.** `batch_periodograms` and `batch_prewhiten` rows
  carry a `source` column: the file's absolute path, or `""` for an in-memory light
  curve (also `prewhiten_to_rows(source=)`). Resuming into a sink that holds a key from
  a different file warns, and those keys are skipped as done.
- **Pooled-band warning.** Reading a multi-filter table as one light curve now names
  the exact remedy (`columns=ColumnMap(band='filtercode')`, or `band_column=` for the
  batch functions) and is attributed to the caller's line, not a library one.
- **CLI.** User errors (missing file, unknown method / backend / device, bad sink suffix,
  non-positive `--period`, empty `-m`, `--n-best` < 1, a GPU-only backend with
  `--device cpu`) are usage errors (exit 2) without a traceback; `batch` and
  `batch-prewhiten` exit 1 when every light curve that ran failed, list every failure in
  `<stem>.errors.csv` beside a file sink (`_errors.csv` inside a directory sink) and
  print the worker count. The `--resume/--no-resume` help of `batch-prewhiten` says what
  resume does now: skip the light curves already in the sink and retry failures,
  matched by key.
- **LINCC interop.** `nested=None` is the new default (the preset's nest, else `"lc"`);
  an explicit `nested=` wins over a preset; a `value=` naming another column than the
  preset's drops the preset's error column and domain; `append_columns=True` onto
  existing result columns raises, pointing at `prefix=`; the `period_i` columns follow
  the settings' peak fields, as in `cup.periodogram`.
- **GUI.** The Options form is grouped into collapsible sections (search range first,
  Advanced collapsed, Multiband only when a joint fit runs) with units, a Reset to
  defaults button, stop-criterion check boxes and editable SuperSmoother spans; it shows
  the values that run (samples per peak 10, explicit values honoured,
  `CUPERIOD_<METHOD>_<FIELD>` overrides applied, automatic values to six significant
  digits) and no longer offers `n_peaks`, `store_spectra` or `downsample_points`. Results
  are cached by a fingerprint of the data, not the file name, in caches bounded at
  512 MiB. A loaded curve is drawn at once; errors are explained in plain language with
  the full text on demand; the Backend and Band choices persist; the spectrum remembers
  its x-axis and log choice per kind of result; clicking a marker selects the whole peak
  (BLS/TLS t0 included); the Peaks table shows frequency and names each method's
  statistic; both tables have Copy / Export CSV buttons and a column chooser; the
  Frequencies CSV has one fixed header and a pre-whitening spectrum CSV names its column
  `amplitude`; plot axes no longer rescale by SI prefixes; the window sizes itself to
  the screen and keeps its layout, analysis and options; plot toolbars wrap, and a pane
  wider than its dock scrolls instead of squeezing the plots. With *Show Raw LC* on, the
  raw curve sits beside the fold when there is room for both and otherwise shares the
  fold's pane as *Phased light curve* / *Raw light curve* tabs (Left/Right switch; a
  point tool or Ctrl+3 brings the raw curve forward); its tools fit one compact row, and
  the spectrum and the light curve get plots of equal height until the split is
  dragged.
- **Release process.** The release workflow refuses to publish when the pushed `v*` tag
  differs from the built version, and a manually started run publishes to TestPyPI only.
- **Development.** GPU tests carry the registered `gpu` marker: `pytest -m "not gpu"`
  deselects them and hides the GPUs from the rest of the run (default-backend tests run
  on the CPU); with the GPU stack but no device, GPU tests skip.
- **Docs.** New guide pages on interactive pre-whitening (with the GUI's shortcut tables
  generated from its command registry at build time) and on period change.

### Deprecated

- `PreWhitenSettings.downsample_points` never had an effect and is no longer a settings
  field (it is absent from `model_fields` and `model_dump()`). Reading it warns
  (`DeprecationWarning`) and returns 2000, its 1.2.0 default; passing it, or setting
  `CUPERIOD_PREWHITEN_DOWNSAMPLE_POINTS`, warns and is ignored. It will be removed.
- `BatchSettings` is read by no batch driver, so it and `CUPERIOD_BATCH_*` never had an
  effect; building it now warns (`DeprecationWarning`), so does a batch run while a
  `CUPERIOD_BATCH_*` variable is set, and it will be removed. Pass `device=`,
  `workers=`, `n_best=`, `store_raw=` and `resume=` to the batch functions.

### Fixed

- **`backend="auto"` failed when the CUDA fast path was half-installed.** On a machine
  with cupy and a GPU but no cufinufft, `"auto"` resolved GLS to `cufinufft` and raised
  `ModuleNotFoundError` deep in the computation. Resolution now checks the fast path's
  package as well as the device and falls back to torch on a GPU, then the CPU;
  `backend="gpu"` raises `BackendUnavailableError` naming the missing package.
- **Quoted globs on Windows.** `cuperiod batch "lcs/*.csv"` (and `batch-prewhiten`)
  stopped with an "unexpected extra argument" usage error: click expanded the quoted
  pattern into one argument per match. The commands now receive the pattern and resolve
  it themselves, as on POSIX shells.
- **`cuperiod prewhiten` and `batch-prewhiten` ignored `CUPERIOD_PREWHITEN_*`.** Their
  `--max-frequencies`, `--snr`, `--stop`, `--uncertainty`, `--combinations` and
  `--backend` options always passed the CLI's own defaults, which hid the environment.
  An option now overrides only when it is given, so the environment applies as in the
  library, and each help line names the effective default (for example *default:
  CUPERIOD_PREWHITEN_SNR_THRESHOLD, else 4.0*).
- **Files with a UTF-8 byte-order mark** (Excel's *CSV UTF-8*) load:
  `LightCurve.from_file`, `MultiBandLightCurve.from_file`, the batch functions and the
  GUI strip the mark from the first column name, where it made that column
  unrecognisable (usually the time column); compressed files too.
- **Inputs.** Masked entries are NaN on every input route — numpy masked arrays,
  astropy `MaskedColumn` / `Masked`, pandas NA, pyarrow nulls — so tuples and
  `LightCurve.from_arrays` no longer analyse the fill values under a mask; masked integer,
  boolean and short text columns load (they raised `Cannot convert fill_value nan` or
  `could not convert string to float: 'n'`). Compressed FITS (`.fits.gz`, `.bz2`, `.xz`)
  and `.fts` files are read as FITS; integer-labelled frames (`pd.read_csv(header=None)`)
  load with `ColumnMap(time=0, value=1, error=2)`; Parquet categorical columns with nulls
  no longer take a neighbouring category; conversion errors name the column. A
  placeholder error column (all 0, −1 or NaN) is reported as such by every method, the
  multi-band path and pre-whitening, instead of "no usable time baseline".
- **Directory inputs skipped readable files.** A directory given to the batch functions,
  `cuperiod batch` / `batch-prewhiten` or the GUI (Open folder, a dropped folder or
  files) takes every format the reader reads: `.fz` and `.tab` files and compressed
  (`.gz`, `.bz2`, `.xz`) light curves were skipped, and `.fts` files are taken too.
- **Backends.** Selectors ignore case and surrounding spaces and the unknown-backend
  error lists `auto` / `cpu` / `gpu` / `torch[:<device>]`; a named backend that is not
  installed says which extra installs it (`pip install 'cuperiod[fast]'` for numba,
  `'cuperiod[gpu]'` for cupy and cufinufft, which also need a CUDA device);
  `is_gpu_backend("torch")` is true only when torch has a GPU, so a CPU-only torch run is
  no longer labelled GPU; a one-sample trial grid of the opposite kind no longer crashes
  the torch backends.
- **`cuperiod methods` and `cuperiod doctor`.** `methods` lines up its columns for the
  long names (STRINGLENGTH, SUPERSMOOTHER). `doctor` labelled the best torch device as
  used by `backend='auto'` even where `auto` resolves to numba or finufft; it now says
  `backend='torch'`, and adds `'auto'` only when `auto` really resolves to torch.
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
  records `torch:<device>`. The CUDA kernel broke ties between equally good boxes by
  thread id, so on sparse curves it reported another (equally good) mid-transit time and
  duration than numpy/astropy at ~70% of periods; ties now go to the earliest (duration,
  start), as astropy does (power never differed).
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
  on the grid edge gets a finite FAP; `n_bootstrap < 1` raises.
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
- **`cuperiod run --out`** saved different BLS peaks from the ones it printed; the
  printout and the JSON now list the same peaks, alias-diverse for BLS and TLS.
- **`suggest_gpu_workers` suggested 16 workers on almost any card.** It now counts each
  worker's measured CUDA context (~200–235 MiB) and assumes a 100k-point grid, caps at 4
  workers without NVIDIA MPS (16 with) and at the host CPUs, and rejects a bad
  `headroom`. A GPU batch with `workers=None` is sized for the most demanding of its
  methods and for `grid=` when given, not for the first method listed.
- **LINCC interop.** `nested_periodogram` on a zero-row frame returns typed float64
  result columns.
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
  right-click menus no longer block the window. The window title names the application
  once, at the end (`star.csv — cuPeriod`; on Windows it read
  `cuPeriod — star.csv - cuPeriod`).
- **GUI: plots and tables.** The spectrum is drawn end to end on a period or log axis
  (index-based decimation dropped the long-period end and its peaks) and re-binned on
  zoom; the phased and raw views draw around NaN values and accept empty curves; stop
  reasons containing `<` show verbatim; a BLS/TLS marker click keeps the peak's t0 and
  the tables follow selections made in the spectrum; PNG exports append `.png` (asking,
  without blocking the window, before replacing an existing image) and report failures,
  as CSV exports do; the Peaks table keeps locale-independent formats (a t0 near
  JD 2459001.9 read `2.459e+06`) and exports full precision; the Frequencies ID column
  sorts F1…F12 in order; Ctrl+C in a table acts only while it has the focus and Tab
  leaves it; échelle points follow a theme switch; strokes keep their width on HiDPI
  screens; the crosshair readout no longer resizes the panes; log-axis tick labels no
  longer overprint, the top peak's label is not clipped, and legends have a background.
- **GUI: keyboard and layout.** Tab no longer gets stuck in the Options after Run /
  Compute, the Options scroll to the field it lands on, and tab-bar scroll arrows stay
  out of the Tab order. The Period-spacing pane wraps its controls instead of hiding
  them in a narrow dock; the window's minimum size follows what its panes need; the
  spectrum, fold, Peaks and Frequencies *Export…* buttons use the window's file dialog.
  Smaller: the spectrum readout keeps its start and reads c/d, no empty period badge,
  and "no components yet".

### Security

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
- **Benchmark downloads.** `download_s82_rrlyrae.py` reads astroML-data at a pinned
  commit with a 120 s timeout, and it and the new `download_ogle_triples.py` and
  `download_v1309sco.py` (the OGLE archive, over HTTPS) verify a pinned SHA-256 for every
  file, on download and for cached copies. The shared fetcher refuses any URL but HTTPS;
  the two downloaders built on third-party clients (lightkurve from MAST over HTTPS, the
  ASAS-SN Sky Patrol API over plain HTTP) say in their docstrings that no reference
  digest exists for what they fetch.

### Performance

- **Pre-whitening** runs about 1.7× faster end to end (amplitude-only in-loop spectra
  with cached normal-equation terms, guarded Cholesky solves with factor reuse, reused
  designs across cyclic sweeps, window-only engines, no CUDA probing for explicit CPU
  backends, a vectorised combination search). CPU spectra of small transforms (points +
  frequencies ≤ 150 000) run on one thread, up to ~7× faster per spectrum.
- **GLS**: small finufft transforms run on one thread (4–11× faster single-curve GLS on
  many-core CPUs); CPU GLS reuses one finufft plan per batch worker and per LINCC
  partition (`FinufftGLS`); the multi-band offsets model computes its per-band sums in
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
- **Core**: backend discovery is memoised per process (~1 ms → ~1 µs; small PDM/CE
  periodograms up to 3× faster), explicit CPU requests no longer import cupy (~0.25 s per
  process or batch worker), default settings are no longer rebuilt from the environment
  on every call, and peak-preserving downsampling is vectorised (12–23×; batch rows with
  `store_raw=True` ~12×).
- **Batch**: CPU pools spread small batches over all workers, pin BLAS/OpenMP threads
  only while the pool exists and give numba `cores // workers` threads per worker. The
  nested-pandas partition tier no longer imports cupy for a CPU GLS partition.
- **GUI**: light curves above 20 000 points are drawn as a fixed random subset and the
  hidden raw panel builds only when shown (refolds of 1e5–5e5-point curves: 0.2–2 s →
  ~30 ms); the Load demo menu fills when first opened (~0.8 s off start-up).
- **GPU paths measured and tightened** (RTX 5070 Ti, warm, min of N):
  - One-off GPU GLS periodograms (API, GUI, multi-band) reuse a process-wide,
    thread-safe, size-capped cufinufft engine instead of planning per call: 2.2 → 1.2 ms
    at 300 points, 9.5 → 4.6 ms at 20k; `free_gpu_memory()` releases it.
  - Pre-whitening on cufinufft solves the normal equations and forms the window terms,
    spectral window and full spectrum on the device: a scan at 1.4M frequencies
    23.6 → 6.7 ms; `cup.prewhiten` 310 → 128 ms (20k points), 2.15 → 1.25 s (100k).
  - GPU BLS uploads all period segments of a curve once and downloads once (5–15%).

### Validated

- **End-to-end injection-recovery of the dP/dt chain**
  (`benchmarks/pdot_recovery.py`): 60 two-harmonic stars over 1500 days with drifts
  spanning 0.5–8 accumulated cycles, plus 60 drift-free nulls. The search localized
  the planted (P, dP/dt) in **60/60** stars, the tracker's F-test detected **60/60**
  at p < 1e-3 with **0/60** false detections on the nulls, and the pull distributions
  are calibrated (null rms 1.07, injected 0.8–1.3). Median cost per star on one CPU
  core: 0.05 s for the 201×601 search, 0.008 s for the tracking measurement.
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
  `period_profile` at 50 points per interval, then the laws fitted to that track): the
  period is not constant (`compare_period_laws` puts a constant period last, 38 behind
  on `bic_scaled`), but the track alone cannot tell the published `singular` law from a
  constant or exponential rate of change — on `bic_scaled`, the ranking
  `compare_period_laws` and `law="auto"` use, `linear`, `exponential` and `singular`
  lie within 0.4 of each other, `linear` first. (On the plain BIC, which the
  understated errors inflate, `singular` leads `linear` by 188.) Fitting the published
  `singular` law to the track, the measured P(t) points scatter about the published
  Eq. (1) by **0.19% of P** with a +0.064% mean offset, and the two fitted P(t) curves
  differ by at most **0.33% of P** anywhere in the observed span. The total change is
  **−1.47%** from the law and −1.11% from the endpoint intervals against the published
  −1.2% (and −1.14% from integrating Eq. (1) over this exact window); the decay
  timescale runs **875 → 185 yr** against Eq. (1)'s 1730 → 156 and the quoted
  ~1000 → ~170. Repeating at 35/40/50/60 points per interval spans −1.07% to −1.47%,
  which is the method's own systematic and is reported rather than hidden. Forcing a
  single constant rate instead returns −1.91%, and the tools say why: the F-test is
  decisive (p = 3e-8) while the O−C residuals fall 194 → 89 → 16 minutes from a
  quadratic to a quartic ephemeris.

  The `singular` law's `b` and `t0` are individually degenerate here (they trade off
  strongly, and the fit reports errors that say so) while the P(t) they predict over the
  observed span is not — which is why the comparison above is made on the predictions.
- **Light-time orbits against an independent solver** (`tests/test_pdot_ltte.py`):
  planted circular and eccentric orbits are recovered by `ltte_search` within their
  errors against an independent Kepler solver, and a planted 800-day cycle that the
  parabola alone turned into a 2σ "drift" comes out null in the joint fit.
- **Rates and detection floors** (`tests/test_pdot_physics.py`): `detectable_pdot`'s
  floor is pinned against the closed form and against the Monte Carlo scatter of the
  quadratic ephemeris fit, and the mass-loss and mass-transfer relations against the
  published scalings, their signs and limits.
- **Light-time orbits on real OGLE triples** (`benchmarks/pdot_ogle_triples.py`,
  `download_ogle_triples.py`; write-up with figures in
  `benchmarks/REPORT_ogle_triples.md`): six of Hong et al. (2022)'s sinusoidal contact
  binaries, 14.3 yr of OGLE-III + IV photometry each, run through detrend →
  `phase_track` (40 segments) → `ltte_search` with no per-season binary modelling.
  **6/6** recovered outer periods agree with the published ones within 2σ (median
  offset 4%; e.g. 6.43 ± 0.09 yr vs 6.2 ± 0.2 for OGLE-BLG-ECL-289126, amplitude
  660 ± 17 s, ΔBIC +2081), while the parabola alone claimed a ≥ 3σ dP/dt on **3/6**
  and the joint fit's secular rate on 1/6. These stars are where the half-period
  symmetry of contact-binary templates showed up: without `symmetry_order`, the O−C of
  four of the six jumped by P/2 for up to three quarters of the segments.
- **Injection-recovery of the cyclic O−C** (`benchmarks/pdot_ltte_recovery.py`): 80
  planted 120–600 s orbits at outer periods of 0.25–1.5 baselines plus 40 nulls. With
  the cycle covered (≤ 1 baseline), the outer period is recovered within 10% in 81%
  and the amplitude within 3σ in 98%; the parabola alone reports a ≥ 3σ "dP/dt" on
  41% of stars that have no secular rate at all (64% when the cycle is 0.75–1
  baselines long) against 20% for the joint fit, and there are no false cycles and
  no false rates on the nulls. Beyond one baseline the cycle is detected in 12%: the
  degenerate regime, stated as such.
- **The whole benchmark suite re-measured for this release** (`benchmarks/REPORT.md`;
  RTX 5070 Ti and a 16-core/32-thread CPU, best of N on a quiet machine). Single-curve
  CPU: GLS 12.6 → 5.8 ms, BLS 171 → 39 ms (89× astropy's `BoxLeastSquares`, was 20×;
  bit-identical power and the same best period on 126/126 stars), String-Length 43 →
  8.9 ms, SuperSmoother 99 → 64 ms, TLS 132 → 126 ms on its 2.5× denser default grid
  (2.6× less per trial period) — while `torch:cuda` TLS takes about 30% longer per trial
  period, the price of fitting the out-of-transit level jointly with the depth. GPU GLS
  5.4 → 1.7 ms; batch GLS peaks at 728 light curves/s on one GPU (was 574; >2.6
  million/hour), with CPU pools now running CPU kernels only. CPU↔GPU parity tightened
  for MHAOV (1.1e-5 → 1.2e-10), PDM (2.0e-10 → 2.6e-15) and BLS (1.2e-11 → 2.3e-12), TLS
  moved to 9.0e-9 and GLS to 3.1e-6 (relative, at near-zero powers; its CPU and GPU
  NUFFTs agree to their 1e-9 tolerance in absolute power), and every backend picks the
  identical best period on every star. Recovery is unchanged except TLS exact, 85.7% →
  82.1% (one W UMa binary now found at P/2; harmonic-aware unchanged). On the real
  Stripe 82 RR Lyrae MHAOV rises from 83% to 92% strict (84% → 94% harmonic-aware) and
  BLS falls from 22% to 20%; GPU top periods agree on 99.7% of runs with no strict
  recovery changing. Simulated multi-band recovery moved only for GLS `perband` at 30
  epochs (17.7% → 19.7%) and injection–recovery in 2 of 77 cells (TLS transits at SNR ≤
  1); Kepler TLS is unchanged (10/12).

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
- Pre-whitening checks `min_separation_rayleigh` when a component is extracted, not
  after the final polish, which moves each frequency by up to `refine_bound_rayleigh`.
  With a tight guard two components can converge on one frequency with large,
  cancelling amplitudes (flagged `blended`). Keep the guard above the refine box, as the
  defaults (1.5 and 1.0) are; a guard of 1 Rayleigh width or less can collapse even
  with a box no wider than it.
- The fold methods' default grids are not exactly invariant to the time origin: the last
  sample sits on a phase-bin edge, so JD and MJD inputs differ at the O(1/N) level.
- **PySide6 6.12 on Python 3.11.** PySide6 6.12.0 drops a reference to `None` whenever
  a Qt call returns null; on Python 3.11, where `None` is not immortal, the GUI aborts
  ("deallocating None") within seconds. The `[gui]` extra keeps Python 3.11 below 6.12,
  and `cuperiod-gui` refuses to start on an environment that already has it, naming
  `pip install 'PySide6<6.12'`. Python 3.12 and later are not affected.
- numba's on-disk cache can fail (`FileNotFoundError` on a `*.nbi.tmp` or `*.nbc.tmp`
  file) when cuPeriod is installed under a very long Windows path and long paths are
  disabled: set `NUMBA_CACHE_DIR` to a short folder, or enable Windows long paths.

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
