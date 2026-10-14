# API reference

The complete public API, generated from the source docstrings. Everything below is
importable directly from the top-level `cuperiod` package (the recommended alias is
`cup`):

```python
import cuperiod as cup
```

```{eval-rst}
.. currentmodule:: cuperiod
```

## Entry points

The two functions most users need, plus their helpers.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   periodogram
   batch_periodograms
   best_periods
   to_input
```

## Pre-whitening

Automated frequency extraction for pulsators, and the g-mode period-spacing tools that
follow it (see {doc}`../guide/prewhitening`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   prewhiten
   batch_prewhiten
   PreWhitenResult
   Sinusoid
   Combination
   identify_combinations
   fit_multisine
   MultiSineFit
   amplitude_spectrum
   spectral_window
   baluev_fap
   AmplitudeSpectrum
   SpectrumEngine
   find_period_spacing
   PeriodSpacingSeries
   spacing_spectrum
   SpacingSpectrum
   echelle
   buoyancy_radius
```

## Interactive pre-whitening

The Period04-style session — stepping, manual edits, data edits, undo, projects and
scripts — and the constraint model behind it (see {doc}`../guide/interactive-prewhitening`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   PreWhitenSession
   StepReport
   VariationResult
   SessionSnapshot
   Partition
   prewhiten.suggest_partition
   prewhiten.SubsetSolution
   FitConstraints
   ComponentConstraint
   FrequencyTie
   FitStart
   prewhiten.FitDetail
```

A session's behaviour (undo depth, caches, project limits) is set by
{class}`SessionSettings`, listed under [Settings](#settings). The session's errors and
warning: {class}`SessionError` (an operation the session cannot
take outside {meth}`PreWhitenSession.do`, which reports it as ``refused`` instead),
{class}`ProjectError` (anything wrong with a project file or its replay),
{class}`ConstraintError` (an impossible constraint) and {class}`SessionWarning` — listed
under [Exceptions](#exceptions).

## Period change

Joint (period, dP/dt) searching, automated O−C phase tracking, ephemeris fitting,
model-free P(t) tracks and the laws fitted to them, time-resolved maps, the
random-wander test, the light-travel-time (cyclic O−C) search, the physical
interpretation and sensitivity helpers, and the catalogue-scale driver
(see {doc}`../guide/period-change`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pdot_search
   PdotSearchResult
   PdotBest
   phase_track
   PhaseTrackResult
   PhaseTemplate
   SegmentShift
   fit_ephemeris
   EphemerisFit
   period_profile
   PeriodProfile
   PeriodPoint
   fit_period_law
   compare_period_laws
   PeriodLawFit
   available_laws
   available_models
   detrend
   running_median
   sliding_periodogram
   SlidingPeriodogram
   wwz
   WWZResult
   eddington_plakidis
   EddingtonPlakidisResult
   ltte_search
   LTTESearchResult
   LTTEFit
   detectable_pdot
   PdotSensitivity
   mass_loss_rate
   mass_transfer_rate
   pdot_from_mass_loss
   pdot_from_mass_transfer
   convert_pdot
   batch_pdot
   weighted_epoch
```

Diagnostic figures live in {mod}`cuperiod.pdot.plots`, which needs matplotlib
(`pip install "cuperiod[plot]"`) and so is imported on demand:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pdot.plots.pdot_figure
   pdot.plots.plot_light_curve
   pdot.plots.plot_phase_fold
   pdot.plots.plot_search_map
   pdot.plots.plot_oc
   pdot.plots.plot_period_profile
   pdot.plots.plot_timescale
```

## Multi-band

Joint multi-band false-alarm calibration (see {doc}`../guide/multiband`). The multi-band
models themselves are selected through {class}`GLSSettings` and run through
{func}`periodogram`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   multiband_fap
   MultibandFAP
```

## Diagnostics

Is the best peak the true frequency, or a sidelobe of the sampling's spectral window?

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   alias_diagnostics
   AliasReport
   AliasCandidate
   WindowPeak
```

## Light curves & inputs

Containers for the data, and the column/domain mapping that ingests heterogeneous tables.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   LightCurve
   MultiBandLightCurve
   ColumnMap
   Domain
   core.coerce_domain
   core.raise_if_unusable
   core.lightcurve.stack_bands
```

## Results

What a run returns.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Periodogram
   Peak
   MultiResult
```

## Settings

Per-method tuning models (see {doc}`../guide/tuning`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GLSSettings
   BLSSettings
   PDMSettings
   CESettings
   MHAOVSettings
   StringLengthSettings
   SuperSmootherSettings
   TLSSettings
   PreWhitenSettings
   PdotSettings
   SpacingSettings
   SessionSettings
   BatchSettings
```

:::{note}
`BatchSettings` is **deprecated**: no batch driver reads it, so building one raises a
`DeprecationWarning`, and the batch functions warn when a `CUPERIOD_BATCH_*` environment
variable is set (it has no effect). Pass `device`, `workers`, `n_best`, `store_raw` and
`resume` to {func}`batch_periodograms` directly.
:::

## Grids

Trial frequency/period grids, their builders, and the size guard every default grid
goes through (see {doc}`../guide/tuning`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GridSpec
   uniform_frequency_grid
   log_period_grid
   default_frequency_grid
   pseudo_nyquist_frequency
   regular_cadence_nyquist
   uniform_grid_size
   check_grid_size
```

`GridTooLargeError` and `GridSizeWarning` are listed under [Exceptions](#exceptions).

## Method registry

Introspect and look up the registered methods.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   list_methods
   method_names
   get_method
   MethodInfo
```

## Devices & GPU

GPU discovery, memory management, and batch worker sizing (see {doc}`../guide/backends`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   gpu_info
   GpuInfo
   suggest_gpu_workers
   free_gpu_memory
```

## Backend resolution

How a backend request becomes a concrete backend, for code that drives methods itself
(see {doc}`../guide/backends`). These live in submodules, not at the top level.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   core.backend.refresh_backends
   core.backend.cpu_backends
   core.backend.cpu_pool_backend
   core.backend.requested_backend
   core.backend.normalize_backend
   methods.base.select_settings
   methods.base.PeriodogramMethod
   methods.gls.FinufftGLS
```

## Batch

The batch run summary (see {doc}`../guide/batch`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   BatchSummary
```

Worker sizing for CPU pools lives in `cuperiod.batch.sizing`:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   batch.sizing.usable_cpu_count
   batch.sizing.cpu_worker_count
   batch.sizing.clamp_pool_workers
   batch.sizing.numba_threads_per_worker
   batch.sizing.pinned_worker_threads
   batch.sizing.init_worker_threads
```

## LINCC interoperability

Adapters for nested-pandas / lsdb catalogs (see {doc}`../guide/interop`). These live in
the optional `cuperiod.interop` subpackage — `pip install "cuperiod[nested]"` — and are
imported from there, not from the top level.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   interop.nested_periodogram
   interop.partition_periodogram
   interop.resolve_nested_columns
   interop.NestedColumns
   interop.COLUMN_PRESETS
```

The survey column layouts {data}`~cuperiod.interop.COLUMN_PRESETS` accepts as
`preset=` are tabulated in {doc}`../guide/interop`.

## Exceptions

All cuPeriod errors derive from {class}`CuPeriodError`, so a single `except` catches the
family; each also derives from the closest built-in (`ValueError`/`RuntimeError`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   CuPeriodError
   BackendUnavailableError
   ColumnResolutionError
   InsufficientDataError
   UnknownMethodError
   GridTooLargeError
   ConstraintError
   SessionError
   ProjectError
```

Warnings (subclasses of {class}`UserWarning`):

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GridSizeWarning
   SessionWarning
```
