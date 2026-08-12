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
   AmplitudeSpectrum
   SpectrumEngine
   find_period_spacing
   PeriodSpacingSeries
   spacing_spectrum
   SpacingSpectrum
   echelle
   buoyancy_radius
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
   TLSSettings
   PreWhitenSettings
   SpacingSettings
   BatchSettings
```

## Grids

Trial frequency/period grids and their builders (see {doc}`../guide/tuning`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GridSpec
   uniform_frequency_grid
   log_period_grid
```

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

## Batch

The batch run summary (see {doc}`../guide/batch`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   BatchSummary
```

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
```
