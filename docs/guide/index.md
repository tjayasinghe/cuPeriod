# User Guide

The guide builds on the {doc}`../quickstart` and covers cuPeriod in depth — what it
accepts, how to choose a method, how to read and tune results, and how to scale up. Each
page stands on its own; read top to bottom for a complete tour.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {doc}`light-curves`
Every accepted input — arrays, dicts, DataFrames, astropy/pyarrow tables, and files —
plus column auto-detection and the magnitude/flux domain.
:::

:::{grid-item-card} {doc}`methods`
A decision guide to the seven methods: what each is for, its objective sense, and its
key knobs.
:::

:::{grid-item-card} {doc}`results`
`Periodogram`, `Peak`, and `MultiResult`: peak finding, alias-aware selection,
per-method extras, and serialization.
:::

:::{grid-item-card} {doc}`backends`
`auto` / `cpu` / `gpu` and the concrete backends — what each method runs on and what's
fast where.
:::

:::{grid-item-card} {doc}`tuning`
Per-method settings models, environment-variable overrides, and custom frequency/period
grids.
:::

:::{grid-item-card} {doc}`multiband`
Jointly model several filters of the same star with GLS, BLS, and MHAOV.
:::

:::{grid-item-card} {doc}`batch`
Scale to millions of light curves over CPU pools or the GPU, with resumable Parquet
output.
:::

:::{grid-item-card} {doc}`cli`
The `cuperiod` command line: `run`, `batch`, `methods`, `gpu-info`, `grid-info`.
:::

::::

## Mental model

A run is always the same three steps, whether you call the function or the CLI:

1. **Input** → a {class}`~cuperiod.LightCurve` (or {class}`~cuperiod.MultiBandLightCurve`).
   cuPeriod coerces tuples, tables, and files for you ({doc}`light-curves`).
2. **Method + grid** → the trial frequencies/periods to evaluate. Each method builds a
   sensible default grid; you can override it ({doc}`tuning`).
3. **Backend** → where the math runs (`auto` picks the GPU when present). The result is
   a {class}`~cuperiod.Periodogram` you query for the best periods ({doc}`results`).

The single entry point {func}`cuperiod.periodogram` ties these together; its batch
sibling {func}`cuperiod.batch_periodograms` does the same for many light curves at once.
