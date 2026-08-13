# LINCC catalogs (nested-pandas / lsdb)

The LINCC Frameworks stack keeps a star's light curve **inside its object row**: one row
per object, the per-epoch arrays in a nested column (a nested-pandas `NestedFrame`), and
`lsdb` spreads that frame over the dask partitions of a HATS catalog. The usual way to run
a period search over such a table — explode to long format, `groupby` the object id,
rebuild a DataFrame per star — undoes exactly the layout the format exists for.

`cuperiod.interop` runs cuPeriod directly on that layout: no flattening, no `groupby`, no
per-object DataFrames.

```bash
pip install "cuperiod[nested]"    # nested-pandas + a recent pyarrow
pip install "cuperiod[lsdb]"      # + lsdb, for HATS catalogs
```

Importing `cuperiod` never pulls in nested-pandas, and importing `cuperiod.interop` does
not either — the dependency is checked on first *use*, so even `COLUMN_PRESETS` can be
inspected in a bare install.

## Two tiers

| | Function | Unit of work | Reach for it when |
| --- | --- | --- | --- |
| **Tier 1** | {func}`~cuperiod.interop.nested_periodogram` | one row | CPU backend, a quick look, a small catalog |
| **Tier 2** | {func}`~cuperiod.interop.partition_periodogram` | one partition | GPU throughput over a survey |

They take the same column and method arguments and produce the same result columns; what
differs is how the work is executed (and that Tier 2 returns the result columns only).
Both accept an in-memory `NestedFrame` or a lazy lsdb `Catalog` (which stays lazy), resolve
their columns from the nest's **schema alone** — never by computing — and turn a failed
object into NaN result columns rather than an aborted run.

## Tier 1 — row-wise

`nested_periodogram` wraps `map_rows`: one light curve per row, one periodogram per row,
results appended as ordinary base columns.

```python
from cuperiod.interop import nested_periodogram

out = nested_periodogram(frame, "lc", preset="ztf_dr22")
out[["best_period", "best_power", "fap"]].head()
```

The result columns are `best_period`, `best_power`, `fap` (NaN when the method reports
none), plus `period_2 … period_n` when you ask for `n_best > 1`. `prefix=` prepends a
string to every one of them, so several methods can live in one frame;
`append_columns=False` returns only the results.

On a lazy lsdb `Catalog` nothing changes except that you call `.compute()` at the end:

```python
import lsdb
from cuperiod.interop import nested_periodogram

cat = lsdb.open_catalog("dp1_object")
res = nested_periodogram(
    cat, preset="rubin_dp1_object", method="GLS", n_best=3, prefix="gls_"
)
res[["gls_best_period", "gls_period_2"]].compute()
```

`meta` for the dask graph is built automatically (`{column: float}`) for catalogs; pass
your own with `meta=` if you need to.

Multi-band comes from one argument. When `band=` resolves to an in-nest sub-column, each
row is built as a {class}`~cuperiod.MultiBandLightCurve` grouped on that column and the
method's multi-band model runs ({doc}`multiband`); otherwise every epoch is one band. The
band column is **never** auto-detected, so a nest that happens to carry a `band` column is
not silently reinterpreted as a joint fit.

## Tier 2 — partition-wise

`partition_periodogram` reads a whole partition's nested column **once** through its Arrow
buffers (list offsets plus struct fields), slices each object out of the flat arrays, and
evaluates every object in the partition against **one** method engine built per partition.

```python
from cuperiod.interop import partition_periodogram

res = partition_periodogram(frame, "lc", method="GLS")
res["best_period"].head()
```

That is the GPU differentiator: plan and kernel setup is amortized over a whole partition
of stars instead of paid per star. On a CPU backend (no engine) it is still the cheaper
path, because the per-object DataFrame round-trip is skipped. The result is one row per
object, indexed like the input, carrying the result columns only.

:::{note}
**Deploying on GPUs with dask.** Run **one worker per GPU** and let each worker own its
device:

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

client = Client(LocalCUDACluster())
res = partition_periodogram(cat, preset="ztf_alerts", method="GLS", backend="gpu")
res.compute()
```

The callable dask ships to the workers holds only picklable configuration — column names,
method name, settings, grid. Compute engines are **never** pickled: each is built lazily
inside the worker, on that worker's own device, and released when the partition finishes.
:::

## Column presets

Survey layouts are stable enough to name. `preset=` fills in every column parameter you
did not set yourself; anything you pass explicitly wins.

| Preset | Nest | Time | Value / error | Band | Domain |
| --- | --- | --- | --- | --- | --- |
| `"ztf_dr22"` | `lc` | `hmjd` | `mag` / `magerr` | — (see below) | magnitude |
| `"ztf_alerts"` | `lc` | `lc_mjd` | `lc_magpsf` / `lc_sigmapsf` | `lc_fid` | magnitude |
| `"rubin_dp1_object"` | `objectForcedSource` | `midpointMjdTai` | `psfFlux` / `psfFluxErr` | `band` | flux |
| `"rubin_dp1_dia"` | `diaObjectForcedSource` | `midpointMjdTai` | `psfFlux` / `psfFluxErr` | `band` | flux |

ZTF DR22 keeps the filter **outside** the nest: it has one row per (object, filter), with
the filter in the base column `filterid` (1 = *g*, 2 = *r*, 3 = *i*). The preset therefore
sets `band=None` and each row is a single-band search. A joint multi-band run needs a
self-join that nests one object's per-filter rows into one row first
(`NestedFrame.join_nested`).

Without a preset, the time/value/error sub-columns are auto-detected from the nest's
schema by the same survey-aware {class}`~cuperiod.ColumnMap` machinery as everywhere else
({doc}`light-curves`); {func}`~cuperiod.interop.resolve_nested_columns` runs that
resolution on its own if you want to see what it decided:

```python
from cuperiod.interop import resolve_nested_columns

resolve_nested_columns(cat, preset="rubin_dp1_object")
# NestedColumns(nested='objectForcedSource',
#               time='objectForcedSource.midpointMjdTai', ...)
```

:::{warning}
**Rubin fluxes are nJy and can legitimately be negative.** DP1 forced photometry — and
difference-imaging photometry in particular — has real negative flux measurements, so
converting to magnitudes would drop or corrupt them. Search in the **flux** domain: both
`rubin_*` presets set `domain=Domain.FLUX` for you, and you can force it anywhere else
with `domain="flux"`.
:::

## Two version worlds, one code path

nested-pandas renamed its dtype introspection API across releases, and the ecosystem is
currently split:

- **lsdb pins `nested-pandas < 0.7`**, which is the pandas-2 world.
- **standalone `nested-pandas >= 0.7`** is the pandas-3 world.

The adapter supports **both** with a single code path: it reads sub-column names through
whichever spelling the installed dtype offers, and it uses only `map_rows` /
`map_partitions` / `join_nested` — never `reduce`, which upstream removed in 0.7.0. You do
not need to pick a side; install whichever of `[nested]` / `[lsdb]` fits your stack.

:::{note}
The two worlds pin incompatible pandas majors, so a single environment cannot hold both.
cuPeriod's own `pyproject.toml` declares `[lsdb]` and `[dev]` as conflicting extras for
exactly this reason, and its CI exercises the adapter against the 0.7 line while the code
stays 0.6.10-compatible.
:::

## Failure handling

At survey scale a single bad light curve must never abort a run. An object that cannot be
searched — too few detections, no time baseline, a degenerate fit — yields NaN result
columns and the run continues. Filter on `best_period.notna()` afterwards to see how many
objects were searchable.

---

Next: {doc}`batch` — the file/DataFrame batch runner, for data that is not in a LINCC
catalog.
