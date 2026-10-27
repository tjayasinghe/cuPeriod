# Batch processing

{func}`cuperiod.batch_periodograms` runs one or more methods over **many** light curves —
thousands to millions — using a CPU process pool or the GPU, and writes one row per light
curve to Parquet or CSV. It's the scaling path for whole surveys.

## The basics

```python
import cuperiod as cup

if __name__ == "__main__":
    summary = cup.batch_periodograms(
        "lightcurves/*.parquet",     # inputs (see below)
        ["GLS", "BLS"],              # method(s)
        device="cpu",                # "cpu" or "gpu"
        workers=8,                   # None = auto
        sink="results/",             # file or directory; None = in memory
        n_best=10,                   # peaks stored per light curve
    )
    print(summary.n_done, summary.n_failed, summary.n_skipped)
```

:::{important}
Worker pools start fresh Python processes on every operating system, and each one
imports your script. Put a pooled batch call under `if __name__ == "__main__":` (as
above), or run with `workers=1`. An unguarded script now fails with a message saying
exactly that: it ends each worker's traceback and the final `BrokenProcessPool`, which
used to be bare. In a notebook or the REPL nothing needs guarding.
:::

## Inputs

`inputs` is flexible:

| Form | Example |
| --- | --- |
| A glob string | `"lcs/*.csv"` |
| A directory | `"lcs/"` |
| An iterable of light curves | `[lc1, lc2, ...]` |
| An iterable of `(key, lightcurve)` pairs | `[("star1", lc1), ...]` |
| An iterable of file paths | `["a.fits", "b.fits"]` |
| A `(DataFrame, group_column)` tuple | `(df, "object_id")` |

The last form splits one big table into per-object light curves on the group column —
ideal when an entire survey lives in a single Parquet file. Column handling
({class}`~cuperiod.ColumnMap`, `domain`, `band_column`) works exactly as in the
single-curve API; `ColumnMap(band=...)` makes each group a
{class}`~cuperiod.MultiBandLightCurve`, as it does for file inputs.

- A directory is scanned for every format {meth}`LightCurve.from_file
  <cuperiod.LightCurve.from_file>` reads — `.csv`, `.ecsv`, `.fits`/`.fit`/`.fts`/`.fz`,
  `.parquet`/`.pq`, `.tsv`/`.tab`, `.dat`, `.txt` — with the text and FITS formats also
  taken compressed (`star.csv.gz`, `star.fits.bz2`, `.xz`); other files are ignored. A
  glob takes whatever it matches.
- An existing path is always taken literally (even one containing `[`), and a glob that
  matches nothing raises `FileNotFoundError` instead of succeeding with no work.
- Keys are file stems; stems that collide (`a.csv` beside `a.parquet`, or the same name
  in two folders) get the shortest unique trailing path (`fieldB/lc_0000`) with a
  warning, and every row records its `source` file. A key that still repeats — given
  twice, or two light curves with one `meta["id"]` — keeps its first occurrence and the
  later ones become `key#2`, `key#3`, … (with a warning), so no two inputs ever share a
  result row or a resume entry; keys a CSV sink would store identically count as
  repeats. `(key, lightcurve)` keys may be strings or integers (NumPy integers
  included); a float key raises `TypeError`.
- `grid=` applies one trial grid to every light curve and method, as in
  {func}`~cuperiod.periodogram`, and per-method `settings` mappings match their keys like
  method names (a typo, or a settings object of the wrong class, raises). Rows honour
  the settings' peak fields (`n_peaks`, separations, alias tolerance,
  `downsample_points`).

## CPU vs GPU

- **`device="cpu"`** — a process pool. `workers=None` uses the CPUs this process may run
  on (its affinity mask and cgroup quota, not the machine's count) minus one, and every
  pool is capped at Windows' 61-process limit (a larger explicit request warns). While
  the pool exists, BLAS/OpenMP threads are pinned to 1 per worker and numba gets
  `cores // workers` threads per worker, so workers don't oversubscribe the cores.
  **A CPU pool never touches the GPU**: `backend="auto"` means the best CPU backend for
  each method, and a GPU-only backend (`"gpu"`, `"cupy"`, `"cufinufft"`,
  `"torch:cuda"`) raises `ValueError` before any work.
- **`device="gpu"`** — with one worker, a single long-lived process that reuses
  plan/kernel engines (the throughput path for one device). With several workers, a pool
  where each worker builds its own engines once, overlapping CPU-side work over the GPU.
  `"auto"` maps to the GPU, a concrete GPU backend (`torch`, `cupy`, `cufinufft`) is
  kept, and a CPU-only backend raises.
- `device="hybrid"` is planned and currently raises `NotImplementedError`.

`chunk_size=None` (the default) cuts the inputs into about four tasks per worker (at
most 256 light curves each), so small batches use the whole pool. Rows, errors and file
sinks come back in **input order** whatever order the workers finish in.

:::{note}
Before this release a default `device="cpu"` batch with `backend="auto"` ran every
CPU-pool worker on the GPU on CUDA machines. Such batches now run float64 CPU kernels:
values can move at the 1e-6 level and the rows' `backend` column changes.
:::

With `workers=None` a GPU pool is sized by {func}`~cuperiod.suggest_gpu_workers` for
the most demanding of the requested methods (every worker runs all of them) and for
`grid=` when one is given. It counts each worker's own CUDA context (~200–235 MiB
measured on an RTX 5070 Ti) plus the method's working set against the free memory, and
caps the count at 4 without NVIDIA MPS — the processes take turns on the device, so
GLS throughput measured 550 / 860 / 1150 / 1150 curves/s for 1 / 2 / 4 / 8 workers —
at 16 with MPS, and at the CPUs this process may use. To choose it yourself:

```python
summary = cup.batch_periodograms(
    (df, "object_id"), "GLS",
    device="gpu", workers=cup.suggest_gpu_workers("GLS"),
    sink="out.parquet",
)
```

## Output sinks

`sink` controls where results go:

- **`None`** — results are returned in memory as `summary.rows` (a list of dicts). Good for
  small runs and notebooks.
- **A `.parquet` / `.csv` file** — all rows written to that one file.
- **A directory** — one part file per task (`part-00000.parquet`, …), all with the same
  schema. This is the choice for large runs.

Failures are returned in `summary.errors` (below); `batch_periodograms` writes no
errors file. The `cuperiod batch` command ({doc}`cli`) also writes them to
`<stem>.errors.csv` beside a file sink (`out.parquet` → `out.errors.csv`), or to
`_errors.csv` inside a directory sink. Sinks are written through a uniquely named
temporary file and renamed into place; an unreadable existing sink (or part) is moved
aside to `<name>.corrupt-<timestamp>` rather than overwritten, and an impossible sink (a `.csv`
path that is a directory, an uncreatable location) is refused before anything is
computed. CSV sinks escape text cells a spreadsheet would run as a formula (`=`, `+`,
`-`, `@`, tab, CR; plain numbers untouched); Parquet stores values verbatim.

### Resumability

With `resume=True` (the default) both kinds of sink resume **by light curve**: a re-run
reads the (key, method) pairs already present and computes only what is missing, in new
part files, retrying the light curves that failed. An interrupted million-curve job
picks up where it left off, and adding, removing or reordering inputs — or resuming with
another `chunk_size` — neither drops nor duplicates a light curve:

```python
cup.batch_periodograms("survey/*.parquet", "GLS", sink="results/")   # writes parts
cup.batch_periodograms("survey/*.parquet", "GLS", sink="results/")   # computes nothing new
```

A directory records the request that wrote it. Adding a method is fine, but resuming a
directory that already holds part files with different settings, `n_best`, `store_raw`,
columns or grid — or with a method that would add per-peak columns — raises
`ValueError`, because the new rows would not match the old ones. A directory whose
earlier run produced no rows takes the new request. `resume=False` starts again: it
first removes the directory's `part-*.parquet` files (and nothing else). A file sink
that already holds everything is not rewritten.

## What's stored per light curve

Each row carries the input key, its `source`, the method, the backend, and the top
`n_best` peaks (period, power, and the method's `extra` scalars — depth/duration for BLS,
fap for GLS, etc.). Every row has a column `{extra}_{k}` for each of the method's
per-peak extras and every rank up to `n_best`, `NaN` where a light curve has fewer
peaks, so every part of a directory has one schema and dataset readers keep all
columns. BLS and TLS rows use the same alias-diverse peak policy as the CLI printout and
the GUI. Set `store_raw=True` to also store the peak-preserving downsampled spectrum.

## The summary

{func}`~cuperiod.batch_periodograms` returns a {class}`~cuperiod.BatchSummary`:

```python
summary.n_inputs     # total inputs discovered
summary.n_done       # results produced
summary.n_failed     # light curves that errored (one bad curve never kills the batch)
summary.n_skipped    # light curves whose every method was already in the sink
summary.methods      # methods run
summary.n_workers    # the worker count actually used
summary.errors       # list of (key, message) for failures
summary.rows         # the rows, when sink is None
```

Failures are isolated: a single malformed light curve is recorded in `summary.errors` and
the batch continues. A batch with no inputs warns and leaves its sink untouched.

## Pre-whitening and period changes at scale

{func}`~cuperiod.batch_prewhiten` ({doc}`prewhitening`) and {func}`~cuperiod.batch_pdot`
({doc}`period-change`) use the same inputs, workers and sinks, with the same device
rules: with `device="cpu"`, `"auto"` means a CPU backend (the method's settings backend
is the fallback) and a GPU-only backend raises. `batch_prewhiten(backend=None)` — the
default — takes `settings.backend`, so `CUPERIOD_PREWHITEN_BACKEND` applies, as for
{func}`~cuperiod.prewhiten`. Both resume by key and retry failures; `batch_pdot` does so
across its three tables (see {doc}`period-change`). Their rows carry the `source` file
too, and all three drivers warn when a resumed sink already holds a key that came from
a different file. For CPU pools without a directory
sink `chunk_size` is an upper bound, and rows and errors come back in input order: 16
light curves on 4 workers went from 13.9 s to 2.2 s for `batch_pdot`.

## Throughput

On one GPU, batch GLS peaks at **~728 light curves/second** for short survey curves
(>2.6 million/hour), and PDM at ~534/second. These are single-shot rates that include the
worker pool's start-up (process spawn and a CUDA context per worker); the worker sweep
above, which leaves start-up out, reached ~1150 GLS curves/second. With the `[fast]`
extra's multicore `numba` CPU tier, a CPU process pool on a 32-thread machine stays
within 1.0–1.8× of the GPU at the batch sizes where both were measured (256 and 1024
curves): GLS 1.2× and 1.0×, PDM
1.8× and 1.3× — see {doc}`../benchmarks` for the current breakdown before assuming the
GPU wins by default.

---

Next: {doc}`cli` — the same machinery from the command line.
