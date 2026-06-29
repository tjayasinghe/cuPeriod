# Batch processing

{func}`cuperiod.batch_periodograms` runs one or more methods over **many** light curves —
thousands to millions — using a CPU process pool or the GPU, and writes one row per light
curve to Parquet or CSV. It's the scaling path for whole surveys.

## The basics

```python
summary = cup.batch_periodograms(
    "lightcurves/*.parquet",     # inputs (see below)
    ["GLS", "BLS"],              # method(s)
    device="cpu",                # "cpu" or "gpu"
    workers=8,                   # None = auto
    sink="results/",            # file or directory; None = in memory
    n_best=10,                   # peaks stored per light curve
)
print(summary.n_done, summary.n_failed, summary.n_skipped)
```

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
single-curve API.

## CPU vs GPU

- **`device="cpu"`** — a process pool, one chunk per worker, math threads pinned to 1 so
  workers don't oversubscribe cores. `workers=None` uses all-but-one core.
- **`device="gpu"`** — with one worker, a single long-lived process that reuses
  plan/kernel engines (the throughput path for one device). With several workers, a pool
  where each worker builds its own engines once, overlapping CPU-side work over the GPU.
- `device="hybrid"` is planned and currently raises `NotImplementedError`.

Size the GPU pool from probed device memory with
{func}`~cuperiod.suggest_gpu_workers`:

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
- **A directory** — one part file per chunk (`part-00000.parquet`, …). This is the choice
  for large runs because it's **resumable**.

### Resumability

With a directory sink and `resume=True` (the default), a re-run **skips chunks whose part
file already exists** — so an interrupted million-curve job picks up where it left off,
and adding new inputs only processes the new chunks:

```python
cup.batch_periodograms("survey/*.parquet", "GLS", sink="results/")   # writes parts
cup.batch_periodograms("survey/*.parquet", "GLS", sink="results/")   # skips done parts
```

A single-file sink is also merged on resume (existing keys are kept, new ones appended).

## What's stored per light curve

Each row carries the input key, the method, the backend, and the top `n_best` peaks
(period, power, and the method's `extra` scalars — depth/duration for BLS, fap for GLS,
etc.). Set `store_raw=True` to also store the peak-preserving downsampled spectrum.

## The summary

{func}`~cuperiod.batch_periodograms` returns a {class}`~cuperiod.BatchSummary`:

```python
summary.n_inputs     # total inputs discovered
summary.n_done       # results produced
summary.n_failed     # light curves that errored (one bad curve never kills the batch)
summary.n_skipped    # chunks skipped on resume
summary.methods      # methods run
summary.errors       # list of (key, message) for failures
summary.rows         # the rows, when sink is None
```

Failures are isolated: a single malformed light curve is recorded in `summary.errors` and
the batch continues.

## Throughput

On one GPU, batch GLS peaks at **~620 light curves/second** for short survey curves
(>2 million/hour). The GPU's edge grows with grid size, point count, and the
box/fold methods — see {doc}`../benchmarks`.

---

Next: {doc}`cli` — the same machinery from the command line.
