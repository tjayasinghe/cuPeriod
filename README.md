# cuPeriod

**Optimized, GPU-accelerated periodograms for astronomy.**

cuPeriod computes period-search statistics for variable stars and transiting systems with
fast CPU backends and CUDA-accelerated paths that scale from a single light curve to
millions. One API and CLI cover every method, with frictionless column handling, multi-band
support, raw-spectrum output, and an N-best-periods utility.

Every implementation is validated against the standard reference (astropy
`LombScargle` / `BoxLeastSquares`) to floating-point round-off.

## Status

Implemented now, each with optimized CPU and GPU backends:

| Method | What it's for | GPU | Multi-band |
| --- | --- | --- | --- |
| **GLS** | general variability (Lomb-Scargle) | yes | yes |
| **BLS** | eclipses / box-like transits | yes | yes |
| **MHAOV** | sharply non-sinusoidal signals (multiharmonic AOV) | yes | planned |
| **TLS** | limb-darkened transit matched filter | planned | — |
| **PDM** | non-sinusoidal folds (Stellingwerf) | yes | — |
| **CE** | sparse survey data (conditional entropy) | yes | — |
| **String-Length** | eclipsing / eccentric shapes | yes | — |

Plus the full single/batch/CLI machinery. Planned next: a GPU kernel for TLS and multi-band
MHAOV.

## Install

```bash
pip install cuperiod            # CPU (numpy, scipy, astropy, finufft)
pip install "cuperiod[gpu]"     # + CUDA 12 GPU backends (cupy, cufinufft)
pip install "cuperiod[pandas]"  # + pandas DataFrame ingestion
```

GPU acceleration needs an NVIDIA GPU with the CUDA 12 runtime; the `[gpu]` extra pulls in
`cupy-cuda12x`, `cufinufft`, and the CUDA runtime wheels.

## Quick start (Python)

The recommended import alias is `cup`:

```python
import cuperiod as cup

# A single light curve, straight from arrays:
pg = cup.periodogram((time, mag, mag_err), "GLS")
print(pg.best_period())
for peak in pg.best_periods(10):
    print(peak.period, peak.power, peak.extra.get("fap"))

# From a table with arbitrary column names (auto-detected, or pinned):
pg = cup.periodogram(df, "BLS",
                     columns=cup.ColumnMap(time="HJD", value="flux", error="flux_err"))

# Several methods at once:
res = cup.periodogram(lc, ["GLS", "BLS"])      # -> MultiResult
res["BLS"].best_periods(5, alias_diverse=True)

# Raw spectrum for your own analysis:
frequency, power = pg.frequency, pg.power
```

Method names are case-insensitive (`"gls"` == `"GLS"`).

### Multi-band (one star, several filters)

GLS, BLS, and (soon) MHAOV jointly model two or more bands of the same star:

```python
mb = cup.MultiBandLightCurve.from_light_curves({"g": lc_g, "r": lc_r})
pg = cup.periodogram(mb, "GLS")          # VanderPlas & Ivezić shared-phase model
```

### Backends

`backend="auto"` (default) uses the GPU when available and falls back to CPU. Force a path
with `backend="cpu"`, `backend="gpu"`, or a concrete name (`"finufft"`, `"cufinufft"`,
`"numpy"`, `"astropy"`, `"cupy"`).

## Batch processing (millions of light curves)

```python
# CPU pool across cores, written to Parquet:
cup.batch_periodograms("lightcurves/*.parquet", ["GLS", "BLS"],
                       device="cpu", workers=8, sink="results/")

# GPU, with an auto-sized worker count:
cup.batch_periodograms(df_groups, "GLS", device="gpu",
                       workers=cup.suggest_gpu_workers("GLS"), sink="out.parquet")
```

Inputs can be an iterable of light curves, a glob, a directory, or a `(DataFrame,
group_column)` pair. A directory sink is **resumable** — re-running skips chunks already
written. `suggest_gpu_workers` sizes the GPU pool from probed device memory.

## Command line

```bash
cuperiod run star.csv --method GLS,BLS --n-best 10
cuperiod batch "lcs/*.csv" --method GLS --device gpu --out results/
cuperiod methods                 # list methods and backends
cuperiod gpu-info                # device + suggested worker counts
cuperiod grid-info star.fits -m GLS
```

`run` accepts `--time/--value/--error/--band` overrides and `--domain magnitude|flux`, and
can write JSON (`--out`) and the raw spectrum (`--save-periodogram`).

## Light-curve inputs

Time may be JD/HJD/BJD/MJD; values may be magnitude or flux; errors are optional. Column
names are auto-detected (case-insensitive) and can be pinned with `ColumnMap`. Box/transit
methods (BLS, TLS) work in flux — magnitudes are converted automatically.

## License

GPL-3.0-or-later.
