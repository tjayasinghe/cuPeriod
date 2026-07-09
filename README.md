# cuPeriod

**Optimized, GPU-accelerated periodograms for astronomy.**

[![Documentation Status](https://readthedocs.org/projects/cuperiod/badge/?version=latest)](https://cuperiod.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/cuperiod)](https://pypi.org/project/cuperiod/)
[![Python versions](https://img.shields.io/pypi/pyversions/cuperiod)](https://pypi.org/project/cuperiod/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/tjayasinghe/cuPeriod/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tjayasinghe/cuPeriod/actions/workflows/ci.yml)

cuPeriod computes period-search statistics for variable stars and transiting systems with
fast CPU backends and GPU-accelerated paths that scale from a single light curve to
millions. The NVIDIA CUDA fast paths are joined by a portable **PyTorch** backend that also
runs on AMD, Intel, and Apple GPUs (and a CPU-only path), so the accelerated code is no
longer NVIDIA-only. One API, one CLI, and an optional desktop GUI cover every method, with
frictionless column handling, multi-band support, raw-spectrum output, and an
N-best-periods utility.

📖 **Documentation:** <https://cuperiod.readthedocs.io> — a [5-minute
quickstart](https://cuperiod.readthedocs.io/en/latest/quickstart.html), a full user guide,
and the complete API reference, including
[installation](https://cuperiod.readthedocs.io/en/latest/installation.html),
[backends](https://cuperiod.readthedocs.io/en/latest/guide/backends.html),
[the GUI](https://cuperiod.readthedocs.io/en/latest/guide/gui.html), and
[benchmarks](https://cuperiod.readthedocs.io/en/latest/benchmarks.html) pages.

Every implementation is validated against the standard reference (astropy
`LombScargle` / `BoxLeastSquares`) to floating-point round-off.

## Why cuPeriod

- **Validated, not just fast.** GLS and BLS match astropy to floating-point round-off,
  and every other method is checked against an independent reference implementation
  (PyAstronomy, or a direct implementation of the published algorithm) on identical
  grids. CPU, CUDA, and the portable PyTorch backend agree to round-off and pick the
  identical best period on 126 real ASAS-SN light curves
  across six variability classes plus 12 confirmed *Kepler* transits — 88–96%
  harmonic-aware period recovery on this deliberately heterogeneous sample (a synthetic
  injection–recovery sweep further characterizes sensitivity vs. signal-to-noise) — see the
  [full benchmark report](benchmarks/REPORT.md) and the
  [benchmarks docs page](https://cuperiod.readthedocs.io/en/latest/benchmarks.html).
- **A fast CPU tier, no GPU required.** The `[fast]` extra's multicore `numba` kernels
  make `backend="cpu"` 18x faster than astropy's `BoxLeastSquares` and 2106x faster than
  PyAstronomy's PDM on a representative light curve, while recovering the same periods.
- **GPU acceleration beyond NVIDIA.** The portable PyTorch backend runs every method on
  AMD (ROCm), Intel (XPU), and Apple (MPS) GPUs, in addition to the NVIDIA CUDA fast
  paths — so the accelerated code isn't locked to one vendor.
- **Built for catalogue scale.** `batch_periodograms` sustains up to 587 light curves/s
  (>2 million/hour) on a single GPU, with a resumable batch sink for runs spanning
  millions of curves.
- **Seven methods, one API.** GLS, BLS, PDM, CE, String-Length, MHAOV, and TLS share one
  entry point, one CLI, and an optional desktop GUI, with frictionless column handling and
  multi-band support.

## Status

Implemented now, each with optimized CPU and GPU backends:

| Method | What it's for | GPU | Multi-band |
| --- | --- | --- | --- |
| **GLS** | general variability (Lomb-Scargle) | yes | yes |
| **BLS** | eclipses / box-like transits | yes | yes |
| **MHAOV** | sharply non-sinusoidal signals (multiharmonic AOV) | yes | yes |
| **TLS** | limb-darkened transit matched filter | yes | — |
| **PDM** | non-sinusoidal folds (Stellingwerf) | yes | — |
| **CE** | sparse survey data (conditional entropy) | yes | — |
| **String-Length** | eclipsing / eccentric shapes | yes | — |

All seven methods have CPU and GPU backends, plus the full single/batch/CLI machinery.

## Install

```bash
pip install cuperiod            # CPU (numpy, scipy, astropy, finufft)
pip install "cuperiod[gpu]"     # + CUDA 12 GPU backends (cupy, cufinufft)
pip install "cuperiod[torch]"   # + portable PyTorch backend (AMD/Intel/Apple GPUs + CPU)
pip install "cuperiod[fast]"    # + numba multicore CPU kernels (all 7 methods, 20-300x)
pip install "cuperiod[gui]"     # + interactive desktop GUI (cuperiod-gui)
pip install "cuperiod[pandas]"  # + pandas DataFrame ingestion
```

The `[gpu]` extra needs an NVIDIA GPU with the CUDA 12 runtime; it pulls in `cupy-cuda12x`,
`cufinufft`, and the CUDA runtime wheels. The `[torch]` extra adds a portable PyTorch
backend that reaches AMD (ROCm), Intel (XPU), and Apple-Silicon (MPS) GPUs — and a CPU path
everywhere — so the accelerated code runs beyond NVIDIA (install the wheel matching your
accelerator from [pytorch.org](https://pytorch.org/get-started/locally/); the default is
CPU-only). The `[fast]` extra adds multicore `numba` CPU kernels that become the default
`"cpu"`/`"auto"` backend for **every** method — BLS, PDM, CE, String-Length, MHAOV, and
TLS — one to two orders of magnitude faster than the fallback CPU paths and matching them
to floating point.

Not sure what will run where? `cuperiod doctor` reports every installed backend, the torch
devices it sees and the precision each uses, and what `backend="auto"` resolves to.

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

> 📓 **New here?** The [`examples/cuperiod_tour.ipynb`](examples/cuperiod_tour.ipynb)
> notebook works through real light curves — a Cepheid, an RR Lyrae, an eclipsing binary,
> a Mira, and a *Kepler* exoplanet — showing each periodogram and phase-folded result.

### Multi-band (one star, several filters)

GLS, BLS, and MHAOV jointly model two or more bands of the same star:

```python
mb = cup.MultiBandLightCurve.from_light_curves({"g": lc_g, "r": lc_r})
pg = cup.periodogram(mb, "GLS")          # VanderPlas & Ivezić shared-phase model
```

### Backends

`backend="auto"` (default) uses the GPU when available and falls back to CPU. Force a path
with `backend="cpu"`, `backend="gpu"`, or a concrete name (`"finufft"`, `"cufinufft"`,
`"numpy"`, `"astropy"`, `"cupy"`). The portable PyTorch backend runs any method on any torch
device: `backend="torch"` (best device present) or `"torch:cpu"` / `"torch:cuda"` /
`"torch:mps"` / `"torch:xpu"`. On Apple MPS (no float64) it uses float32; `precision="auto"`
keeps float64 everywhere else.

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
cuperiod doctor                  # backends, torch devices, precision, auto-resolution
cuperiod grid-info star.fits -m GLS
```

`run` accepts `--time/--value/--error/--band` overrides and `--domain magnitude|flux`, and
can write JSON (`--out`) and the raw spectrum (`--save-periodogram`).

## Desktop GUI

An interactive periodogram explorer ships with the `[gui]` extra — run any method with all
of its options, explore the full-resolution spectrum, and watch the phased light curve
update live as you drag across peaks. Single curves or a whole folder in batch mode,
multi-band overlays, and a dark/light theme remembered across launches.

```bash
pip install "cuperiod[gui]"      # add [gpu] or [torch] for accelerated backends
cuperiod-gui                     # or:  python -m cuperiod.gui
```

It opens with bundled demo light curves (a *Kepler* transit, six ASAS-SN variables, a
synthetic multi-band curve), so there's something to explore on first launch.

## Light-curve inputs

Time may be JD/HJD/BJD/MJD; values may be magnitude or flux; errors are optional. Column
names are auto-detected (case-insensitive) and can be pinned with `ColumnMap`. Box/transit
methods (BLS, TLS) work in flux — magnitudes are converted automatically.

## Citing cuPeriod

If you use cuPeriod in your research, please cite it — see [`CITATION.cff`](CITATION.cff)
for the machine-readable record (also picked up by GitHub's "Cite this repository").

```bibtex
@software{jayasinghe_cuperiod,
  author  = {Jayasinghe, Tharindu},
  title   = {cuPeriod},
  version = {1.1.0},
  date    = {2026-07-08},
  url     = {https://github.com/tjayasinghe/cuPeriod}
}
```

## License

GPL-3.0-or-later.
