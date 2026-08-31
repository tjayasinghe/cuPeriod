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
frictionless column handling, joint multi-band search, alias diagnostics, raw-spectrum
output, and an N-best-periods utility.

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
  (PyAstronomy, the `supersmoother` package and gatspy, or a direct implementation of the
  published algorithm) on identical grids. Across the seven benchmark-suite methods, CPU,
  CUDA, and the portable PyTorch backend agree to round-off and pick the
  identical best period on 126 real ASAS-SN light curves
  across six variability classes plus 12 confirmed *Kepler* transits — 88–96%
  harmonic-aware period recovery on this deliberately heterogeneous sample (a synthetic
  injection–recovery sweep further characterizes sensitivity vs. signal-to-noise) — see the
  [full benchmark report](benchmarks/REPORT.md) and the
  [benchmarks docs page](https://cuperiod.readthedocs.io/en/latest/benchmarks.html). The
  multi-band methods are further validated blind on 100 real SDSS Stripe 82 RR Lyrae with
  literature periods — the pooled fold statistics recover up to 93% of periods strictly
  (97% harmonic-aware) on real five-band data.
- **A fast CPU tier, no GPU required.** The `[fast]` extra's multicore `numba` kernels
  make `backend="cpu"` 20x faster than astropy's `BoxLeastSquares` and 2177x faster than
  PyAstronomy's PDM on a representative light curve, while recovering the same periods.
- **GPU acceleration beyond NVIDIA.** The portable PyTorch backend runs every method on
  AMD (ROCm), Intel (XPU), and Apple (MPS) GPUs, in addition to the NVIDIA CUDA fast
  paths — so the accelerated code isn't locked to one vendor.
- **Built for catalogue scale.** `batch_periodograms` sustains up to 574 light curves/s
  (>2 million/hour) on a single GPU, with a resumable batch sink for runs spanning
  millions of curves.
- **Eight methods, one API.** GLS, BLS, PDM, CE, String-Length, MHAOV, TLS, and
  SuperSmoother share one entry point, one CLI, and an optional desktop GUI, with
  frictionless column handling.
- **Multi-band search that earns its keep.** Seven of the eight methods fit several filters
  of one star jointly. The native multi-band GLS offers three models — shared-phase
  offsets (the default), independent per-band sinusoids, and a regularized per-band
  harmonic model — runs on every backend (a six-band star over 200k frequencies: 55 s
  through astropy, **0.13 s** native), and comes with **bootstrap false-alarm
  probabilities**, which astropy's `LombScargleMultiband` does not provide at all. On a
  simulated Rubin-like cadence with 30 epochs spread over six bands, single-band GLS
  recovers 0% of faint RRab proxies and the shared-phase model recovers **82%**.
- **Survey catalogues, in place.** `cuperiod.interop` runs a search directly over LINCC
  nested-pandas / lsdb light curves — one row per object, epochs in a nested column — with
  no flattening or `groupby`. A partition tier reuses **one GPU engine** across every
  object in a dask partition, with verified column presets for ZTF and Rubin DP1.
- **Alias-checked periods.** `alias_diagnostics` measures the sampling's spectral window,
  predicts the alias family it implies, scores each competitor against the peak you got,
  and says plainly when the period is ambiguous.
- **Pulsators get a frequency solution, not just a period.** `prewhiten` automates the
  whole Period04-style loop — GPU/NUFFT amplitude spectrum, iterative sinusoid extraction,
  simultaneous re-fitting, principled stopping criteria, propagated uncertainties,
  combination-frequency identification, and g-mode period-spacing tools — for one star or
  a million.

## Status

Implemented now, each with optimized CPU and GPU backends:

| Method | What it's for | GPU | Multi-band |
| --- | --- | --- | --- |
| **GLS** | general variability (Lomb-Scargle) | yes | yes |
| **BLS** | eclipses / box-like transits | yes | yes |
| **MHAOV** | sharply non-sinusoidal signals (multiharmonic AOV) | yes | yes |
| **TLS** | limb-darkened transit matched filter | yes | — |
| **PDM** | non-sinusoidal folds (Stellingwerf) | yes | yes |
| **CE** | sparse survey data (conditional entropy) | yes | yes |
| **String-Length** | eclipsing / eccentric shapes | yes | yes |
| **SuperSmoother** | any repeating shape, non-parametric (Friedman) | yes | yes |

All eight methods have CPU and GPU backends, plus the full single/batch/CLI machinery.

## Install

```bash
pip install cuperiod            # CPU (numpy, scipy, astropy, finufft)
pip install "cuperiod[gpu]"     # + CUDA 12 GPU backends (cupy, cufinufft)
pip install "cuperiod[torch]"   # + portable PyTorch backend (AMD/Intel/Apple GPUs + CPU)
pip install "cuperiod[fast]"    # + numba multicore CPU kernels (7 methods, no GLS, 20-300x)
pip install "cuperiod[gui]"     # + interactive desktop GUI (cuperiod-gui)
pip install "cuperiod[pandas]"  # + pandas DataFrame ingestion
pip install "cuperiod[nested]"  # + nested-pandas light curves (cuperiod.interop)
pip install "cuperiod[lsdb]"    # + lsdb HATS catalogs (lazy, dask-partitioned)
```

The `[gpu]` extra needs an NVIDIA GPU with the CUDA 12 runtime; it pulls in `cupy-cuda12x`,
`cufinufft`, and the CUDA runtime wheels. The `[torch]` extra adds a portable PyTorch
backend that reaches AMD (ROCm), Intel (XPU), and Apple-Silicon (MPS) GPUs — and a CPU path
everywhere — so the accelerated code runs beyond NVIDIA (install the wheel matching your
accelerator from [pytorch.org](https://pytorch.org/get-started/locally/); the default is
CPU-only). The `[fast]` extra adds multicore `numba` CPU kernels that become the default
`"cpu"`/`"auto"` backend for **every method but GLS** — BLS, PDM, CE, String-Length,
MHAOV, TLS, and SuperSmoother — one to two orders of magnitude faster than the fallback
CPU paths and matching them to floating point.

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

GLS, BLS, MHAOV, PDM, CE, String-Length, and SuperSmoother jointly model two or more bands
of the same star — one period, but each band keeps its own mean, amplitude, and
normalization:

```python
mb = cup.MultiBandLightCurve.from_light_curves({"g": lc_g, "r": lc_r})
pg = cup.periodogram(mb, "GLS")          # VanderPlas & Ivezić shared-phase model

# ...or straight from a long-format file / the CLI / batch:
mb = cup.MultiBandLightCurve.from_file("star_ugrizy.parquet", band_column="band")

# One of three joint models (offsets / perband / flex):
settings = cup.GLSSettings(mb_model="flex")
pg = cup.periodogram(mb, "GLS", settings=settings)

# Honest false-alarm probabilities, from a within-band bootstrap:
calib = cup.multiband_fap(mb, settings, n_bootstrap=1000)
print(calib.fap(pg.best_periods(1)[0].power), calib.level(0.01))
```

`mb_model` is `"offsets"` (shared phase + per-band offsets, the default), `"perband"`
(independent per-band sinusoids, `chi2_0`-weighted), or `"flex"` (regularized per-band
harmonics, matching astropy to ~2e-10). All three run natively on finufft, cufinufft, and
torch — astropy is kept only as the reference they're tested against.

Period recovery on a simulated Rubin-like six-band cadence
(`benchmarks/multiband_recovery.py`; faint RRab proxies, 0.20 mag per-point noise, 300
stars per cell, top period within 1% and no harmonic credit), by *total* epochs across all
six bands:

| strategy | 30 epochs | 60 epochs | 120 epochs |
| --- | --- | --- | --- |
| best single band (r) | 0.0% | 37.3% | 97.3% |
| any single band | 0.0% | 49.3% | 99.0% |
| multi-band `offsets` (1,0) | **81.7%** | **99.7%** | 100.0% |

`perband` and `flex` land in between (17.7% / 20.0% at 30 epochs, 97.0% at 60): sharing
the phase is what buys sparse-cadence recovery, which is why `"offsets"` is the default.
These are simulations on a deliberately simplified cadence (random nights, no rolling
cadence) — read the ordering, not the absolute numbers. See the
[multi-band guide](https://cuperiod.readthedocs.io/en/latest/guide/multiband.html).

### Survey catalogs (LINCC: nested-pandas / lsdb)

```python
from cuperiod.interop import nested_periodogram, partition_periodogram

out = nested_periodogram(frame, "lc", preset="ztf_dr22")            # row-wise
res = partition_periodogram(cat, preset="rubin_dp1_object",         # one GPU engine
                            method="GLS", backend="gpu")            # per partition
```

Runs on the nested layout directly — no flattening, no `groupby`. Column presets for ZTF
DR22, ZTF alerts, and Rubin DP1 object/DIA photometry; a failed object yields NaN rather
than aborting the run. See the
[interop guide](https://cuperiod.readthedocs.io/en/latest/guide/interop.html).

### Backends

`backend="auto"` (default) uses the GPU when available and falls back to CPU. Force a path
with `backend="cpu"`, `backend="gpu"`, or a concrete name (`"finufft"`, `"cufinufft"`,
`"numpy"`, `"astropy"`, `"cupy"`). The portable PyTorch backend runs any method on any torch
device: `backend="torch"` (best device present) or `"torch:cpu"` / `"torch:cuda"` /
`"torch:mps"` / `"torch:xpu"`. On Apple MPS (no float64) it uses float32; `precision="auto"`
keeps float64 everywhere else.

## Pre-whitening a pulsator

```python
solution = cup.prewhiten((time, mag, mag_err))
print(solution.summary())        # ranked frequencies with 1-sigma uncertainties and S/N

for c in solution.components:
    print(c.label, c.frequency, c.frequency_error, c.snr, c.combination)

series = cup.find_period_spacing(  # gamma Dor / SPB g-mode pattern
    [c.period for c in solution.independent()],
    [c.amplitude for c in solution.independent()],
)
```

Every judgement call an interactive session leaves to the operator is an explicit setting,
and the result records **why the extraction stopped**. See the
[pre-whitening guide](https://cuperiod.readthedocs.io/en/latest/guide/prewhitening.html).

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
cuperiod prewhiten star.csv --snr 4.6 -n 30 --spacing
cuperiod batch-prewhiten "lcs/*.csv" --out modes.parquet
cuperiod methods                 # list methods and backends
cuperiod gpu-info                # device + suggested worker counts
cuperiod doctor                  # backends, torch devices, precision, auto-resolution
cuperiod grid-info star.fits -m GLS
```

`run` accepts `--time/--value/--error` column overrides and `--domain magnitude|flux`, and
can write JSON (`--out`) and the raw spectrum (`--save-periodogram`). `--band` splits a
long-format file on its filter column and runs a **joint** multi-band fit; the Python API's
`batch_periodograms(..., band_column=...)` takes the same (the `batch` command has no band
option yet).

## Desktop GUI

An interactive explorer ships with the `[gui]` extra — run any method with all of its
options, explore the full-resolution spectrum, and watch the phased light curve update
live as you drag across peaks. Single curves or a whole folder in batch mode, multi-band
overlays, and a dark/light theme remembered across launches. An **Analysis** picker
switches the same window to pre-whitening: the amplitude spectrum with the residual
overlaid, a frequency table with uncertainties, and a g-mode period-spacing explorer.

```bash
pip install "cuperiod[gui]"      # add [gpu] or [torch] for accelerated backends
cuperiod-gui                     # or:  python -m cuperiod.gui
```

The toolbar's **Load demo** menu always offers a synthetic multi-band curve, so there's
something to explore on first launch. In a source checkout (or with `CUPERIOD_EXAMPLE_DATA`
pointed at `examples/data`) it also lists a *Kepler* transit and six ASAS-SN variables.

## Light-curve inputs

Time may be JD/HJD/BJD/MJD; values may be magnitude or flux; errors are optional. Column
names are auto-detected (case-insensitive) and can be pinned with `ColumnMap`. Box/transit
methods (BLS, TLS) work in flux — magnitudes are converted automatically.

## Citing cuPeriod

If you use cuPeriod in your research, please cite the Research Note
([ADS](https://ui.adsabs.harvard.edu/abs/2026RNAAS..10..244J),
[doi:10.3847/2515-5172/ae9b06](https://doi.org/10.3847/2515-5172/ae9b06)). It is also the
`preferred-citation` in [`CITATION.cff`](CITATION.cff), so GitHub's "Cite this repository"
button and `cffconvert` both hand it back.

```bibtex
@ARTICLE{2026RNAAS..10..244J,
       author = {{Jayasinghe}, Tharindu},
        title = "{cuPeriod: Seven Validated Periodograms for CPUs and Any GPU}",
      journal = {Research Notes of the American Astronomical Society},
     keywords = {Stellar astronomy, Time domain astronomy, Variable stars, Astronomy software, 1583, 2109, 1761, 1855},
         year = 2026,
        month = aug,
       volume = {10},
       number = {8},
          eid = {244},
        pages = {244},
          doi = {10.3847/2515-5172/ae9b06},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026RNAAS..10..244J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The Note describes the v1.1 release (the seven methods of its title); multi-band, SuperSmoother
and pre-whitening arrived in v1.2. Cite the version you ran alongside it if that matters for
reproducibility — `cuperiod.__version__` reports it.

## License

GPL-3.0-or-later.
