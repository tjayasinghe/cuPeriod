---
sd_hide_title: true
---

# cuPeriod

:::{div} sd-text-center sd-fs-2 sd-font-weight-bold
cuPeriod
:::

:::{div} sd-text-center sd-fs-5 sd-text-secondary
Optimized, GPU-accelerated periodograms for astronomy
:::

<br>

**cuPeriod** computes period-search statistics for variable stars and transiting
systems — from a single light curve to millions. One Python API, one command-line
tool, and an optional desktop GUI cover seven methods, each with a fast CPU backend and
GPU-accelerated paths: the NVIDIA CUDA fast paths plus a portable PyTorch backend that
also reaches AMD, Intel, and Apple GPUs (and a CPU-only path). Add frictionless column
handling, multi-band support, raw-spectrum output, and an N-best-periods utility.

Every implementation is validated against an established reference (astropy's
`LombScargle` / `BoxLeastSquares`, and others) to floating-point round-off.

```python
import cuperiod as cup

pg = cup.periodogram((time, mag, mag_err), "GLS")
print(pg.best_period())          # the most significant period, in days
for peak in pg.best_periods(10):
    print(peak.period, peak.power)
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Get started in 5 minutes
The {doc}`quickstart` runs end-to-end on a synthetic light curve — no data needed —
then shows how to load your own.
:::

:::{grid-item-card} 📖 Learn the package
The {doc}`User Guide <guide/index>` walks through inputs, methods, results, backends,
tuning, multi-band, batch, and the CLI.
:::

:::{grid-item-card} ⚡ Scale to millions
{doc}`Batch processing <guide/batch>` over CPU pools or the GPU, written to Parquet,
resumable across runs.
:::

:::{grid-item-card} 🔬 Trust the numbers
The {doc}`benchmarks` page shows parity, period recovery, and speedups on real
survey data.
:::

:::{grid-item-card} 🖥️ Explore interactively
The {doc}`desktop GUI <guide/gui>` runs any method and folds the light curve live as
you drag across peaks — `pip install "cuperiod[gui]"`, then `cuperiod-gui`.
:::

::::

## Which method should I use?

| Method | What it's for | GPU | Multi-band |
| --- | --- | :---: | :---: |
| **GLS** | general variability (generalized Lomb–Scargle) | ✅ | ✅ |
| **BLS** | eclipses / box-like transits | ✅ | ✅ |
| **MHAOV** | sharply non-sinusoidal signals (multiharmonic AOV) | ✅ | ✅ |
| **TLS** | limb-darkened transit matched filter | ✅ | — |
| **PDM** | non-sinusoidal folds (Stellingwerf) | ✅ | — |
| **CE** | sparse survey data (conditional entropy) | ✅ | — |
| **String-Length** | eclipsing / eccentric shapes | ✅ | — |

All seven share one API, one CLI, and the full single/batch machinery. See
{doc}`guide/methods` for a decision guide.

## Install

```bash
pip install cuperiod            # CPU (numpy, scipy, astropy, finufft)
pip install "cuperiod[gpu]"     # + CUDA 12 GPU backends (cupy, cufinufft)
pip install "cuperiod[torch]"   # + portable PyTorch backend (AMD/Intel/Apple GPUs + CPU)
pip install "cuperiod[fast]"    # + numba multicore CPU kernels (all methods, 20-300×)
pip install "cuperiod[gui]"     # + interactive desktop GUI (cuperiod-gui)
```

See {doc}`installation` for the full matrix and GPU requirements.

:::{toctree}
:hidden:
:caption: Getting started

installation
quickstart
:::

:::{toctree}
:hidden:
:caption: User guide

guide/index
guide/light-curves
guide/methods
guide/results
guide/backends
guide/tuning
guide/multiband
guide/batch
guide/cli
guide/gui
:::

:::{toctree}
:hidden:
:caption: Reference

benchmarks
api/index
:::

:::{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/tjayasinghe/cuPeriod>
:::
