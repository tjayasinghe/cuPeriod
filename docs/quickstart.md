# Quickstart

This page gets you from nothing to a recovered period in about five minutes. Every
snippet runs on the **CPU** with the base install — no GPU, and no data of your own
required: we'll make a synthetic light curve first, then point you at loading real data.

The recommended import alias is `cup`:

```python
import cuperiod as cup
```

## 1. Make a light curve to play with

A *light curve* is just times, brightnesses, and (optionally) brightness errors. Let's
plant a sinusoidal signal at a known period so we can check the answer:

```python
import numpy as np

rng = np.random.default_rng(42)
n = 500
time = np.sort(rng.uniform(0.0, 120.0, n))        # 500 epochs over ~4 months (days)
true_period = 3.14                                # days
mag = 12.0 + 0.4 * np.sin(2 * np.pi * time / true_period)
mag += 0.03 * rng.standard_normal(n)              # photometric noise
err = np.full(n, 0.03)                            # 1-sigma uncertainties
```

## 2. Run a periodogram

Pass the arrays as a `(time, value, error)` tuple and name a method. **GLS**
(generalized Lomb–Scargle) is the workhorse for periodic variables:

```python
pg = cup.periodogram((time, mag, err), "GLS")
print(f"best period = {pg.best_period():.4f} d")
# best period = 3.1379 d   (≈ the planted 3.14)
```

`pg` is a {class}`~cuperiod.Periodogram` — the full spectrum plus peak-finding helpers.
Method names are case-insensitive, so `"gls"` and `"GLS"` are the same.

## 3. Read the best periods

`best_periods(n)` returns the `n` most significant peaks, already ranked and separated
so you don't get the same peak ten times:

```python
for peak in pg.best_periods(5):
    print(f"rank {peak.rank}: P = {peak.period:8.4f} d   power = {peak.power:.3f}")
```

Each {class}`~cuperiod.Peak` carries its `period`, `frequency`, `power`, `rank`, and a
method-specific `extra` dict. For GLS that includes a false-alarm probability:

```python
top = pg.best_periods(1)[0]
print(top.period, top.extra.get("fap"))
```

## 4. Get the raw spectrum

For your own plotting or analysis, the spectrum arrays are right there — stored in
ascending frequency, with `period = 1 / frequency`:

```python
frequency = pg.frequency       # cycles/day, ascending
power = pg.power               # the GLS statistic
period = pg.period             # days
```

A quick plot (matplotlib is not a cuPeriod dependency — `pip install matplotlib`):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(pg.period, pg.power)
ax.set(xscale="log", xlabel="period (days)", ylabel="GLS power")
ax.axvline(top.period, color="C1", ls="--", label=f"best = {top.period:.3f} d")
ax.legend()
plt.show()
```

## 5. Phase-fold on the best period

Folding the light curve on the recovered period should reveal the sinusoid:

```python
phase = (time / top.period) % 1.0
fig, ax = plt.subplots()
ax.scatter(phase, mag, s=8)
ax.invert_yaxis()                      # magnitudes: brighter is up
ax.set(xlabel="phase", ylabel="magnitude")
plt.show()
```

## Loading real data instead

Real light curves usually live in a table or file with their own column names.
cuPeriod auto-detects the common ones (`HJD`/`BJD`/`MJD`, `mag`/`flux`, `e_mag`/…):

```python
# From a file (CSV / ECSV / FITS / Parquet):
from cuperiod import LightCurve
lc = LightCurve.from_file("star.csv")
pg = cup.periodogram(lc, "GLS")

# From a DataFrame, pinning columns explicitly when needed:
pg = cup.periodogram(df, "BLS",
                     columns=cup.ColumnMap(time="HJD", value="flux", error="flux_err"))
```

See {doc}`guide/light-curves` for every accepted input form.

## Several filters of one star

Model all bands jointly instead of searching each on its own — one period, per-band
offsets, and far better recovery when no single band is well sampled:

```python
mb = cup.MultiBandLightCurve.from_file("star_ugrizy.csv", band_column="band")
pg = cup.periodogram(mb, "GLS")          # shared-phase model (VanderPlas & Ivezić)
print(pg.best_period(), pg.meta["bands"])
```

See {doc}`guide/multiband`.

## Running several methods at once

Pass a list of methods to get a {class}`~cuperiod.MultiResult` keyed by method name:

```python
res = cup.periodogram((time, mag, err), ["GLS", "PDM"])
print(res["GLS"].best_period())
print(res["PDM"].best_period())
```

## Using the GPU

If you installed the `[gpu]` extra and have a CUDA device, the default
`backend="auto"` already uses it. To force a path:

```python
pg = cup.periodogram((time, mag, err), "GLS", backend="gpu")   # or "cpu"
```

The result is identical to floating-point round-off; the GPU just computes it faster on
large grids and big catalogs. On a non-NVIDIA GPU (AMD, Intel, or Apple), install the
`[torch]` extra and use `backend="torch"` instead. See {doc}`guide/backends`.

## Analysing a pulsator

A multiperiodic star needs the whole frequency solution, not one best period:

```python
solution = cup.prewhiten((t, mag, err))
print(solution.summary())
print(solution.frequency, solution.frequency_error)
```

The extraction stops on a stated criterion (`solution.stop_reason`), reports 1-sigma
uncertainties on every frequency, amplitude and phase, and flags components that are
combinations of stronger ones. See {doc}`guide/prewhitening`.

## From the command line

The same machinery is a CLI. Given a file `star.csv`:

```bash
cuperiod run star.csv --method GLS,BLS --n-best 10
```

See {doc}`guide/cli`.

---

**Next steps**

- 📓 The [example notebook](https://github.com/tjayasinghe/cuPeriod/blob/main/examples/cuperiod_tour.ipynb)
  — a guided tour over real light curves (Cepheid, RR Lyrae, eclipsing binary, Mira, and a
  *Kepler* exoplanet) with every periodogram and phase-fold.
- {doc}`guide/methods` — pick the right method for your signal.
- {doc}`guide/results` — everything `best_periods` and `Periodogram` can do.
- {doc}`guide/batch` — scale to thousands or millions of light curves.
