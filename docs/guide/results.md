# Reading results

A run returns a {class}`~cuperiod.Periodogram` (one method) or a
{class}`~cuperiod.MultiResult` (several). This page covers everything you can do with
them — peak finding, the raw spectrum, per-method extras, and serialization.

## The `Periodogram`

A {class}`~cuperiod.Periodogram` is the raw spectrum plus metadata, stored in
**ascending frequency** order (`period = 1 / frequency`):

```python
pg = cup.periodogram(lc, "GLS")

pg.method           # "GLS"
pg.backend          # the concrete backend that ran, e.g. "finufft" or "cufinufft"
pg.frequency        # ndarray, cycles/day, ascending
pg.period           # ndarray, days
pg.power            # ndarray, the method's statistic
pg.objective_sense  # "max" or "min"
pg.n_samples        # number of finite points used
pg.baseline         # time span in days (sets the Rayleigh resolution 1/baseline)
pg.size             # number of grid samples
```

## Finding the best periods

{meth}`~cuperiod.Periodogram.best_periods` is the method you'll use most. It finds local
extrema of the statistic, ranks them by significance (respecting `objective_sense`), and
enforces a minimum separation so you don't get the same peak repeatedly:

```python
peaks = pg.best_periods(10)
best = pg.best_period()          # shorthand for the single best period (a float, days)
```

Parameters:

- **`n`** (default 10) — maximum number of peaks to return.
- **`min_separation_rayleigh`** (default 3.0) — minimum spacing between peaks, in Rayleigh
  widths (`1/baseline`). Raise it to thin out closely-spaced peaks; lower it to resolve
  them.
- **`alias_diverse`** (default `False`) — when `True`, skip candidates that are aliases or
  harmonics of a peak already chosen. **Recommended for box searches (BLS/TLS)**, where the
  same transit shows up at 2×/½× the period.
- **`harmonic_max`** (default 8) — the highest harmonic order considered when
  `alias_diverse` is set.

```python
# Box search: get physically distinct candidates, not harmonics of one transit.
for peak in pg.best_periods(5, alias_diverse=True):
    print(peak.rank, peak.period)
```

## The `Peak`

Each {class}`~cuperiod.Peak` is a small immutable record:

```python
peak.period       # days
peak.frequency    # cycles/day
peak.power        # the statistic at this peak
peak.rank         # 1-based significance rank
peak.extra        # dict of method-specific scalars
peak.to_dict()    # flatten to a plain dict (scalars + extras)
```

### Per-method `extra` scalars

The `extra` dict carries quantities a method computes at each peak:

| Method | `extra` keys |
| --- | --- |
| GLS | `fap` (false-alarm probability) |
| BLS | `depth`, `duration`, `transit_time`, `depth_snr`, `sde` |
| TLS | transit shape scalars (depth, duration, …) |
| others | method-specific where applicable |

```python
top = pg.best_periods(1)[0]
print(top.extra.get("fap"))          # GLS
# print(top.extra["depth"], top.extra["duration"])   # BLS
```

## The raw spectrum

The full arrays are attributes — use them directly for plotting or custom analysis:

```python
import matplotlib.pyplot as plt
plt.plot(pg.period, pg.power)
plt.xscale("log")
```

For a compact, plot-ready version, {meth}`~cuperiod.Periodogram.downsample` returns a
**peak-preserving** downsample (so tall peaks survive the thinning):

```python
freq, power = pg.downsample(2000)    # float32 arrays of length ~2000
```

## Serializing

{meth}`~cuperiod.Periodogram.to_dict` produces a JSON-friendly dict — peaks always, the
raw spectrum optionally:

```python
import json
payload = pg.to_dict(n_best=10, include_raw=False)
json.dumps(payload, indent=2)
```

`include_raw=True` adds a downsampled `frequency`/`power` to the payload.

## Several methods: `MultiResult`

When you pass a list of methods, you get a {class}`~cuperiod.MultiResult` — a mapping from
method name to {class}`~cuperiod.Periodogram`:

```python
res = cup.periodogram(lc, ["GLS", "BLS"])

res["GLS"]                       # the GLS Periodogram (keys are case-insensitive)
list(res.keys())                 # ['GLS', 'BLS']
res.best_periods(5)              # {'GLS': [...], 'BLS': [...]}
res["BLS"].best_periods(5, alias_diverse=True)
res.to_dict(n_best=10)           # every method serialized
```

---

Next: {doc}`backends` — choosing where the computation runs.
