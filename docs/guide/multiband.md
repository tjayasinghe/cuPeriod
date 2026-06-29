# Multi-band light curves

When you have the **same star observed in several filters** (e.g. *g* and *r*), you can
model all bands jointly instead of running each separately. A joint model shares the period
across bands while letting each band have its own amplitude and offset, which boosts
sensitivity when no single band is well-sampled.

Multi-band is supported by **GLS, BLS, and MHAOV**. (The other methods take a single band.)

## Building a `MultiBandLightCurve`

From a `{band: LightCurve}` mapping:

```python
from cuperiod import LightCurve, MultiBandLightCurve

lc_g = LightCurve.from_arrays(t_g, mag_g, err_g)
lc_r = LightCurve.from_arrays(t_r, mag_r, err_r)

mb = MultiBandLightCurve.from_light_curves({"g": lc_g, "r": lc_r})
print(mb.band_names, mb.n_bands)       # ('g', 'r'), 2
```

Or split one **long-format** table on a band/filter column:

```python
mb = MultiBandLightCurve.from_dataframe(df, band_column="filter")
```

The band column is auto-detected (`band`, `filter`, `passband`, `fid`, …) when you don't
name it, and the time/value/error columns resolve exactly as for a single
{doc}`light curve <light-curves>` (with {class}`~cuperiod.ColumnMap` if needed).

## Running a multi-band periodogram

Pass the {class}`~cuperiod.MultiBandLightCurve` to {func}`~cuperiod.periodogram` like any
other input:

```python
pg = cup.periodogram(mb, "GLS")        # VanderPlas & Ivezić shared-phase model
print(pg.best_period())
```

The result is an ordinary {class}`~cuperiod.Periodogram`, so peak finding, the raw
spectrum, and serialization all work as in {doc}`results`.

```python
pg = cup.periodogram(mb, "MHAOV")      # pooled F-statistic across bands
pg = cup.periodogram(mb, "BLS")        # joint box search across bands
```

Asking a single-band method for a multi-band run raises a clear error, so you'll know
immediately if a method doesn't support it.

## A note on the default grid

For multi-band input the default grid is built from the **stacked** baseline of all bands
(their combined time span), so the frequency resolution reflects all your data. You can
still override it with `grid=` ({doc}`tuning`).

---

Next: {doc}`batch` — scaling to many light curves.
