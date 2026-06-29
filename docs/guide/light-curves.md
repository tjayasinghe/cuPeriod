# Light curves & inputs

Every periodogram starts from a light curve: observation **times**, **brightnesses**,
and optional **errors**. cuPeriod accepts that data in whatever form you already have it
and normalizes it internally, so you rarely have to reshape anything by hand.

## Accepted inputs

{func}`cuperiod.periodogram` (and {func}`cuperiod.to_input`, which it calls) accept:

| Input | Example |
| --- | --- |
| A `(time, value)` or `(time, value, error)` tuple of arrays | `cup.periodogram((t, mag, err), "GLS")` |
| A dict of arrays | `cup.periodogram({"hjd": t, "mag": m, "e_mag": e}, "GLS")` |
| A pandas `DataFrame` | `cup.periodogram(df, "GLS")` |
| An astropy `Table` or pyarrow `Table` | `cup.periodogram(table, "GLS")` |
| A {class}`~cuperiod.LightCurve` you built yourself | `cup.periodogram(lc, "GLS")` |
| A `{band: LightCurve}` mapping (multi-band) | see {doc}`multiband` |

Times are in **days** throughout cuPeriod (JD/HJD/BJD/MJD all work — periodograms are
invariant to the time origin). Values may be **magnitude or flux**.

## Building a `LightCurve` directly

For full control, construct a {class}`~cuperiod.LightCurve`:

```python
from cuperiod import LightCurve

lc = LightCurve.from_arrays(time, mag, err)          # error is optional
print(lc.n, lc.baseline)                             # point count, time span (days)
```

Useful attributes and transforms:

- `lc.n` — number of points; `lc.baseline` — `max(t) - min(t)` in days.
- `lc.finite()` — a copy with non-finite points (and non-positive errors) removed.
- `lc.as_flux()` / `lc.as_magnitude()` / `lc.in_domain(...)` — domain conversion (below).

:::{note}
cuPeriod does **not** shift your time origin. Periodograms are invariant to it, and each
method references times to their own minimum internally for numerical stability, so the
container keeps your input times verbatim.
:::

## Loading from a file

{meth}`LightCurve.from_file <cuperiod.LightCurve.from_file>` reads CSV, ECSV, FITS, and
Parquet by extension; {meth}`~cuperiod.LightCurve.from_fits` targets a specific FITS HDU:

```python
lc = LightCurve.from_file("star.parquet")
lc = LightCurve.from_file("star.csv")
lc = LightCurve.from_fits("lightcurve.fits", hdu=1)
```

The originating path is recorded in `lc.meta["source"]`.

## Column auto-detection with `ColumnMap`

Astronomers' tables carry the same three quantities under wildly different names. cuPeriod
detects the common spellings case-insensitively, most-specific first, so a table with both
`BJD_TDB` and `JD` picks the corrected time. The detection lists are:

- **Time** — `bjd_tdb`, `bjd`, `hjd`, `btjd`, `mjd`, `jd`, `time`, `date`, `t`
- **Magnitude** — `mag`, `magnitude`, `vmag`, `gmag`, `rmag`, `phot_mag`, `m`
- **Flux** — `pdcsap_flux`, `sap_flux`, `norm_flux`, `rel_flux`, `flux`, `fnu`, `f`
- **Error** — `mag_err`, `e_mag`, `dmag`, `flux_err`, `err`, `sigma`, … (mag & flux spellings)
- **Band** — `band`, `filter`, `phot_filter`, `passband`, `fid`

When detection isn't enough — ambiguous names, or a column you want to force — pin it with
{class}`~cuperiod.ColumnMap`:

```python
from cuperiod import ColumnMap

cmap = ColumnMap(time="HJD", value="Vmag", error="e_Vmag")
pg = cup.periodogram(df, "GLS", columns=cmap)
```

Any field left `None` is auto-detected; a non-`None` field is honored verbatim (and
raises {exc}`~cuperiod.ColumnResolutionError` if that column is absent). Only `time` and
`value` are required; `error` and `band` are optional.

## Magnitude vs flux: the `Domain`

A value column is either a **magnitude** or a **flux**. cuPeriod infers which from the
column name (`flux`/`fnu` → flux, `mag`/… → magnitude), defaulting to magnitude, and you
can always state it explicitly:

```python
from cuperiod import Domain

pg = cup.periodogram(df, "GLS", domain=Domain.FLUX)     # or domain="flux"
```

Why it matters: **box/transit methods (BLS, TLS) work in flux**, where an eclipse or
transit is a *dip*. If you hand them magnitudes, cuPeriod converts to flux automatically
(`flux = 10 ** (-0.4 * mag)`, with error propagation), so you don't have to. The Fourier
and fold methods (GLS, PDM, CE, string-length, MHAOV) are domain-agnostic.

You can convert by hand too:

```python
lc_flux = lc.as_flux()             # no-op if already flux
lc_mag = lc.as_magnitude()         # no-op if already magnitude
```

## Cleaning

Periodograms need finite inputs. `finite()` drops NaN/inf times and values, and — when
errors are present — any point whose error is non-finite or `<= 0` (an unusable weight):

```python
lc = LightCurve.from_file("messy.csv").finite()
```

Methods also enforce a minimum point count (`min_detections` in their settings) and raise
{exc}`~cuperiod.InsufficientDataError` on too-sparse curves.

---

Next: {doc}`methods` — choosing the right statistic for your signal.
