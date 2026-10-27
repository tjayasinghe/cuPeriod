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

- **Calendar times** are converted: `datetime64` values (pandas datetimes — tz-aware
  ones taken in UTC — and pyarrow timestamps) become MJD days, `NaT` becomes `NaN`, and
  `timedelta64` / `TimeDelta` become days. astropy `Time` columns (`TimeSeries`,
  lightkurve, ECSV round trips, `(Time, flux)` tuples) keep their values in the
  jd / mjd / btjd / bkjd formats and are converted to MJD otherwise. Calendar values in
  a value or error column raise `TypeError`.
- **Units are not applied**: the number in a time column is taken as days, so an
  astropy `Quantity` or a QTable/FITS column in seconds must be converted first.
- **Missing values** — numpy masked arrays, astropy masked columns, pandas `NA` and
  pyarrow nulls — become `NaN` on every input route and are dropped by `finite()`;
  masked integer, boolean and short text columns load too.
- A table with a band/filter column read as a single light curve is **pooled** into one
  curve, with a warning pointing to {class}`~cuperiod.MultiBandLightCurve` /
  `ColumnMap(band=...)`; passing `columns=ColumnMap(band=...)` to
  {func}`~cuperiod.periodogram` or {func}`~cuperiod.to_input` gives a multi-band light
  curve directly (see {doc}`multiband`).
- A bare `(t, y)` / `(t, y, dy)` tuple is read as **magnitudes**. BLS and TLS warn when
  such a tuple looks like a normalised flux (median 0.5–1.5), because in magnitudes every
  dip would be a brightening; pass `domain="flux"`.

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
container keeps your input times verbatim. (One small exception: the fold methods' default
grids place the last sample on a phase-bin edge, so JD and MJD inputs can differ at the
O(1/N) level there.)
:::

## Loading from a file

{meth}`LightCurve.from_file <cuperiod.LightCurve.from_file>` reads CSV, ECSV, FITS, and
Parquet by extension; {meth}`~cuperiod.LightCurve.from_fits` targets a specific FITS HDU:

```python
lc = LightCurve.from_file("star.parquet")
lc = LightCurve.from_file("star.csv")
lc = LightCurve.from_fits("lightcurve.fits", hdu=1)
```

The originating path is recorded in `lc.meta["source"]`. Tab-separated (`.tsv`/`.tab`)
and other text tables (`.dat`/`.txt`, flavour guessed) read too, as do `.fts` and `.fz`
FITS files; the text and FITS formats may also be compressed (`star.fits.gz`,
`star.csv.bz2`, `.xz`). UTF-8 files that start with a byte-order mark — Excel's
*CSV UTF-8* — load with their first column name intact. Parquet categorical columns with
nulls read correctly, and a column that cannot be converted is named in the error
(`time column 'date' could not be converted to float64: ...`).

## Column auto-detection with `ColumnMap`

Light curves will always carry temporal and magnitude/flux measurements, with optional
magnitude/flux errors. cuPeriod detects the common spellings case-insensitively,
most-specific first, so a table with both `BJD_TDB` and `JD` picks the corrected time.
The detection lists are:

- **Time** — `bjd_tdb`, `bjd`, `hjd`, `btjd`, `bkjd`, `mjd`, `hmjd`, `midpointMjdTai`,
  `obsTime`, `jd`, `time`, `date`, `t`
- **Magnitude** — `mag`, `magnitude`, `vmag`/`gmag`/`rmag`/`bmag`/`imag`, `psfMag`,
  `MAG_0`, `m`
- **Flux** — `PDCSAP_FLUX`, `KSPSAP_FLUX`, `SAP_FLUX`, `psfFlux`, `flux`, `uJy`, `mJy`, …
  (corrected/detrended fluxes are preferred over raw)
- **Error** — resolved to match the value's measurement and domain: a flux value pairs
  with `flux_err`/`flux_error`/`psfFluxErr`/`duJy`, a magnitude value with
  `mag_err`/`magerr`/`dm`/`MER_0`, then the neutral `err`/`error`/`sigma`
- **Band** — `band`, `filter`, `filtercode`, `filterID`, `phot_filter`, `passband`, `fid`

The error is matched to the **same measurement** as the value, so a table that carries
both (e.g. ATLAS's `m`/`dm` and `uJy`/`duJy`) pairs `uJy` with `duJy`, never with the
magnitude error.

When detection isn't enough — ambiguous names, or a column you want to force — pin it with
{class}`~cuperiod.ColumnMap`:

```python
from cuperiod import ColumnMap

cmap = ColumnMap(time="HJD", value="Vmag", error="e_Vmag")
pg = cup.periodogram(df, "GLS", columns=cmap)
```

Any field left `None` is auto-detected; a non-`None` field is honored verbatim (and
raises {exc}`~cuperiod.ColumnResolutionError` if that column is absent). Only `time` and
`value` are required; `error` and `band` are optional. Columns may also be named by
position for tables whose labels are integers — `pd.read_csv(..., header=None)` loads
with `ColumnMap(time=0, value=1, error=2)`.

## Supported surveys

The detection lists cover the standard light-curve products of the major time-domain
surveys, so a downloaded table usually needs **no column hints** — `cup.periodogram(df,
"GLS")` just works:

```{list-table}
:header-rows: 1
:widths: 18 16 22 22 16

* - Survey
  - Time
  - Value
  - Error
  - Band
* - ASAS-SN (Sky Patrol)
  - `jd` / `hjd`
  - `mag` / `flux`
  - `mag_err` / `flux_err`
  - `phot_filter`
* - ASAS-3
  - `HJD`
  - `MAG_0`..`MAG_4`
  - `MER_0`..`MER_4`
  - —
* - ATLAS
  - `MJD`
  - `uJy` (or `m`)
  - `duJy` (or `dm`)
  - `F` ¹
* - CRTS / CSS
  - `MJD`
  - `Mag`
  - `Magerr`
  - —
* - ZTF
  - `mjd` / `HMJD`
  - `mag`
  - `magerr`
  - `filtercode` / `filterID`
* - Pan-STARRS (PS1)
  - `obsTime`
  - `psfFlux`
  - `psfFluxErr`
  - `filterID`
* - LSST / Rubin
  - `midpointMjdTai` ²
  - `psfFlux` ²
  - `psfFluxErr` ²
  - `band`
* - TESS (SPOC / QLP)
  - `TIME`
  - `PDCSAP_FLUX` / `KSPSAP_FLUX`
  - `*_FLUX_ERR`
  - —
* - Kepler (SPOC)
  - `TIME`
  - `PDCSAP_FLUX`
  - `PDCSAP_FLUX_ERR`
  - —
* - Gaia (DR3 epoch)
  - `time`
  - `flux` (or `mag`)
  - `flux_error`
  - `band`
* - MACHO
  - `time` / `mjd`
  - `mag` (red/blue)
  - `error`
  - red / blue ³
* - OGLE
  - `HJD`
  - magnitude
  - error
  - — (per file)
```

¹ ATLAS's single-character `F` filter isn't auto-detected (to avoid clashing with a flux
`f`); pass `ColumnMap(band="F")` if you need it for multi-band. The value defaults to the
flux (`uJy`); `domain="magnitude"` picks `m`/`dm` instead (as does pinning
`ColumnMap(value="m")`). ² LSST DP0.2 uses the older
`midPointTai` / `psFlux` / `psFluxErr` / `filterName`, which are also detected. ³ MACHO's
raw per-column header spellings are not consistently documented; if a file uses
non-standard names, pass an explicit {class}`~cuperiod.ColumnMap`.

:::{note}
**Headerless files** — OGLE `.dat` photometry (and some raw ASAS-3 dumps) have no column
names to detect. Read them positionally instead — with `pd.read_csv(path,
header=None, sep=r"\s+")` and `ColumnMap(time=0, value=1, error=2)`, or with
[`numpy.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html)
(the GUI's import dialog recognises such files on its own):

```python
import numpy as np
from cuperiod import LightCurve

hjd, mag, err = np.loadtxt("OGLE-LMC-CEP-0001.dat", usecols=(0, 1, 2), unpack=True)
pg = cup.periodogram(LightCurve.from_arrays(hjd, mag, err), "GLS")
```
:::

## Magnitude vs flux: the `Domain`

A value column is either a **magnitude** or a **flux**. cuPeriod infers which from the
column name (`flux`/`fnu`/a bare `f` → flux, `mag`/… → magnitude), defaulting to
magnitude, and you can always state it explicitly:

```python
from cuperiod import Domain

pg = cup.periodogram(df, "GLS", domain=Domain.FLUX)     # or domain="flux"
```

The string form is exactly equivalent and ignores case and surrounding spaces
(`" Flux "`); an unknown string raises `ValueError`. An explicit `domain=` also chooses
the auto-detected column: on a table that carries both kinds (ASAS-SN, Gaia, ATLAS),
`domain="magnitude"` reads the magnitude column and its error rather than relabelling
the flux, and a `UserWarning` fires when the table has no column of the requested
domain. For multi-band input `domain=` applies to every band.

Why it matters: **box/transit methods (BLS, TLS) work in flux**, where an eclipse or
transit is a *dip*. If you hand them magnitudes, cuPeriod converts to flux automatically
(`flux = 10 ** (-0.4 * mag)`, with error propagation), so you don't have to. The Fourier
and fold methods (GLS, PDM, CE, string-length, MHAOV, SuperSmoother) are domain-agnostic.

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

An error column of placeholders — all `0`, `-1` or `NaN` — would empty the curve, so
every method (and the period-change and pre-whitening tools) reports it as the cause
instead of "no usable time baseline". {meth}`LightCurve.unusable_reason()
<cuperiod.LightCurve.unusable_reason>` returns that reason (or `None`), and
`cuperiod.core.raise_if_unusable(lc, what)` raises it as `InsufficientDataError`.

---

Next: {doc}`methods` — choosing the right statistic for your signal.
