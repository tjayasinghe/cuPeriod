# Examples

## [`cuperiod_tour.ipynb`](cuperiod_tour.ipynb) — a guided tour

A hands-on walkthrough of cuPeriod on **real light curves** — plus one deliberately
simulated star at the end, because the closing lesson is a search *failing*, and for that
you have to know the true period. For each kind of object it loads the data, runs the
appropriate periodogram, reads the peak, and phase-folds to reveal the signal — three
pictures per star (raw → periodogram → phased).

| Object | Method | What it teaches |
| --- | --- | --- |
| Classical Cepheid | **GLS** | the basics: raw → periodogram → phased |
| RR Lyrae | **MHAOV** | sharp, multiharmonic pulsations |
| Eclipsing binary | **BLS** (+ the GLS *P*/2 trap) | choosing the right method |
| Long-period variable (Mira) | **PDM** | non-sinusoidal folds, long baselines |
| Exoplanet (Kepler KIC 7532973) | **TLS** | a transit matched filter |
| — | several at once | comparing methods, reading the N-best peaks |
| Sparse six-band star (synthetic survey cadence) | **multi-band GLS** + FAP + alias diagnostics | why joint fitting wins in the Rubin era |

### Run it

```bash
pip install cuperiod matplotlib pandas pyarrow   # add "cuperiod[gpu]" to use a GPU
jupyter lab cuperiod_tour.ipynb
```

The notebook is **fully self-contained and offline** — the `data/` folder holds the
bundled light curves, so no download is needed. With an NVIDIA GPU the same code runs on
the GPU automatically (`backend="auto"`).

### Data provenance

- `data/asassn_examples.parquet` — 6 public [ASAS-SN](https://asas-sn.osu.edu/) *g*-band
  light curves (one per variability class), each with its VSX literature period.
- `data/kepler_KIC7532973.csv` — *Kepler* PDCSAP flux for a confirmed hot-Jupiter host,
  fetched once with [lightkurve](https://docs.lightkurve.org/).
- The sparse six-band star is **not** a bundled file: it is simulated inline with numpy
  from a fixed seed, on the Rubin/LSST-like cadence of
  [`benchmarks/multiband_recovery.py`](../benchmarks/multiband_recovery.py).

## The desktop GUI (`cuperiod-gui`)

An interactive periodogram explorer built on PySide6 + pyqtgraph: run any method with all
of its options, explore the full-resolution spectrum, and watch the phased light curve
update live as you pick peaks — single curves or a whole folder in batch mode.

```bash
pip install "cuperiod[gui]"     # add [gpu] or [torch] for accelerated backends
cuperiod-gui                    # or:  python -m cuperiod.gui
```

### A two-minute smoke test

1. **Load demo → *Kepler KIC 7532973*.** Pick method **BLS**, press **Compute**. The
   spectrum draws (peak-preserving, smooth to zoom); the phased panel shows the transit as
   a **dip** at the best period.
2. **Slide the teal line** across the spectrum (or click a row in the **Peaks** dock) and
   watch the phased curve re-fold live. Toggle the spectrum's **x-axis** (frequency ↔
   period) and **log** scales, and the **2 cycles** checkbox on the phased panel.
3. **Load demo → *Synthetic multiband*.** With **GLS** the phased panel overlays all three
   bands (colour-coded, legend); the **Band** selector chooses *combined* or one band.
4. **Load demo → *Batch: browse all demo sources*.** Scroll the **Sources** dock (arrow
   keys or Prev/Next); each source computes on demand and revisits are instant (cached).
5. **Toggle the theme** (dark ↔ light, remembered next launch). With `[gpu]`/`[torch]`
   installed, the info bar names the backend and device actually used.
