# Examples

## [`cuperiod_tour.ipynb`](cuperiod_tour.ipynb) — a guided tour

A hands-on walkthrough of cuPeriod on **real light curves**. For each kind of object it
loads the data, runs the appropriate periodogram, reads the peak, and phase-folds to
reveal the signal — three pictures per star (raw → periodogram → phased).

| Object | Method | What it teaches |
| --- | --- | --- |
| Classical Cepheid | **GLS** | the basics: raw → periodogram → phased |
| RR Lyrae | **MHAOV** | sharp, multiharmonic pulsations |
| Eclipsing binary | **BLS** (+ the GLS *P*/2 trap) | choosing the right method |
| Long-period variable (Mira) | **PDM** | non-sinusoidal folds, long baselines |
| Exoplanet (Kepler KIC 7532973) | **TLS** | a transit matched filter |
| — | several at once | comparing methods, reading the N-best peaks |

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
