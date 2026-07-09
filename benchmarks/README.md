# cuPeriod validation & benchmark suite

Reproducible validation and performance benchmarks for every cuPeriod method on
**real survey data**, comparing the CPU and GPU backends against each other and
against established third-party implementations.

The rendered results are in **[REPORT.md](REPORT.md)** with figures in `figures/`.

## What it checks

| | Methods | Reference | Question |
|---|---|---|---|
| CPU↔GPU parity | all 7 | — | do the two backends agree to round-off? |
| 1-to-1 vs reference | GLS, BLS | astropy `LombScargle` / `BoxLeastSquares` | bit-for-bit match a standard? |
| | PDM | PyAstronomy `pyPDM` | match an established package? |
| | CE, String-Length, MHAOV | independent textbook impl. | match the original paper? |
| | TLS | `transitleastsquares` | recover known Kepler periods? |
| Period recovery | all 7 | VSX literature period | find the real period? |
| Performance | all 7 | astropy / PyAstronomy | how much faster, and how does it scale? |

The performance benchmark also times the **portable `torch` backend** (`backend="torch"`;
the resolved device is shown in the `torch_backend` column) for the ported methods — GLS
and BLS — alongside the CPU and CUDA paths, so the cross-vendor path (AMD/Intel/Mac/CPU) is
tracked. A backend absent on the host (no CUDA GPU, or no torch) is recorded blank rather
than failing the sweep.

## Data

* **`dataset/light_curves.parquet`** — 72 real ASAS-SN g-band light curves
  spanning six variability classes (eclipsing binaries, RR Lyrae, Cepheids,
  δ Scuti, long-period and rotational variables), each with its well-established
  VSX (AAVSO Variable Star Index) literature period. One row per star:
  `asas_sn_id, band, vsx_type, broad_class, vsx_period, n_det, baseline, jd[],
  mag[], mag_err[]`. The stars were chosen to have a high-confidence period that
  ASAS-SN photometry independently confirms against the VSX literature value to
  within 1%. This file is self-contained — §1, §2 and §4 of the report need no
  external data or network access.
* **Kepler** confirmed KOIs for TLS, from the Mendeley
  *Dataset_Machine_Learning_Exoplanets_2024* (`wctcv34962`); raw PDCSAP flux is
  fetched from MAST with `lightkurve` into `data/` at run time.

## Environments

Two venvs, because `transitleastsquares` pins an old numba:

* main GPU venv (`.venv`) — cuPeriod + astropy + PyAstronomy (`pip install -e ".[gpu]"`).
* reference venv (`.venv-ref`) — `lightkurve`, `transitleastsquares`.

## Run

```bash
python validate_periodograms.py            # -> results/validation_metrics.parquet   (GPU venv)
python injection_recovery.py               # -> results/injection_recovery.parquet    (GPU venv)
python benchmark.py                        # -> results/bench_*.parquet               (GPU venv)
../.venv-ref/Scripts/python tls_download_ref.py   # -> data/, tls_reference_results.csv
python tls_cuperiod.py                     # -> results/tls_results.parquet           (GPU venv)
python make_report.py                      # -> figures/*.png, REPORT.md
```

`dataset/light_curves.parquet` is committed, so the validation and benchmark
steps reproduce out of the box; only the TLS section downloads Kepler data.
