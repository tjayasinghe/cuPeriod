# cuPeriod validation & benchmark suite

Reproducible validation and performance benchmarks for cuPeriod's period-search
methods on **real survey data**, comparing the CPU and GPU backends against each
other and against established third-party implementations.

The rendered results are in **[REPORT.md](REPORT.md)** with figures in `figures/`.

The suite covers seven methods. SuperSmoother, added in v1.2, is not in it yet —
it is pinned against the reference `supersmoother` package and `gatspy` in the
unit tests (`tests/test_supersmoother.py`) instead.

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
| Multi-band recovery | GLS | single-band GLS | does a joint fit beat per-band searching? |

The performance benchmark also times the **portable `torch` backend** (`backend="torch"`;
the resolved device is shown in the `torch_backend` column) for the ported methods — GLS
and BLS — alongside the CPU and CUDA paths, so the cross-vendor path (AMD/Intel/Mac/CPU) is
tracked. A backend absent on the host (no CUDA GPU, or no torch) is recorded blank rather
than failing the sweep.

## Data

* **`dataset/light_curves.parquet`** — 126 real ASAS-SN g-band light curves
  spanning six variability classes (eclipsing binaries, RR Lyrae, Cepheids,
  δ Scuti, long-period and rotational variables), each with its VSX (AAVSO
  Variable Star Index) literature period. One row per star:
  `asas_sn_id, band, vsx_type, broad_class, vsx_period, n_det, baseline, jd[],
  mag[], mag_err[]`. It combines a 72-star curated core — chosen to have a
  high-confidence period that ASAS-SN photometry independently confirms
  against the VSX literature value to within 1% — with a 54-star extension
  (`dataset/download_extension.py`, re-runnable) that deliberately adds harder,
  less-curated classes (spot-evolving rotators, wandering semiregular/Mira
  periods) for a more realistic recovery-rate estimate; see REPORT.md §3 for
  the core-vs-extension breakdown. This file is self-contained — §1, §2 and §4
  of the report need no external data or network access.
* **Kepler** confirmed KOIs for TLS, from the Mendeley
  *Dataset_Machine_Learning_Exoplanets_2024* (`wctcv34962`); raw PDCSAP flux is
  fetched from MAST with `lightkurve` into `data/` at run time.

## Multi-band recovery (`multiband_recovery.py`)

Single-band vs. joint multi-band period recovery under a Rubin/LSST-like cadence. This
one is **simulated**, not real data: RRab proxies (fundamental plus a phase-locked 0.35
second harmonic, P ~ U(0.35, 0.9) d) observed over a 3-year span, epochs landing on random
nights with the WFD per-band share (u 6%, g 9%, r 26%, i 26%, z 17%, y 16%) and 0.20 mag
per-point noise — about an *r* ≈ 23 halo RR Lyrae in single visits. The *total* number of
epochs across all six bands is swept. Recovery means the periodogram's top period is
within 1% of the truth, with no harmonic credit.

```bash
python multiband_recovery.py --n-stars 300     # -> results/multiband_recovery.parquet
```

Recovery fraction, 300 stars per cell:

| strategy | 30 epochs | 60 epochs | 120 epochs |
|---|---|---|---|
| best single band (r) | 0.0% | 37.3% | 97.3% |
| any single band | 0.0% | 49.3% | 99.0% |
| multiband perband (0,1) | 17.7% | 97.0% | 100.0% |
| multiband flex (1,1) | 20.0% | 97.0% | 100.0% |
| multiband offsets (1,0) | 81.7% | 99.7% | 100.0% |

*any single band* counts a star as recovered if **any** of the six per-band searches lands
on the right period — an optimistic upper bound, since in practice you don't know which
band was right. The cadence is deliberately simple (random nights, no rolling cadence, no
lunation weighting), so the result to read is the ordering rather than the absolute rates:
sharing the phase across bands is what buys sparse-cadence recovery, consistent with
Rubin's own alert-production study and VanderPlas & Ivezić (2015). That is why `offsets`
is cuPeriod's default multi-band model.

Options: `--n-stars`, `--noise` (mag per point), `--backend`. It needs no external data or
network access.

## Environments

Two venvs, because `transitleastsquares` pins an old numba:

* main GPU venv (`.venv`) — cuPeriod + astropy + PyAstronomy (`pip install -e ".[gpu]"`).
* reference venv (`.venv-ref`) — `lightkurve`, `transitleastsquares`.

## Run

```bash
python validate_periodograms.py            # -> results/validation_metrics.parquet   (GPU venv)
python injection_recovery.py               # -> results/injection_recovery.parquet    (GPU venv)
python multiband_recovery.py --n-stars 300 # -> results/multiband_recovery.parquet    (GPU venv)
python bls_numba_parity.py                 # -> results/bls_numba_parity.parquet      (GPU venv)
python benchmark.py                        # -> results/bench_*.parquet               (GPU venv)
../.venv-ref/Scripts/python tls_download_ref.py   # -> data/, tls_reference_results.csv
python tls_cuperiod.py                     # -> results/tls_results.parquet           (GPU venv)
python make_report.py                      # -> figures/*.png, REPORT.md
```

`dataset/light_curves.parquet` is committed, so the validation and benchmark
steps reproduce out of the box; only the TLS section downloads Kepler data.
