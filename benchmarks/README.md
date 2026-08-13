# cuPeriod validation & benchmark suite

Reproducible validation and performance benchmarks for cuPeriod's period-search
methods on **real survey data**, comparing the CPU and GPU backends against each
other and against established third-party implementations.

The rendered results are in **[REPORT.md](REPORT.md)** with figures in `figures/`.

The single-band validation and benchmark sections cover seven methods.
SuperSmoother, added in v1.2, is pinned against the reference `supersmoother`
package and `gatspy` in the unit tests (`tests/test_supersmoother.py`), and is
exercised on real data in the multi-band validation below.

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
| Multi-band recovery (simulated) | GLS | single-band GLS | does a joint fit beat per-band searching? |
| Multi-band recovery (real) | all 7 multi-band methods + 3 GLS models | Sesar 2010 literature periods | recover known periods on real ugriz data? |

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
  the core-vs-extension breakdown. This file is self-contained — §1, §2 and §5
  of the report need no external data or network access.
* **`dataset/s82_rrlyrae.parquet`** — 100 real SDSS Stripe 82 RR Lyrae *ugriz*
  light curves (80 RRab, 20 RRc; ~55 epochs per band over a ~3200 d baseline)
  with the literature periods of Sesar et al. 2010 (ApJ 708, 717), the canonical
  real multi-band validation set — VanderPlas & Ivezić (2015) developed the
  multiband periodogram on these stars. One row per star and band:
  `sesar_id, rrl_type, p_sesar, band, n_det, baseline, mjd[], mag[], mag_err[]`.
  Built by `dataset/download_s82_rrlyrae.py` (re-runnable) from the astroML-data
  GitHub mirror of the paper's tables — the original MPIA host is dead — keeping
  the first 100 stars by ascending Sesar ID, a deterministic cut with no quality
  selection. At ~390 KB it is committed, so §4 of the report runs offline.
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

## Real multi-band validation (`multiband_real.py`)

The same question as above, asked of **real** photometry with known answers. Every star in
`dataset/s82_rrlyrae.parquet` gets one identical blind search — periods 0.15–1.2 d on a
uniform grid at 5 samples per Rayleigh width of its ~3200 d baseline (~97,000 trial
frequencies), every method at its default settings (BLS builds its native duration grid
inside the same window). Scoring follows the suite: **strict** = the top period is within 1%
of Sesar's, with no harmonic credit; **harmonic-aware** = it matches up to a small-integer
harmonic ratio within 2%.

```bash
python multiband_real.py                   # -> results/multiband_real.parquet
```

100 stars, CPU pass (numba/finufft tiers):

| model | strict | harmonic-aware | median CPU s/star |
|---|---|---|---|
| GLS `offsets` (1,0) | 76% | 80% | 0.067 |
| GLS `perband` (0,1) | 78% | 81% | 0.107 |
| GLS `flex` (1,1) | 78% | 81% | 0.348 |
| PDM | **93%** | 94% | 0.011 |
| CE | 85% | 90% | 0.024 |
| String-Length | **93%** | **97%** | 0.039 |
| MHAOV | 83% | 84% | 0.539 |
| SuperSmoother | 85% | 96% | 0.293 |
| BLS | 22% | 34% | 3.044 |

Single-band GLS on one filter at a time is the baseline the joint methods have to beat: 72–78%
strict per band (*z* worst, *r* best), and 92% if you count a star as recovered when **any** of
the five bands lands on the right period — the same optimistic upper bound as above, since in
practice you don't know which band was right. The median fractional period error over strict
hits is ~1e-05 for every model, so a hit is a hit at the grid's resolution.

The read: on curves this well sampled (~280 points across five bands) the pooled fold
statistics lead. PDM and String-Length reach 93% strict, String-Length 97% harmonic-aware —
a dense fold exploits the full non-sinusoidal RRab shape, while the single-harmonic GLS models
stay alias-limited. Those three GLS models are indistinguishable here (76–78%) and no better
than the best single band (78%), which is the *opposite* regime from the simulated sparse
cadence above, where `offsets` recovers 82% at 30 total epochs and the flexible models ≤20%:
dense per-band data reward shape, sparse data reward parsimony. (The fold methods were not run
at sparse cadence, so this says nothing about how they would do there.) SuperSmoother's
strict-vs-harmonic gap, 85% → 96%, is exactly the documented integer-multiples family — of 21
fold-family harmonic-but-not-strict picks, 11 sit at exactly 2P and 5 at 3P — and it shows up
by subtype: 55% strict on the near-sinusoidal RRc against 92.5% on RRab, since a fold at 2P
stays coherent. The documented recipe applies: take the shortest member of a near-tied family,
or let `cuperiod.alias_diagnostics` arbitrate. Of the 163 non-harmonic misses across all
models, 54% sit on the ±1 or ±2 cycle/day loci — the ground-based window function, not noise.
BLS's 22% is expected: it is transit-shaped by design, included for completeness rather than
recommended for RR Lyrae.

A GPU pass runs alongside the scored CPU pass to record agreement and timing. The step needs
no network access — the light curves are committed.

## Environments

Two venvs, because `transitleastsquares` pins an old numba:

* main GPU venv (`.venv`) — cuPeriod + astropy + PyAstronomy (`pip install -e ".[gpu]"`).
* reference venv (`.venv-ref`) — `lightkurve`, `transitleastsquares`.

## Run

```bash
python validate_periodograms.py            # -> results/validation_metrics.parquet   (GPU venv)
python injection_recovery.py               # -> results/injection_recovery.parquet    (GPU venv)
python multiband_recovery.py --n-stars 300 # -> results/multiband_recovery.parquet    (GPU venv)
python multiband_real.py                   # -> results/multiband_real.parquet        (GPU venv)
python bls_numba_parity.py                 # -> results/bls_numba_parity.parquet      (GPU venv)
python benchmark.py                        # -> results/bench_*.parquet               (GPU venv)
../.venv-ref/Scripts/python tls_download_ref.py   # -> data/, tls_reference_results.csv
python tls_cuperiod.py                     # -> results/tls_results.parquet           (GPU venv)
python make_report.py                      # -> figures/*.png, REPORT.md
```

`dataset/light_curves.parquet` and `dataset/s82_rrlyrae.parquet` are committed,
so the validation, multi-band and benchmark steps reproduce out of the box; only
the TLS section downloads Kepler data.
