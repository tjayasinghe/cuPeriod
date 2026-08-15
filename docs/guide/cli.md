# Command line

Installing cuPeriod adds a `cuperiod` command. Every subcommand is a thin wrapper over the
Python API, so the CLI and library share one code path and give identical results. Run
`cuperiod --help` (or `cuperiod <command> --help`) for the full option list.

```text
cuperiod run              one light curve, one or more methods → best periods
cuperiod batch            many light curves with CPU or GPU workers → Parquet/CSV
cuperiod prewhiten        extract a pulsator's frequency solution
cuperiod batch-prewhiten  the same over many light curves → Parquet/CSV
cuperiod methods          list registered methods and their backends
cuperiod gpu-info         show the CUDA GPU and suggested worker counts
cuperiod doctor           diagnose backends, torch devices, and precision
cuperiod grid-info        show a method's trial grid for a light curve (no compute)
```

:::{tip}
Prefer to explore interactively? `cuperiod-gui` opens a desktop periodogram explorer over
the same API — see {doc}`gui`.
:::

## `run` — a single light curve

```bash
cuperiod run star.csv --method GLS,BLS --n-best 10
```

Reads a CSV/ECSV/FITS/Parquet file, computes the method(s), and prints the ranked best
periods for each.

```{list-table}
:header-rows: 1
:widths: 28 72

* - Option
  - Meaning
* - `--method`, `-m`
  - Comma-separated method names (default `GLS`).
* - `--backend`
  - `auto` | `cpu` | `gpu` | a concrete backend name.
* - `--time` / `--value` / `--error`
  - Override column names (otherwise auto-detected).
* - `--band`
  - Band/filter column of a long-format file. Passing it runs a **joint** multi-band fit;
    without it the file is read as a single-band light curve. See {doc}`multiband`.
* - `--domain`
  - `magnitude` | `flux`.
* - `--n-best`
  - Number of peaks to report (default 10).
* - `--out`
  - Write the results as JSON to this path.
* - `--save-periodogram`
  - Write the raw spectra to this `.npz`.
```

Example with explicit columns and JSON output:

```bash
cuperiod run star.fits -m BLS --time BJD --value flux --error flux_err \
    --domain flux --out result.json --save-periodogram spectra.npz
```

## `batch` — many light curves

```bash
cuperiod batch "lcs/*.csv" --method GLS --device gpu --out results/
```

Runs over a glob, directory, or file of light curves with CPU or GPU workers and writes one
row per light curve. Key options:

```{list-table}
:header-rows: 1
:widths: 28 72

* - Option
  - Meaning
* - `--method`, `-m`
  - Comma-separated method names.
* - `--device`
  - `cpu` | `gpu`.
* - `--backend`
  - `auto` | `cpu` | `gpu` | concrete.
* - `--workers`
  - Worker count (omit for auto: all-but-one core on CPU, memory-sized on GPU).
* - `--out`
  - Output `.parquet` / `.csv` file, **or a directory** (resumable, one part per chunk).
* - `--n-best`
  - Peaks stored per light curve (default 10).
* - `--store-raw`
  - Also store the downsampled raw spectrum.
* - `--resume / --no-resume`
  - Skip chunks already written to a directory sink (default on).
```

A directory sink is resumable — re-running skips finished chunks. See {doc}`batch`.

## `prewhiten` — a pulsator's frequency solution

```bash
cuperiod prewhiten star.csv --snr 4.6 -n 30 --spacing --csv modes.csv
```

Runs the automated extraction loop of {doc}`prewhitening` and prints a header with the
reason the run stopped and the fit statistics, followed by the ranked components with
their uncertainties and signal-to-noise.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Meaning
* - `--max-frequencies`, `-n`
  - Cap on extracted components (default 30).
* - `--snr`
  - Breger signal-to-noise threshold (default 4.0).
* - `--stop`
  - Comma-separated criteria: `snr`, `fap`, `bic`, `amplitude`.
* - `--fmin` / `--fmax`
  - Search band in cycles/day (default `1/T` to `max(nyquist_factor × pseudo-Nyquist, 50)`).
* - `--uncertainty`
  - `covariance` | `analytic` | `bootstrap`.
* - `--combinations / --no-combinations`
  - Identify harmonics and combination frequencies (default on).
* - `--spacing`
  - Also search the independent modes for a g-mode period-spacing pattern.
* - `--backend`
  - `auto` | `cpu` | `gpu` | a concrete backend name.
* - `--out` / `--csv` / `--save-spectrum`
  - Write the full solution as JSON, the component table as CSV, and/or the
    amplitude spectra as `.npz` (data, residual, and the spectral window).
```

## `batch-prewhiten` — many pulsators

```bash
cuperiod batch-prewhiten "lcs/*.csv" --out modes.parquet --workers 8 -n 20
```

Takes the same inputs and worker options as `batch`, and the same extraction options as
`prewhiten`. The output has **one row per extracted component**, each carrying the
per-star summary (sample count, baseline, stop reason, fit statistics) so a single table
is self-describing.

## `methods` — what's available

```bash
cuperiod methods
```

Lists each registered method with its objective sense, multi-band support, and backends.

## `gpu-info` — device & worker sizing

```bash
cuperiod gpu-info
```

Shows the CUDA device (name, free/total memory, MPS status) and the suggested batch worker
count per GPU-capable method. If no GPU is present, it says so and exits cleanly.

## `doctor` — full environment diagnosis

```bash
cuperiod doctor
```

A one-stop "will the accelerated paths run here, and on what?" report: which backends are
installed, the NVIDIA CUDA fast paths, the portable **torch** backend and each device it
sees (CUDA/ROCm/MPS/XPU/CPU) with the precision it would use, and what `backend="auto"`
resolves to for every method. Reach for it first when a GPU isn't being picked up or you're
unsure which build of PyTorch you have.

## `grid-info` — inspect a grid

```bash
cuperiod grid-info star.csv --method GLS
```

Prints the default trial grid a method would use for a light curve — sample count and the
period/frequency range — **without** computing the spectrum. The fastest way to check your
period window before a long run.

---

That's the tour. For exact signatures and every option, see the {doc}`../api/index`, or
`cuperiod <command> --help`.
