# Command line

Installing cuPeriod adds a `cuperiod` command. Every subcommand is a thin wrapper over the
Python API, so the CLI and library share one code path and give identical results. Run
`cuperiod --help` (or `cuperiod <command> --help`) for the full option list.

```text
cuperiod run              one light curve, one or more methods → best periods
cuperiod batch            many light curves with CPU or GPU workers → Parquet/CSV
cuperiod prewhiten        extract a pulsator's frequency solution
cuperiod batch-prewhiten  the same over many light curves → Parquet/CSV
cuperiod pdot             search jointly over (period, dP/dt) and measure the drift
cuperiod pdot-batch       period changes for many light curves → up to three tables
cuperiod session          verify, script or replay a pre-whitening project (*.cupw.json)
cuperiod methods          list registered methods and their backends
cuperiod gpu-info         show the CUDA GPU and suggested worker counts
cuperiod doctor           diagnose backends, torch devices, and precision
cuperiod grid-info        show a method's trial grid for a light curve (no compute)
```

:::{tip}
Prefer to explore interactively? `cuperiod-gui` opens a desktop periodogram explorer over
the same API — see {doc}`gui`.
:::

**Errors and exit codes.** A user error — a missing file, an unknown method, backend or
device, a sink with an unsupported suffix, a non-positive `--period`, an empty `-m`,
`--n-best` below 1 (below 0 for `batch`), `--segments` below 4, a `pdot` search the
options cannot set up — is reported in one line, without a traceback, as a usage error
(exit status 2). Other cuPeriod errors, such as a light curve with too few points, print
`Error: …` and exit with status 1. The batch commands exit with status 1 when every light
curve that ran failed.

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
  - `auto` | `cpu` | `gpu` | a concrete backend name (default `auto`).
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
  - Comma-separated method names (default `GLS`).
* - `--device`
  - `cpu` | `gpu` (default `cpu`).
* - `--backend`
  - `auto` | `cpu` | `gpu` | concrete (default `auto`).
* - `--workers`
  - Worker count. Omit for auto: on the CPU, the CPUs this process may use (affinity
    mask, cgroup quota) minus one; on the GPU, sized by free memory. Every pool is
    capped at 61 processes on Windows.
* - `--chunk-size`
  - Light curves per task (omit for automatic: about four tasks per worker, at most
    256).
* - `--time` / `--value` / `--error` / `--domain`
  - As `run`.
* - `--out`
  - Output `.parquet` / `.csv` file, **or a directory** (resumable, one part per chunk).
* - `--n-best`
  - Peaks stored per light curve (default 10; `0` stores only `best_period`).
* - `--store-raw`
  - Also store the downsampled raw spectrum.
* - `--resume / --no-resume`
  - Skip the light curves (and methods) already in the sink (default on).
    `--no-resume` starts again; for a directory it first removes its `part-*.parquet`
    files.
```

Resuming is by light curve, not by chunk: re-running computes only the (light curve,
method) pairs the sink does not hold yet, and retries those that failed. Every failure
is listed in a CSV named after the sink — `out.errors.csv` beside `--out out.parquet`
(or `out.csv`), `_errors.csv` inside a directory sink — and the summary line shows the
worker count used. The errors file is rewritten after each batch that had inputs and
removed when that batch had no failures; a batch with no inputs leaves the sink and its
errors file alone. It is the command that writes it: the Python
{func}`~cuperiod.batch_periodograms` returns its failures in the summary's `errors` and
writes no such file. A rerun is safe when the sink sits inside the inputs' folder or
matches their glob: the batch commands never read their own sink, its errors file,
`.corrupt-*` copies or temporary files as light curves (nor, for a directory sink, its
part files and manifest; for `pdot-batch`, its `--sink`, `--oc-sink` and
`--profile-sink`). See {doc}`batch`.

## `prewhiten` — a pulsator's frequency solution

```bash
cuperiod prewhiten star.csv --snr 4.6 -n 30 --spacing --csv modes.csv
```

Runs the automated extraction loop of {doc}`prewhitening` and prints a header with the
reason the run stopped and the fit statistics, followed by the ranked components with
their uncertainties and signal-to-noise.

An option you do not give is not passed on: its {class}`~cuperiod.PreWhitenSettings`
field keeps the value of its `CUPERIOD_PREWHITEN_<FIELD>` environment variable when that
is set (see {doc}`tuning`; a list is written as JSON, e.g.
`CUPERIOD_PREWHITEN_STOP_CRITERIA='["snr","bic"]'`), else the default below — the same
rule as the library. An option you give always wins.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Meaning
* - `--max-frequencies`, `-n`
  - Cap on extracted components (default: `CUPERIOD_PREWHITEN_MAX_FREQUENCIES`, else 30).
* - `--snr`
  - Breger signal-to-noise threshold (default: `CUPERIOD_PREWHITEN_SNR_THRESHOLD`, else
    4.0).
* - `--stop`
  - Comma-separated criteria: `snr`, `fap`, `bic`, `amplitude` (default:
    `CUPERIOD_PREWHITEN_STOP_CRITERIA`, else `snr`).
* - `--fmin` / `--fmax`
  - Search band in cycles/day (default `1/T` to the automatic maximum: `0.5/cadence` on
    a regular cadence, else `max(nyquist_factor × pseudo-Nyquist, 50)`; see
    {doc}`prewhitening`).
* - `--uncertainty`
  - `covariance` | `analytic` | `bootstrap` (default:
    `CUPERIOD_PREWHITEN_UNCERTAINTY`, else `covariance`).
* - `--combinations / --no-combinations`
  - Identify harmonics and combination frequencies (default:
    `CUPERIOD_PREWHITEN_COMBINATIONS`, else on).
* - `--spacing`
  - Also search the independent modes for a g-mode period-spacing pattern.
* - `--backend`
  - `auto` | `cpu` | `gpu` | a concrete backend name (default: the settings' backend,
    i.e. `CUPERIOD_PREWHITEN_BACKEND`, else `auto`).
* - `--time` / `--value` / `--error` / `--domain`
  - As `run`. A project saved with `--save-project` records the domain, and the column
    mapping too when it references its data file.
* - `--out` / `--csv` / `--save-spectrum`
  - Write the full solution as JSON, the component table as CSV, and/or the
    amplitude spectra as `.npz` (data, residual, and the spectral window).
* - `--save-project`
  - Also save the run as a pre-whitening project (`*.cupw.json`) to open in the GUI,
    edit with {class}`~cuperiod.PreWhitenSession`, or verify later. A data file inside
    the project's folder is referenced; otherwise the data are embedded. Above
    `embed_data_max_points` (200 000 points; `CUPERIOD_SESSION_EMBED_DATA_MAX_POINTS`)
    the library refuses to embed unasked, so the command embeds anyway and prints a
    `note: embedded all N data points …` line. A project larger than
    `max_project_bytes` (64 MiB; `CUPERIOD_SESSION_MAX_PROJECT_BYTES`) is not written:
    the command ends with `Error: …` and exit status 1, after printing the solution, and
    repeats why the data could not be referenced. A project path whose folder does not
    exist or cannot be written is a usage error (exit 2) naming that path.
* - `--export-script`
  - Also write a Python script that reproduces the run with the library. It is written
    even when saving the project fails.
```

## `session` — pre-whitening projects

```bash
cuperiod session verify star.cupw.json
cuperiod session replay star.cupw.json -o replayed.cupw.json
cuperiod session script star.cupw.json -o star_session.py
```

Works on `*.cupw.json` projects written by `prewhiten --save-project`, the library or
the GUI ({doc}`interactive-prewhitening`).

```{list-table}
:header-rows: 1
:widths: 30 70

* - Command
  - Does
* - `verify PROJECT`
  - Replay the project's action log on a fresh session and check that it reproduces the
    stored solution; exits with status 1 when it does not.
* - `replay PROJECT [-o OUT]`
  - Re-create the project from its log, print the solution, and optionally save the
    replayed session (its log and solution) as a new project. A project stores no undo
    states, so the saved one opens at its last state with nothing to undo, like any
    other.
* - `script PROJECT -o OUT`
  - Export a Python script that rebuilds the session with the public API and says
    whether it reproduced it.
* - `--data FILE`
  - (all three) A light-curve file to use instead of the project's own data — the data
    file moved or renamed, say. It is read with the project's recorded domain and, for a
    project that references its data file, the recorded column mapping, so a project made
    with `prewhiten --time/--value/--error` needs no column options here. When the file
    lacks the recorded columns (the same points under other names), its columns are
    detected instead and a `note:` says so. It must hold the same points (the content
    hash is checked).
* - `--backend`
  - (`verify`, `replay`) The spectrum backend; default: the recorded one when this
    machine has it, else the best CPU backend. Steps replay their logged peak seeds, so
    a project verifies on any backend.
```

## `pdot` — is the period changing?

```bash
cuperiod pdot star.csv --period 0.5168 --out drift.json
```

Runs the joint (period, dP/dt) search of {doc}`period-change` and, by default, follows
with the phase-tracking measurement at the found period and a light-time orbit search
over the tracked O−C. The printed summary gives the best cell of the search map (with the
dP/dt = 0 comparison), then the measured dP/dt with its uncertainty and the F-test
against a constant period, then the light-time block: outer period, amplitude, ΔBIC
against the polynomial alone, and the secular dP/dt with the cycle removed. Every error
in that block is the formal error times {attr}`~cuperiod.LTTEFit.error_inflation`
(`sqrt(chi2_reduced)`, floored at 1), and a line `errors  formal x <factor>` says so.

The search knobs follow {class}`~cuperiod.PdotSettings`: one you do not give takes its
`CUPERIOD_PDOT_<FIELD>` environment variable when that is set, else the default in the
table, and one you give always wins (`--help` names each variable; `--pdot-range` is
`CUPERIOD_PDOT_PDOT_OVER_P_RANGE` and `--segments` is `CUPERIOD_PDOT_N_SEGMENTS`).
Options the search cannot run with — `-m BLS` or `-m TLS`, a `--pdot-range` outside the
period law's domain, a grid larger than `max_grid_cells` — are usage errors (exit
status 2).

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Meaning
* - `--period`
  - Center of the search window in days (omit to seed from a standard run).
* - `--method`, `-m`
  - Wrapped periodogram method (default GLS; BLS/TLS are not searchable).
* - `--max-cycle-drift` / `--samples-per-drift` / `--pdot-range`
  - The dP/dt axis: half-range in accumulated drift cycles (default 10), its
    oversampling (default 5), or explicit `(dP/dt)/P` bounds as `lo,hi` in 1/day
    (default: from `--max-cycle-drift`).
* - `--window-rayleigh` / `--samples-per-peak`
  - The frequency window around the center period: its half-width in Rayleigh widths
    (default 30) and its oversampling (default 10).
* - `--harmonics` / `--segments`
  - Tracking template order (default BIC-selected) and segment count (at least 4;
    omit for automatic).
* - `--harmonic-scale`
  - Extra refinement of the `(dP/dt)/P` step for methods that resolve structure sharper
    than a sinusoid (at least 1; default per method).
* - `--model`
  - Period law the search warps by: `constant-pdot` (default) or `exponential`.
* - `--detrend-width`
  - Running-median window in days, applied before anything else. Real survey
    photometry usually needs one — see {doc}`period-change`.
* - `--track / --no-track`
  - Follow the search with the phase-tracking measurement (default on).
* - `--ltte / --no-ltte`
  - Search the tracked O−C for a cyclic, light-travel-time variation with
    {meth}`~cuperiod.PhaseTrackResult.ltte_search` (default on; needs `--track`).
* - `--profile`
  - Also measure the model-free P(t) track and rank the period laws over it —
    the check on whether one constant rate is enough. The law table prints the
    error-scaled BIC (`bic_scaled`, the ranking criterion) and each law's gap to the
    best, and marks fits held at a bound.
* - `--backend`
  - `auto` | `cpu` | `gpu` | a concrete backend name (default `auto`).
* - `--out` / `--save-map`
  - Write the results as JSON, and/or the 2-D search map as `.npz`.
```

## `pdot-batch` — period changes for a catalogue

```bash
cuperiod pdot-batch "lcs/*.csv" --out pdot.parquet --detrend-width 40 --workers 8
```

Runs the whole chain of {doc}`period-change` over many light curves, with the same
inputs and worker options as `batch` and the same search knobs as `pdot`, with the same
`CUPERIOD_PDOT_*` defaults. `-m BLS` or `-m TLS` is a usage error here too; a search one
star cannot run (too large a grid, say) fails that star alone. It writes up to three
tables — the summary (one row per light curve), and optionally the O−C segments and the
P(t) intervals, which are large and are only written when asked for. A light curve that
fails still gets a summary row carrying the reason in `status`; those rows, not an
errors file, list the failures.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Meaning
* - `--out` / `--sink`
  - The summary table: `.parquet`/`.csv`, or a directory (resumable).
* - `--oc-sink` / `--profile-sink`
  - The per-segment O−C and per-interval P(t) tables. Directory sinks must
    differ from each other and from `--out`.
* - `--period`
  - Seed period in days for every light curve (omit to seed each search).
* - `--detrend-width` / `--detrend-method`
  - Running-median trend removal: window in days, and `median` (default) or `binned`
    (`binned` skips bins with fewer than `min_points` points as nodes).
* - `--profile / --no-profile`
  - Measure the model-free P(t) track and fit period laws to it (default on).
* - `--ltte / --no-ltte`
  - Search each star's O−C for a light-time orbit (default on), into the summary's
    `ltte_*` columns.
* - `--laws`
  - `auto` (default), `none`, or a comma-separated subset of the registered laws.
* - `--points-per-interval` / `--period-window`
  - Interval size for the P(t) track, and the fractional half-width of each
    interval's period search.
* - `--plots` / `--plot-keys`
  - Directory for per-star diagnostic figures, optionally restricted to
    listed keys. Needs `pip install "cuperiod[plot]"`; without matplotlib the command
    stops before any work. The O−C panel draws the light-time orbit only when the O−C
    supports it (ΔBIC > 10, outer period not at the edge of the scan).
* - `--workers` / `--chunk-size` / `--resume / --no-resume`
  - As `batch`. One light curve costs a whole grid search, so chunks are small
    (default 32 light curves). Resuming works per light curve across the three
    tables: a star is skipped only when its summary row is `ok` and every requested
    table holds all of its rows; failed stars are retried and replace their rows.
```

With the default `--device cpu`, `--backend auto` runs every worker on a CPU backend and
a GPU-only backend is refused as a usage error.

## `batch-prewhiten` — many pulsators

```bash
cuperiod batch-prewhiten "lcs/*.csv" --out modes.parquet --workers 8 -n 20
```

Takes the same inputs and worker options as `batch` (including `--chunk-size`), and the
same extraction options as `prewhiten` (`-n`, `--snr`, `--stop`, `--fmin`/`--fmax`,
`--uncertainty`, `--combinations`, `--backend`, the column options) — not `--spacing` or
`prewhiten`'s per-star outputs. As there, an option you do not give takes its
`CUPERIOD_PREWHITEN_<FIELD>` value when that is set. The output has **one row per
extracted component**, each carrying the per-star summary (sample count, baseline, stop
reason, fit statistics) so a single table is self-describing. With the default
`--device cpu` a GPU-only backend is refused. Resuming skips the light curves already in
the sink, by key, and retries failures; failures are listed in an errors file named as
for `batch`.

## `methods` — what's available

```bash
cuperiod methods
```

Lists each registered method with its objective sense, multi-band support, and backends.
The backends are those the method has, installed or not; `doctor` says which are
installed here.

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
resolves to for every method. The best torch device is tagged as the one
`backend="torch"` uses; the tag names `'auto'` as well only when `auto` really resolves to
torch for some method, which takes a non-CPU torch device and no CUDA fast path. Reach for
it first when a GPU isn't being picked up or you're unsure which build of PyTorch you
have.

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
