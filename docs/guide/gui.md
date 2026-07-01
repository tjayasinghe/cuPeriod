# Desktop GUI

cuPeriod ships an interactive **periodogram explorer** — a desktop app that runs any method
with all of its options, draws the full-resolution spectrum, and folds the light curve live
as you drag across peaks. It's a thin presentation layer over the same API the rest of this
guide describes ({func}`~cuperiod.periodogram`, {func}`~cuperiod.batch_periodograms`), so
every result matches the library exactly.

## Install & launch

The GUI is an optional extra (PySide6 + pyqtgraph):

```bash
pip install "cuperiod[gui]"
```

Then launch it from the console script or as a module:

```bash
cuperiod-gui
# or
python -m cuperiod.gui
```

Add `[gpu]` or `[torch]` alongside `[gui]` to explore on an accelerator —
`pip install "cuperiod[gui,gpu]"`. Without the extra installed, `cuperiod-gui` exits with an
actionable message instead of a traceback.

:::{tip}
The window opens with **bundled demo light curves** — a *Kepler* transit, six ASAS-SN
variables (each labelled with its VSX type and literature period), and a synthetic
multi-band curve — so there's something to explore on first launch, no data of your own
required.
:::

## A two-minute tour

1. **Load demo → *Kepler KIC 7532973*.** Pick method **BLS** and press **Compute**. The
   spectrum draws (peak-preserving, smooth to zoom); the phased panel shows the transit as a
   **dip** at the best period.
2. **Slide the marker** across the spectrum (or click a row in the **Peaks** dock) and watch
   the phased curve re-fold live. Toggle the spectrum's **x-axis** (frequency ↔ period) and
   **log** scales, and the **2 cycles** checkbox on the phased panel.
3. **Load demo → *Synthetic multiband*.** With **GLS** the phased panel overlays all three
   bands (colour-coded, with a legend); the **Band** selector chooses *combined* or a single
   band.
4. **Load demo → a folder in batch mode.** Scroll the **Sources** dock (arrow keys or
   Prev/Next); each source computes on demand and revisits are instant (cached).
5. **Toggle the theme** (dark ↔ light — remembered next launch). With `[gpu]`/`[torch]`
   installed, the info bar names the backend and device actually used.

## The interface

| Area | What it does |
| --- | --- |
| **Controls** (left) | Choose the method and edit its settings. The form is built automatically from each method's settings model ({doc}`tuning`), so every knob — grid bounds, `n_harmonics`, transit-duration fractions, backend/device/precision — is exposed with the right type and defaults. Press **Compute** to run. |
| **Spectrum** (centre) | The full-resolution periodogram, rendered at interactive speed. Drag the marker to select a trial period; toggle the **x-axis** between frequency and period and switch either axis to **log**. |
| **Phased** | The light curve folded on the selected period, updating live as you move the marker. **2 cycles** repeats the fold; for multi-band data a **Band** selector overlays all bands or isolates one. |
| **Raw light curve** | The unfolded time series for the loaded source. |
| **Peaks** (dock) | The ranked N-best periods ({doc}`results`). Click a row to jump the marker (and the fold) to that peak. |
| **Sources** (dock) | In batch mode, the list of light curves; navigate with Prev/Next or the arrow keys. |
| **Info bar** | The backend and device actually used for the last run, plus timing — handy for confirming that `auto` reached your GPU. |

Compute runs **off the UI thread**, and only the latest request wins, so the window stays
responsive even while a long grid is evaluating and rapid re-runs don't queue up.

## Loading your own data

**File → Open** reads a single light curve through the same auto-detecting loader as the CLI
(`*.csv`, `*.ecsv`, `*.fits`/`*.fit`/`*.fz`, `*.parquet`/`*.pq`, `*.tsv`/`*.tab`, `*.dat`,
`*.txt`). For tabular files a **preview dialog** shows the first rows and the column mapping
cuPeriod auto-detected (time / value / error / band) before you commit — the same
{class}`~cuperiod.ColumnMap` resolution described under {doc}`light-curves`. If a band/filter
column is present the file loads as a {class}`~cuperiod.MultiBandLightCurve`; otherwise as a
single {class}`~cuperiod.LightCurve`.

**Batch mode** points the **Sources** browser at a whole set of light curves — a folder, a
glob, or a list of files — reusing the same input resolution as
{func}`~cuperiod.batch_periodograms` ({doc}`batch`). Sources are read lazily (a big folder
isn't loaded up front) and each computed source is cached, so scrolling back and forth is
instant.

## Backends & devices

The GUI honours the same `backend`, `device`, and `precision` settings as the library
({doc}`backends`). With `[gpu]` installed on an NVIDIA machine, `backend="auto"` reaches the
CUDA fast paths; with `[torch]` it reaches an AMD/Intel/Apple GPU or the portable CPU path.
The info bar reports which backend and device a run actually used. To see the full picture
of what's available before launching, run `cuperiod doctor` ({doc}`cli`).

## Notes

:::{note}
**Windows OpenMP.** Like the tests and CLI, the `cuperiod-gui` entry point sets
`KMP_DUPLICATE_LIB_OK=TRUE` before importing anything, so a torch backend can load after
numpy/MKL without the *"libiomp5md.dll already initialized"* abort. See the OpenMP note under
{doc}`../installation`.
:::

The GUI is a **pure presentation layer** — it adds no compute of its own and imports nothing
from the core beyond the public API, so anything you can do in the app you can reproduce in a
script with {func}`~cuperiod.periodogram`.

---

That completes the user guide. For exact signatures, see the {doc}`../api/index`; for
measured parity and speedups, the {doc}`../benchmarks`.
