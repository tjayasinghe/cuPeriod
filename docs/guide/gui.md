# Desktop GUI

cuPeriod ships an interactive **periodogram explorer** — a desktop app that runs any method
with all of its options, draws the full-resolution spectrum, and folds the light curve live
as you move across peaks — and a **Period04-style pre-whitening workbench** with an
editable frequency table, point editing, subsets, undo and project files. It's a thin
presentation layer over the same API the rest of this guide describes
({func}`~cuperiod.periodogram`, {class}`~cuperiod.PreWhitenSession`), so every result
matches the library exactly. Everything in it can be reached from the keyboard.

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
**Load demo** (on the toolbar and in the File menu) always offers a **synthetic
multi-band demo curve**, so there's something to explore on first launch, no data of
your own required. In a source checkout (or with `CUPERIOD_EXAMPLE_DATA` pointed at
`examples/data`) the same menu also lists a *Kepler* transit and six ASAS-SN variables,
each labelled with its VSX type and literature period; picking one also applies its
suggested method (BLS for the transit).
:::

## A two-minute tour

1. **Load demo → *Kepler KIC 7532973*.** The raw light curve is drawn at once. Method
   **BLS** is already chosen; press **Compute** (<kbd>F5</kbd> or
   <kbd>Ctrl+Return</kbd>). The spectrum draws (peak-preserving, smooth to zoom); the
   phased panel shows the transit as a **dip** at the best period.
2. **Click a peak** in the spectrum (or a row in the **Peaks** pane, or drag the shaded
   band) and watch the phased curve re-fold. With the spectrum focused
   (<kbd>Ctrl+1</kbd>), <kbd>Left</kbd> / <kbd>Right</kbd> walk from peak to peak and
   <kbd>Enter</kbd> folds there. Toggle the spectrum's **x-axis** (frequency ↔ period)
   and **log** scales, the **2 cycles** box, and **×2** / **÷2** beside the period badge.
3. **Load demo → *Synthetic multiband*.** With **GLS** the phased panel overlays all three
   bands, each with its own colour *and* marker shape; the **Band** selector chooses
   *combined* or a single band.
4. **Load demo → a folder in batch mode.** Scroll the **Sources** pane (arrow keys or
   Prev/Next); each source computes on demand and revisits are instant (cached).
5. **Switch Analysis to *Pre-whitening*.** The data's amplitude spectrum appears on its
   own. Press **Run pre-whitening** (Auto, <kbd>F5</kbd>): the residual spectrum is
   overlaid, every extracted component is marked, and the **Frequencies** pane lists them
   with uncertainties and S/N. Now take over: <kbd>F7</kbd> steps one peak at a time,
   <kbd>Shift</kbd>+click adds a peak, typing `=F1+F2` in a frequency cell ties a
   combination, a lasso in the raw light curve deletes outliers, and <kbd>Ctrl+Z</kbd>
   undoes any of it — see {doc}`interactive-prewhitening`. The **window** box overlays
   the sampling's spectral window, the **A/Asp** column flags amplitudes entangled with a
   neighbour (✱), and for a g-mode star the **Period spacing** pane scans for a regular
   spacing and draws the échelle diagram.
6. **Toggle the theme** (<kbd>Ctrl+T</kbd>) and **zoom** the interface
   (<kbd>Ctrl++</kbd> / <kbd>Ctrl+-</kbd>) — both are remembered. With `[gpu]`/`[torch]`
   installed, the info bar names the backend and device actually used.

## The window

| Area | What it does |
| --- | --- |
| **Menu bar** | File, Edit, Pre-whitening, Data, View and Help, each with an <kbd>Alt</kbd> mnemonic on every menu and item (see [below](#menus)). |
| **Controls** (left) | **Analysis** (Periodogram or Pre-whitening), the method, the **Backend** and **Band** pickers, and the **Options** form for the method's settings. **Compute** — **Run pre-whitening** with **Step** beside it in pre-whitening — runs; **Stop** (<kbd>Esc</kbd>) stops. A disabled Compute button says why in its tooltip. |
| **Spectrum** (centre) | The full-resolution periodogram or amplitude spectrum. Click a peak to fold on it; drag the shaded band; toggle frequency ↔ period and **log** scales; **peaks** hides the markers and the band. The x-axis and log choice are remembered per kind of result; **Reset** or a double click restores the default view. |
| **Phased** | The light curve folded on the selected period. **2 cycles** repeats the fold; **×2** / **÷2** and a typed **P (d)** refold at another period. Multi-band data are overlaid, one colour and one marker shape per band. |
| **Raw light curve** | The unfolded time series (View ▸ *Show raw light curve*, <kbd>Ctrl+R</kbd>); it stacks under the fold when there is no room beside it. Curves above 20 000 points are drawn as a fixed random subset (*N of M points shown*) — computations always use every point. In a pre-whitening session it also shows residuals and the model, edits points, and a click puts a time cursor where Subsets ▸ *Split at* cuts. |
| **Peaks** (pane) | The ranked N-best periods ({doc}`results`), with frequency and the method's own statistic (power, θ, entropy, string length…). Click a row to fold on that peak. |
| **Frequencies** (pane) | In pre-whitening, the extracted components with 1σ uncertainties, S/N, false-alarm probability, blend ratio (**A/Asp**) and combination identifications, plus the stop reason and fit statistics. In a session the **session bar** sits above it and the table is editable. |
| **Subsets**, **Variations**, **History** (panes) | Pre-whitening sessions: partitions with zero points, weights and per-subset amplitudes; amplitude/phase variation tests; the action log with undo to any state. |
| **Period spacing** (pane) | In pre-whitening, the comb scan over trial spacings and the échelle diagram of the modes in the series, with the mean spacing, its tilt and the buoyancy radius Π₀. Runs on the *independent* components, so identified combinations cannot pollute the pattern. |
| **Sources** (pane) | In batch mode, the list of light curves; Prev/Next or the arrow keys move through it. |
| **Info bar** | What runs and for how long, then the backend and device a run actually used and its timing; *settings changed* marks a result whose controls no longer match it. |

The result panes share one tab group on the right; each has 24 px **Float** / **Close**
buttons, View ▸ *Panes* brings back any closed pane, and View ▸ *Reset layout* puts
everything back. In a narrow dock the Subsets, Variations and Period spacing panes and
the session bar wrap their controls onto more rows instead of pushing them out of sight,
and a pane still wider than its dock scrolls rather than squeezing the plots. The window
sizes itself to the screen — its minimum size follows what the panes need, also after a
zoom — and keeps its layout, the analysis, every form's options, the raw-view choice,
the theme and the zoom across launches. Switching between Periodogram and Pre-whitening
keeps each one's result.

### Menus

| Menu | Holds |
| --- | --- |
| **File** | New session (<kbd>Ctrl+N</kbd>), Open light curve… (<kbd>Ctrl+O</kbd>), Open folder… (<kbd>Ctrl+Shift+F</kbd>), Open recent, Load demo, Open project… / Save project / Save project as… (<kbd>Ctrl+Shift+O</kbd> / <kbd>Ctrl+S</kbd> / <kbd>Ctrl+Shift+S</kbd>), Export script… (<kbd>Ctrl+E</kbd>), Export frequencies…, Verify project…, Unsaved sessions, Quit (<kbd>Ctrl+Q</kbd>). |
| **Edit** | Undo (<kbd>Ctrl+Z</kbd>), Redo (<kbd>Ctrl+Shift+Z</kbd>, <kbd>Ctrl+Y</kbd>), Delete. |
| **Pre-whitening** | Auto, Step, Add peak, Add at frequency…, Add peak at cursor, Calculate, Improve all / selected, Use / don't use, Fix, Edit tie…, Compute uncertainties, Polish and prune, Renumber, Stop. |
| **Data** | Restore selected / all points, the Pan / Rectangle / Lasso tools, Clear selection. |
| **View** | Focus a pane (<kbd>Ctrl+1</kbd> … <kbd>Ctrl+5</kbd>), Analysis (with Compute periodogram), Panes, Show raw light curve (<kbd>Ctrl+R</kbd>), theme (<kbd>Ctrl+T</kbd>), Zoom in / out / Reset zoom, Reset layout. |
| **Help** | Keyboard shortcuts (<kbd>F1</kbd>), What's This? (<kbd>Shift+F1</kbd>), Documentation, About cuPeriod. |

Commands that need a session or are not possible while something runs are disabled, so
the menus always show what can be done now.

## The Options form

The form is built automatically from each method's settings model ({doc}`tuning`), so
every knob is exposed with the right type and defaults, grouped into sections: **Search
range** first, then the method's model (or *Extraction* and *Stopping* for
pre-whitening), **Peaks**, **Multiband** (only when a joint multiband fit will run — with
*combined (all bands)* chosen), and *Uncertainties*, *Combinations* and *Advanced*
collapsed. Labels carry units; **Reset to defaults** puts everything back.

- A line under the form shows the description of the focused option; every field also
  has a status tip and What's This? help (<kbd>Shift+F1</kbd>, then click it).
- Automatic values (a grid bound that depends on the data, say) are shown read-only, so
  they stay readable and focusable; the grid hint describes the band that will be
  analysed.
- Values are used exactly as typed and keep full precision (`1e-12`, `1234567.5`).
  *Samples per peak* shows the GUI's default of 10, and an explicit 5 is honoured.
- `CUPERIOD_<METHOD>_<FIELD>` environment overrides seed the form, and the Backend
  picker's *auto* defers to the settings' `backend`.
- Pre-whitening **stop criteria** are check boxes; the thresholds of criteria that are
  off are greyed out.
- Peak separation, alias tolerance and harmonic order re-select the peaks without
  recomputing. `n_peaks` (owned by the Peaks spin box), `store_spectra` and
  `downsample_points` are not in the form.
- The Backend and Band choices persist across loads, batch steps and method switches
  while they are offered.

## Loading your own data

**Open light curve…** (<kbd>Ctrl+O</kbd>), a drop on the window, Open recent and a batch
file that failed to load all go through the **import dialog**. It shows the file and
lets you choose:

- the **Time**, **Brightness**, **Error** and **Band** columns (error and band can be
  *none*), pre-filled with what cuPeriod auto-detects — the same
  {class}`~cuperiod.ColumnMap` resolution described under {doc}`light-curves`;
- whether the values are **Magnitudes** or **Fluxes** (inferred from the column until
  you choose);
- for text tables, the **delimiter** and whether the **first row holds the column
  names** — semicolon-separated and headerless files (OGLE, Period04 `JD mag err`) are
  recognised on opening.

The *File* tab labels each column's role and the *As loaded* tab shows the first rows
exactly as they will be read, at full precision. A note names the time scale the numbers
look like (BJD, HJD, MJD, BTJD, …) and warns when they look like seconds or decimal
years. **Load** is unavailable, with the reason shown, while the chosen columns cannot
work (unset, clashing, not numeric). The dialog remembers the last choice and never
blocks the window.

Supported formats: `*.csv`, `*.ecsv`, `*.fits`/`*.fit`/`*.fz` (also compressed),
`*.parquet`/`*.pq`, `*.tsv`/`*.tab`, `*.dat`, `*.txt`. A band/filter column gives a
{class}`~cuperiod.MultiBandLightCurve` in every format; rows without a band label are
dropped and reported, and an auto-detected band column with more than 32 labels is
treated as an identifier (the file loads as one band). Wide one-row-per-object tables
and header-only files are refused with a message. Notes from the reader (dropped rows,
an ignored band column) are shown with the loaded curve, which is drawn at once — the
spectrum and the fold follow the first compute.

**Batch mode** — **Open folder…** (<kbd>Ctrl+Shift+F</kbd>), or dropping several files or
folders on the window — points the **Sources** browser at a whole set of light curves,
reusing the input resolution of {func}`~cuperiod.batch_periodograms` ({doc}`batch`).
Sources are read lazily and each computed source is cached. A source that cannot be read
is greyed out with the reason in its tooltip, skipped by Prev/Next and the arrow keys,
and counted (*N unreadable*); when its columns could not be resolved, the error banner
offers *Choose columns…*, which opens the import dialog for it. A drop may mix files and
folders (a dropped `*.cupw.json` opens as a project); anything unrecognised is skipped
and named in the status bar. Loading a new batch clears the source filter.

## Computing, stopping and errors

Compute runs **off the UI thread**, so the window stays responsive while a long grid
evaluates, and only the latest request is shown. **Stop** (<kbd>Esc</kbd>) releases a
running periodogram at once — the abandoned run finishes in the background and its
result is cached, not shown — and ends a pre-whitening run after its current step,
keeping what it found. Closing the window during a computation ends the program
straight away.

Results are cached by the **content** of the data, not the file name, so a different
star in a same-named file, or a file edited on disk, never shows a stale result; each
cache holds at most 512 MiB. Errors are explained in plain language with a next step;
the banner stays until it is dismissed or the next attempt, its tooltip and
*Show Details* hold the full text, and screen readers announce it. A failed run leaves
the result on screen untouched. Every command answers in the status bar, also when it
changes nothing — Step at the stopping rule, *Renumber* with nothing to renumber,
*Restore all points* with nothing deleted, an Improve with nothing to refine — and an
edit that is refused says why in the window's terms (*Add exactly here*, untie it
first), never as a keyword argument of the library.

## Exports and tables

The Peaks and Frequencies tables have **Copy** and **Export CSV** buttons, a header
right-click menu to show or hide columns, and full-precision, locale-independent
numbers; <kbd>Ctrl+C</kbd> copies while the table has the focus, and <kbd>Tab</kbd>
leaves the table (the arrow keys move between cells). The Peaks CSV has a `frequency
(c/d)` column and names the method's statistic; the Frequencies CSV always has the
{meth}`Sinusoid.to_dict <cuperiod.Sinusoid.to_dict>` columns (plus the session columns
in a session); a pre-whitening spectrum CSV names its column `amplitude`. PNG exports
render at the screen's scale and append `.png` when the name has no image suffix. Every
*Export…* — the spectrum's and the fold's CSV / PNG, the tables' Export CSV — asks for
the file in the window's usual Save dialog, like *Save project*.

## Pre-whitening interactively

In pre-whitening the Frequencies pane becomes a Period04-style workbench: Auto, Step and
Add peak; a pick cursor and Shift+click in the spectrum; an editable table with fixes,
ties, per-subset amplitudes and notes; rectangle and lasso deletes in the light curve;
nightly zero points, weights and variation tests in the Subsets and Variations panes;
undo to any state from the History pane; and `*.cupw.json` projects. The session for the
displayed curve opens on its own — on the CPU — so the data spectrum appears without a
click; when it cannot be computed the spectrum pane says why (*Spectrum unavailable:
…*) and the session commands stay off. **Calculate** (<kbd>F8</kbd>) and **Improve**
(<kbd>F9</kbd>, <kbd>Shift+F9</kbd>) fill in the table's uncertainties (the quick
re-calculation after each edit shows "–"), and Pre-whitening ▸ *Compute uncertainties*
runs the configured estimator, a bootstrap say. Unsaved work survives switching source,
band or analysis (File ▸ *Unsaved sessions*, by the name each source was loaded under);
closing, *New session* or opening other data asks **Save / Discard / Cancel** first. The
title bar shows the data and the project, with `*` for unsaved work:
`cuPeriod — star.csv — star.cupw.json*`. All of it is described in
{doc}`interactive-prewhitening`.

## Keyboard and accessibility

Every command is in the menu bar and reachable from the keyboard, with a visible focus
ring on every control, and nothing is conveyed by colour alone.

- **Focus.** <kbd>Tab</kbd> moves through the controls in reading order — the pickers,
  the Options fields (the form scrolls to the field Tab reaches), the Compute / Run
  buttons, then on to the plots and the panes, and round again; <kbd>Shift+Tab</kbd>
  goes back the same way. Panes hidden behind another tab and the tab bars' scroll arrows are
  skipped. <kbd>Ctrl+1</kbd> … <kbd>Ctrl+5</kbd> jump to the spectrum, the frequency
  table, the light curve, the subsets and the history.
- **The spectrum** is keyboard-operable: <kbd>Left</kbd> / <kbd>Right</kbd> move a cursor
  between local maxima (<kbd>Ctrl</kbd>: marked peaks; <kbd>Shift</kbd>: one grid
  sample), <kbd>Home</kbd> goes to the tallest, <kbd>Enter</kbd> folds there,
  <kbd>+</kbd> / <kbd>-</kbd> zoom about it, <kbd>0</kbd> resets, and the
  <kbd>Menu</kbd> key opens the right-click menu (*Fold here*, *Move cursor here*,
  *Zoom to ±1 c/d*, *Copy frequency*). A click folds at the nearest peak within 12 px
  (<kbd>Alt</kbd>+click: exactly there) and a double click only resets the view.
- **The light curve** in a pre-whitening session takes the tool letters
  (<kbd>V</kbd> / <kbd>X</kbd> / <kbd>G</kbd>), <kbd>Del</kbd>, <kbd>U</kbd> and
  <kbd>Esc</kbd> for the point selection, and <kbd>Left</kbd> / <kbd>Right</kbd> move its
  time cursor (<kbd>Shift</kbd>: ten times further); its description names the cursor's
  time.
- **Screen readers** get accessible names and status tips on every control, What's
  This? help on every option, and a live description of the spectrum (method, range,
  peaks, best period, selection, cursor).
- **Colour.** Both themes meet WCAG 2.1 AA contrast (4.5:1 for text, 3:1 for borders,
  focus rings and selections). Bands differ by marker shape as well as colour, component
  markers by shape (circle, square, diamond, hollow), the residual spectrum has its own
  colour that stays distinct under colour-vision deficiencies, and a row that fails a
  criterion carries a `!`, not only a colour.
- **Size.** <kbd>Ctrl++</kbd> (or <kbd>Ctrl+=</kbd>), <kbd>Ctrl+-</kbd> and
  <kbd>Ctrl+0</kbd> zoom text and plot labels together and are remembered; a larger
  system or application font also enlarges the interface. Check boxes, spin buttons,
  splitters, scroll bars and pane buttons are large pointer targets.
- **Help.** <kbd>F1</kbd> lists every shortcut; <kbd>Shift+F1</kbd> then a click explains
  any control.

Window keys that are not in the command list below:

| Key | Does |
| --- | --- |
| <kbd>F5</kbd>, <kbd>Ctrl+Return</kbd> | Compute the periodogram (in pre-whitening the same keys run Auto) |
| <kbd>Ctrl+Shift+F</kbd> | Open a folder of light curves |
| <kbd>Ctrl+R</kbd> | Show or hide the raw light curve |
| <kbd>Ctrl+T</kbd> | Toggle the dark / light theme |
| <kbd>Ctrl++</kbd>, <kbd>Ctrl+=</kbd> / <kbd>Ctrl+-</kbd> / <kbd>Ctrl+0</kbd> | Zoom in / out / reset |
| <kbd>Shift+F1</kbd> | What's This? |
| <kbd>Ctrl+Q</kbd> | Quit (asks about unsaved work) |

The commands of the menu bar, generated from the app's command registry (single letters
work only while their pane has the focus, never in the Options form):

```{shortcut-table}
```

## Backends & devices

The GUI honours the same `backend`, `device`, and `precision` settings as the library
({doc}`backends`). With `[gpu]` installed on an NVIDIA machine, `backend="auto"` reaches the
CUDA fast paths; with `[torch]` it reaches an AMD/Intel/Apple GPU or the portable CPU path.
The *runs on* hint resolves `auto` and bare `torch` with the options' backend and device
exactly as the run will, and the info bar reports what a run actually used — *CPU* for a
`torch:cpu` run, *GPU (MPS)* or *GPU (XPU)* for other torch devices. Pre-whitening
previews always run on the CPU; a session runs on a GPU only when you choose one. To see
the full picture of what's available before launching, run `cuperiod doctor` ({doc}`cli`).

## Notes

:::{note}
**Windows OpenMP.** Like the tests and CLI, the `cuperiod-gui` entry point sets
`KMP_DUPLICATE_LIB_OK=TRUE` before importing anything, so a torch backend can load after
numpy/MKL without the *"libiomp5md.dll already initialized"* abort. See the OpenMP note under
{doc}`../installation`.
:::

- Text taken from a file — column headers, band names, file names, error messages
  quoting them — is always shown literally, never as rich text, so a header such as
  `<b>…</b>` or `<img src=…>` is displayed as typed and never rendered or fetched.
- The header shows a git revision only when cuPeriod runs from a source checkout (it is
  read from the checkout, without running git). Demo data come from the checkout's
  `examples/data` or `CUPERIOD_EXAMPLE_DATA` only.
- The GUI is a **pure presentation layer** — it adds no compute of its own and imports
  nothing from the core beyond the public API, so anything you do in the app you can
  reproduce in a script with {func}`~cuperiod.periodogram` or
  {class}`~cuperiod.PreWhitenSession` (File ▸ *Export script…* writes that script for a
  pre-whitening session).

---

That completes the user guide. For exact signatures, see the {doc}`../api/index`; for
measured parity and speedups, the {doc}`../benchmarks`.
