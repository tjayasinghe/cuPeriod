# Interactive pre-whitening

{func}`cuperiod.prewhiten` runs the extraction loop on its own and stops where its
stopping rule says ({doc}`prewhitening`). Real light curves usually need a hand on the
wheel as well: a peak the rule rejected but you know is there, a combination frequency
that should follow its parents, a few outliers, a night with a zero-point jump, a
frequency to hold while the rest are refined, a mode whose amplitude might change from
season to season. That is the Period04 way of working (Lenz & Breger 2005), and cuPeriod
supports it end to end:

- in the **desktop app**, where the spectrum, the frequency table, the light curve and
  the subsets are all editable, with undo, a history and project files;
- as a **library object**, {class}`cuperiod.PreWhitenSession`, which is what the app
  drives — every button is one call of it, so whatever you do by hand can be written
  down, replayed and checked.

Every action is logged with the arguments it really used (including the peak a step
started from), so a session can be **saved** as a `*.cupw.json` project, **verified**
by replaying its log, and **exported as a Python script** that reproduces it with the
public API. And `cup.prewhiten(...)` is exactly a session run to its stopping rule and
finalized — the numbers do not depend on which route you take.

## In the desktop app

Switch **Analysis** to *Pre-whitening* (View ▸ Analysis, or the selector at the top of
the Controls panel) and load a light curve. The data's amplitude spectrum appears after
a moment without a click: it is computed on the CPU, so browsing never wakes up a GPU.
When it cannot be computed — a curve with no time baseline, say — the spectrum pane
says why instead (*Spectrum unavailable: …*) above the light curve, and the session
commands stay off until the Options or the data change; **Run pre-whitening** and
**Step** still try, and the error banner says what went wrong.

### Extracting frequencies

| Action | How | What happens |
| --- | --- | --- |
| **Auto** | **Run pre-whitening** button, <kbd>F5</kbd> or <kbd>Ctrl+Return</kbd> | Extract until the stopping rule fires, then polish and prune — `cup.prewhiten` exactly. The table and the spectrum fill in live; **Stop** (<kbd>Esc</kbd>) ends after the current step and keeps what was found. |
| **Step** | **Step** button, <kbd>F7</kbd>, or <kbd>S</kbd> in the spectrum or the table | One iteration of Auto: the tallest untested peak is fitted and kept only if it passes the stopping criteria. |
| **Add peak** | <kbd>F6</kbd> | The tallest untested peak, added *without* the acceptance test. |
| **Add at the cursor** | click the spectrum to move the pick cursor, then <kbd>A</kbd> or <kbd>Insert</kbd>; or <kbd>Shift</kbd>+click | The residual peak at the cursor is added and refined. <kbd>Ctrl+Shift</kbd>+click adds exactly where you clicked. |
| **Add a typed frequency** | the *Add frequency* field above the table (<kbd>Ctrl+F6</kbd>) | Accepts `12.34` (c/d), `100 µHz`, periods such as `0.081 d` or `2 h`, and combinations such as `=F1+F2` or `2f1` (added and tied in one step). |
| **Calculate** | <kbd>F8</kbd>, or <kbd>C</kbd> | Re-solve amplitudes, phases and zero points with every frequency held; fills in the uncertainties. |
| **Improve all / selected** | <kbd>F9</kbd> / <kbd>Shift+F9</kbd>, or <kbd>I</kbd> / <kbd>Shift+I</kbd> | Non-linear refinement of every free frequency, or only of the selected rows (the others held); fills in the uncertainties too. |
| **Polish and prune** | Pre-whitening menu | The final simultaneous fit and the S/N re-check of Auto. |
| **Compute uncertainties** | Pre-whitening menu | The full result with the configured estimator (a bootstrap can take a while; Stop ends it). |

Above the table, the **session bar** repeats these buttons together with Delete, Undo,
Redo and Stop, and a chip shows what Step would try next, for example
`12.34012 c/d · P 0.081037 d · A 3.9 mmag · S/N ≈ 8.2 · near F1 + F2`. <kbd>N</kbd> in
the spectrum moves the pick cursor to that candidate. In a narrow pane the bar's buttons
wrap onto more rows rather than hiding.

Every command answers in the status bar, also when it changes nothing: Step at the
stopping rule, *Renumber* with the numbering already in order, *Restore all points* with
nothing deleted or an Improve with nothing free to refine says so, and an edit the
session refuses says why in the window's terms (*Add exactly here*, untie the row
first) — a refused table edit puts the cell back.

In pre-whitening the spectrum shows the data spectrum, the residual spectrum of the
current model, the optional spectral window and **S/N limit** (the acceptance threshold
times the local noise), and one marker per component. The marker's **shape** carries its
state, so nothing depends on colour: a filled circle is a free component, a square a
fixed frequency, a diamond a tied one, a hollow marker a component that is not used, and
a hollow triangle labelled *next* is the candidate. Edits keep the zoom; only Reset, a
double click or a new source re-range the plot.

Two things in the spectrum are kept apart on purpose:

- the **pick cursor** (a dash-dot line) decides where <kbd>A</kbd> / <kbd>Insert</kbd>
  add; a click moves it to the nearest peak within 12 px (<kbd>Alt</kbd>+click: exactly
  there), and it never moves the fold;
- the **selection** — the shaded band, the folded light curve and the selected table
  row — follows a click on a component marker, a row of the table, or a drag of the
  band. A double click only resets the view.

### Editing the frequency table

In a session the **Frequencies** table is editable, and every edit is one undoable
action:

- the check box in the **ID** column puts a component in or out of the model
  (<kbd>Space</kbd>); a component that is not used keeps its values and leaves the fit
  and the peak search;
- **fix f**, **fix A** and **fix ph** hold the frequency, amplitude or phase
  (<kbd>L</kbd> fixes the frequency, <kbd>Shift+L</kbd> all three);
- typing a number into **frequency**, **amplitude** or **phase** sets it (an amplitude
  or phase you type is also held); a period such as `0.081 d` works in the frequency
  cell;
- typing `=F1+F2` or `2f1` into the frequency cell, or any expression into the **tie**
  cell (<kbd>T</kbd>), ties the frequency; an empty tie unties it, and a tie the
  parser refuses reverts with the reason in the status bar and the cell's tooltip;
- double-clicking an identified **combination** ties the row to it;
- **per subset** lets the amplitude, the phase or both take one value per subset (see
  below), and **note** is free text.

<kbd>Space</kbd>, <kbd>L</kbd>, <kbd>Shift+L</kbd>, <kbd>T</kbd>, <kbd>Del</kbd> and
<kbd>Shift+I</kbd> act on every selected row, and the right-click menu has the same row
actions plus Copy and Export CSV. The quick re-calculation after each edit leaves the
uncertainties out: they read "–" with the tooltip *Not computed yet: Calculate (F8) or
Improve (F9, Shift+F9) gives the errors*. An **S/N** with a `!` fails an
active stopping criterion; a fixed or tied value reads "fixed" / "tied" in its error
column. **Export CSV** of a session writes every component, unused ones included, with
`id`, `active`, the fix flags, `tie`, `variation`, `amplitude_rms`, `note` and `flags`
after the usual columns.

### Editing the light curve

With a session open, the raw light-curve panel (View ▸ *Show raw light curve*,
<kbd>Ctrl+R</kbd>) plots **Data**, **Residuals** or **Data + model**, colours and shapes
points by subset, and has three tools: **Pan** (<kbd>V</kbd>; a <kbd>Shift</kbd>-drag
still selects), **Rectangle** (<kbd>X</kbd>) and **Lasso** (<kbd>G</kbd>);
<kbd>Ctrl</kbd> adds to the selection. The selection bar reads *37 points selected* with
**Delete** (<kbd>Del</kbd>), **Restore** (<kbd>U</kbd>) and **Clear** (<kbd>Esc</kbd>).
**Select beyond N σ** selects the residual outliers with a robust σ per zero-point
subset — the keyboard route to outliers. Deleted points stay visible as hollow grey
crosses and can be selected and restored; one delete is one undoable action, and Data ▸
*Restore all points* (<kbd>Ctrl+Shift+R</kbd>) brings everything back.

A plain click on the plot — with any tool; a drag still pans or selects — puts the
**time cursor** there: a dash-dot line labelled *cursor*, whose time (in the data's own
time scale, not the axis's days since the first sample) the panel title and the status
bar show. With the plot focused (<kbd>Ctrl+3</kbd>), <kbd>Left</kbd> / <kbd>Right</kbd>
move it by 1 % of the visible range (<kbd>Shift</kbd>: 10 %). The Subsets pane's
**Split at** cuts there.

### Subsets: zero points, weights and per-subset amplitudes

The **Subsets** pane splits the points into a *partition* — **Split by** nights (a gap
longer than a given length), TESS sectors, Kepler quarters, seasons, chunks of N days,
bands, a label column of the file, or hand-entered time ranges; when the sampling looks
nightly a hint offers the split. Three check boxes then give the partition a role:

- **Zero point per subset** fits one zero point per subset instead of one mean — the
  usual cure for night-to-night calibration offsets;
- **Weight per subset** applies the relative weights in the table (edit them, set them
  equal, or set them to **1/rms²** of each subset's residuals);
- **Use for amplitude/phase** is the partition the *per subset* column refers to.

Zero points and amplitude/phase use **independent** partitions — a zero point per night
and an amplitude per season is the common case. The **use** box of a row leaves a whole
subset out of the fit (partly checked means some of its points are deleted; the tooltip
says how many) without touching individual deletions. Subsets can be renamed, merged,
split at a time — the light curve's time cursor, a typed time, or else the middle of the
selected subset — and crossed with another partition; selecting rows selects their
points in the light curve (so Delete and Restore act on that night), and short subsets
or subsets whose zero point competes with a low frequency are listed as warnings. In a
narrow dock the pane's controls wrap onto more rows, as the Variations and Period
spacing panes' do.

### Amplitude and phase variations

The **Variations** pane asks whether a component's amplitude or phase changes between
subsets, without changing the model: pick components (by default the selected rows), a
mode (A and φ, A, φ) and a partition, and press **Calculate**. Two stacked plots show the
amplitude per subset with 1σ bars around the common value, and the phase difference —
switchable to O−C in seconds. A table gives the Wald statistic, its p-value and ΔBIC for
each test, in words as well as numbers. **Apply to model** makes the session's model vary
the same way, as one undoable action.

### History, undo and projects

The **History** pane lists every action — an Auto run or a typed combination is one
collapsible group — with the current state marked ▶, undone actions marked *(undone)*
and actions no longer in memory greyed *(not in memory)*. Double-click (or
<kbd>Enter</kbd>) returns to any state; <kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Shift+Z</kbd>
(<kbd>Ctrl+Y</kbd>) step back and forth; *Copy as Python* (<kbd>Ctrl+C</kbd>) copies one
action as library code, and **Export script…** (<kbd>Ctrl+E</kbd>) writes the whole
session as a script.

**Projects.** File ▸ *Save project* (<kbd>Ctrl+S</kbd>), *Save project as…*
(<kbd>Ctrl+Shift+S</kbd>) and *Open project…* (<kbd>Ctrl+Shift+O</kbd>) read and write
`*.cupw.json` files (see [the project file](#the-project-file)). The title bar reads
`cuPeriod — star.csv — star.cupw.json*`; the `*` means unsaved work. Opening a project
whose data file has moved asks for the file. *Verify project…* replays a project's log
and compares it with the stored result.

Unsaved work is never dropped silently. Switching to another source, band or analysis
keeps an edited session alive — it comes back when you return, and File ▸ *Unsaved
sessions* lists them by the name the source was loaded under (a demo's star, a batch
source's label, a file's name) — and closing the window, *New session* (<kbd>Ctrl+N</kbd>),
opening other data or another project asks **Save / Discard / Cancel** when edited work
would be lost.

### How Auto treats your edits

- On an untouched session, Auto with changed Options starts again from scratch, exactly
  like `cup.prewhiten` with those settings — and its result is cached, so going back to
  the same settings shows it at once, together with its session.
- On a session you have edited, Auto **continues** it: the Options form's settings are
  applied to it (a manual edit), then it runs to the stopping rule and finalizes. Step
  and Improve never restart either.
- The spectrum backend is fixed for the life of a session; choosing another backend
  applies to new sessions.
- Any manual edit puts the session back into *searching* with a fresh attempt budget,
  so Step and Auto can carry on after it.

### Keyboard shortcuts

Every command is in the menu bar (File, Edit, Pre-whitening, Data, View, Help, each with
an <kbd>Alt</kbd> mnemonic) and most have a key. Global keys work anywhere in the window;
the single letters work only while their pane has the keyboard focus (move it with
<kbd>Ctrl+1</kbd> … <kbd>Ctrl+5</kbd>), never while you type in the Options form. Help ▸
*Keyboard shortcuts* (<kbd>F1</kbd>) shows the same list in the app. This table is
generated from the app's command registry when the documentation is built:

```{shortcut-table}
```

The focused **spectrum** also takes these keys of its own:

| Key | Does |
| --- | --- |
| <kbd>Left</kbd> / <kbd>Right</kbd> | Move the cursor to the previous / next local maximum (<kbd>Ctrl</kbd>: the previous / next marked peak or component; <kbd>Shift</kbd>: one grid sample) |
| <kbd>Home</kbd> | The tallest peak |
| <kbd>N</kbd> | The next candidate (pre-whitening) |
| <kbd>Enter</kbd> | Fold at the cursor |
| <kbd>+</kbd> / <kbd>-</kbd> / <kbd>0</kbd> | Zoom in / out about the cursor; reset the view |
| <kbd>Menu</kbd> | The right-click menu: add the peak at the cursor (with its S/N), add exactly here, fold here, move the cursor, zoom to ±1 c/d, copy the frequency |

The focused **light curve** moves its time cursor with <kbd>Left</kbd> /
<kbd>Right</kbd> (1 % of the visible range; <kbd>Shift</kbd>: 10 %).

The window-level keys that are not session commands — open folder, raw light curve,
theme, zoom, What's This? — are listed in {doc}`gui`.

## As a library: `PreWhitenSession`

A session is created like `cup.prewhiten` is called — same inputs, same settings, same
grid, same checks — and then driven one action at a time. The examples below run as
written; they use a synthetic δ Scuti-like star observed on 40 nights, with two modes,
their sum frequency, a zero-point offset per night and two outliers:

```python
import numpy as np
import cuperiod as cup

rng = np.random.default_rng(7)
time = np.concatenate([
    2459000.0 + night + rng.uniform(0.0, 0.05) + np.arange(0.0, 0.3, 1 / 288)
    for night in range(40)
])
mag = (
    12.0
    + 0.012 * np.sin(2 * np.pi * 12.3456 * time + 0.3)
    + 0.007 * np.sin(2 * np.pi * 17.8100 * time + 1.1)
    + 0.003 * np.sin(2 * np.pi * 30.1556 * time + 2.0)
    + rng.normal(0.0, 0.004, 40).repeat(87)       # a zero-point offset per night
    + rng.normal(0.0, 0.002, time.size)
)
err = np.full(time.size, 0.002)
mag[[100, 2000]] += 0.08                          # two outliers

settings = cup.PreWhitenSettings(maximum_frequency=50.0)
session = cup.PreWhitenSession((time, mag, err), settings=settings)
```

The printed numbers come from the `numpy` backend; other backends agree to the digits
shown.

:::{note}
`t_ref` (the phase epoch, by default the weighted mean epoch) and the trial-frequency
grid are **frozen when the session is created**, so phases and spectra stay comparable
across edits and a replay is deterministic. The baseline used by the resolution guards
follows the points in use. The `backend` is fixed for the life of the session; `load`,
`restore` and `replay` take `backend=` to move a session to another machine.
:::

### Stepping through the extraction

```python
session.candidate()          # what step() would try next; nothing changes
report = session.step()      # one iteration of the automatic loop
print(report.kind, report.message)
# accepted F1 at 12.3456848 c/d (S/N 71.99)

report = session.run()       # step() until the stopping rule fires (one undo group)
print(report.kind, report.message)
# rejected S/N 3.05 < 4

for c in session.components:
    print(c.label, f"{c.frequency:.6f}", f"{c.amplitude:.5f}", c.origin)
# F1 12.345685 0.01197 auto
# F2 17.810142 0.00681 auto
# F3 30.156283 0.00300 auto
```

Every action returns a {class}`~cuperiod.StepReport`: a `kind`, one human-readable
`message`, the component and frequency it concerned, `statistics` (`snr`, `fap`,
`delta_bic`, `amplitude`, `rms` where they apply), what it `changed` and an optional
`warning`.

| `kind` | Committed? | Meaning |
| --- | --- | --- |
| `accepted` | yes | A step's component passed the stopping criteria and was kept. |
| `rejected` | yes | A step's component failed a criterion; the search is now *stopped* (as in `cup.prewhiten`). |
| `blocked` | yes | The peak could not be fitted there (it collapsed onto an existing component); it is remembered and skipped next time. {meth}`~cuperiod.PreWhitenSession.advance` steps past blocked candidates. |
| `stopped` | yes | The search ended without a new component: no admissible peak left, residuals at numerical round-off, a failed trial fit, or a guard such as the frequency cap or too few degrees of freedom. |
| `edited` | yes | A manual or data operation changed the session. |
| `refused` | no | Bad input, or an action the session cannot take; nothing changed. |
| `noop` | no | The action would change nothing. |

`run(max_steps=, should_stop=, on_step=)` stops after a number of steps, polls a
callback before each step (a stopped run resumes where it left off) and reports each
step as it goes. {meth}`~cuperiod.PreWhitenSession.finalize` is Auto's ending — the final
simultaneous fit and the S/N re-check, per the settings — and returns a
{class}`~cuperiod.PreWhitenResult`; {meth}`~cuperiod.PreWhitenSession.result` reports
the current state without changing it, and `polish()`, `prune()` and `renumber()` are
available on their own.

### One entry point that never raises for bad input

Every typed method is a one-line call of {meth}`~cuperiod.PreWhitenSession.do`, which
validates the arguments, applies the operation, logs it and makes it undoable. Bad input
and impossible actions come back as a `refused` report and change nothing — they never
raise. Only programming errors propagate.

```python
report = session.tie("F9", "F1 + F2")
print(report.kind, report.message)
# refused no component F9

report = session.do("tie", id=3, expression="0.5 F2")
print(report.kind, report.message)
# refused non-integer coefficient '0.5' in tie '0.5 F2'
```

The 25 operations, all available through `do(op, **args)` and as methods of the same
name:

| Class | Operations | Effect on the session |
| --- | --- | --- |
| automatic | `step`, `polish`, `prune`, `renumber` | The automatic loop. A session that only ever ran these is *pristine*. |
| attempt | `add_peak` | Adds the tallest admissible peak without the acceptance test. |
| manual | `add`, `remove`, `set_active`, `fix`, `set_values`, `tie`, `set_variation`, `calculate`, `improve`, `delete_points`, `restore_points`, `exclude_subsets`, `set_subset_weights`, `use_offsets`, `use_variation`, `use_weights`, `configure` | An edit: the model is re-solved with the frequencies held (unless `SessionSettings.recalculate_on_edit` is off), the session is *searching* again and the attempt budget restarts. |
| neutral | `set_note`, `define_partition`, `drop_partition` | Changes nothing the fit sees. |

Components are addressed by id: `1`, `"F1"` or `"f1"`. Ids are stable — the label of
component 7 is `F7` whatever is removed around it — and only `prune` and `renumber`
compact them (rewriting every tie).

### Components by hand

```python
report = session.add(24.6912)          # snapped to the residual peak within one Rayleigh width
print(report.message)
# F4 added at 24.7170104 c/d
session.undo()                         # F4 is gone again
session.redo()                         # ... and back
session.remove("F4")

session.fix("F1", frequency=True)      # hold F1's frequency
with session.group():                  # one undoable action
    session.set_values("F2", amplitude=0.007)
    session.fix("F2", amplitude=True)  # a value you set is re-solved unless it is held
session.undo()
session.set_active("F2", False)        # out of the model, values kept
session.set_active("F2", True)
session.set_note("F1", "dominant radial mode")
session.fix("F1", frequency=False)
```

- `add(frequency, snap=True, refine=None, allow_close=False)` adds a component at the
  residual peak nearest `frequency` (`SessionSettings.snap_rayleigh`, one Rayleigh width
  by default; `snap=False` takes the value as given) and refines it. A frequency closer
  than `collision_rayleigh` (0.25) Rayleigh widths to another component, or a refinement
  that would land on one, is refused unless `allow_close=True`.
- `add_peak()` adds the tallest admissible peak of the residual spectrum without the
  acceptance test — the stopping rules inform, they never block.
- `remove(*ids, cascade=False)` refuses to remove a component that others are tied to;
  `cascade=True` removes it and dissolves those ties.
- `fix(ids, frequency=, amplitude=, phase=)` holds (`True`) or frees (`False`) each
  quantity at its current value; `set_values(id, frequency=, amplitude=, phase=)` writes
  values (a new frequency also unties the row). A held quantity reports an uncertainty of
  `0.0` and the flag `fixed_f`, `fixed_a` or `fixed_phi`.
- `calculate()` re-solves amplitudes, phases and zero points with every frequency held;
  `improve()` refines every free frequency, `improve(["F2", "F5"])` only those, with the
  others held.

A held phase is defined at the session's `t_ref`. With a fixed phase the fitted amplitude
may come out negative: the session and the table keep the signed value (flag
`negative_amplitude`), while the {class}`~cuperiod.Sinusoid` of a result reports
`|A|` with the phase shifted by π, so S/N, pruning and combination searches always read a
non-negative amplitude.

### Ties: combination frequencies that follow their parents

```python
report = session.tie("F3", "F1 + F2")
print(report.message)
# tied F3 = F1 + F2 (30.155827 c/d)
session.tie("F3", None)                # untie
session.tie("F3", "F1 + F2")
```

A tie makes a frequency an integer combination of others: its value is computed from
its parents at every fit, its uncertainty is propagated from theirs, and the row is
reported as `tied`. The grammar:

- terms are component labels with optional integer coefficients: `F1 + F2`, `2F1`,
  `2*f1 - f3`, `F1 - F2`; labels are case-insensitive, and the Unicode minus sign `−`
  is accepted;
- the `F3 = F1 + F2` form is accepted when the left-hand label is the component being
  tied;
- repeated labels merge (`f1 + f1` is `2F1`).

It is parsed by a small tokenizer and **never evaluated**. Refused, with the reason in
the report: a non-integer coefficient (`0.5f1`), a bare number (a tie has no constant),
a missing or trailing operator, a combination that sums to zero (`f1 - f1`), a
coefficient above 99, a total order Σ|nᵢ| above 20, an expression longer than 200
characters, a parent that does not exist, a cycle (`F1 = 2F3` while `F3 = F1 + F2`), a
predicted frequency outside the search band, or — unless `allow_close=True` — one
within `collision_rayleigh` of another component. A tied row's fixed frequency gives way
to the tie. In the result, a tie whose parents are all held reports both `tied` and
`fixed_f`.

### The data: deletions, subsets, zero points and weights

Nothing is ever removed from the session's data: deleting points and leaving subsets out
are records in the state, so they are undoable and saved with the project. All indices
refer to `session.data` — the finite points of the input, in input order.

```python
resid = session.residuals
scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
print(session.delete_points(np.flatnonzero(np.abs(resid) > 5 * scale)).message)
# deleted 2 points (2 deleted in all)

nights = cup.Partition.from_gaps(session.data.time, min_gap=0.3, name="night")
session.define_partition(nights)                 # 40 subsets: N1 ... N40
session.use_offsets("night")                     # one fitted zero point per night

weights = session.subset_rms_weights("night")    # 1/rms² of each night's residuals
with session.group():
    session.set_subset_weights("night", weights)
    session.use_weights("night")

session.improve()
sub = session.result().subsets
print(sub.offsets_by, sub.offset_labels[:3], sub.offset[:3], sub.offset_error[:3])
# night ('N1', 'N2', 'N3') [11.9993 12.0001 12.0003] [0.00019 0.00022 0.00019]

session.exclude_subsets("night", ["N7"])         # leave a whole night out
```

- `delete_points(indices)` takes integer indices, a boolean mask or index runs
  `[(start, stop), ...]`; `restore_points(indices)` brings points back, and
  `restore_points()` restores all of them.
- A {class}`~cuperiod.Partition` assigns every point to a subset and records the recipe
  that made it, so a script can rebuild it: `Partition.from_gaps(time, min_gap=, max_span=)`
  (nights; TESS sectors with `min_gap=0.5, max_span=28.5`), `from_chunks(time,
  length=)`, `from_labels(values, name=)` (a label column such as a site or a band),
  `from_ranges(time, edges)` and `a.crossed(b)`; hand edits (`merged`, `split_at`,
  `renamed`, `with_weights`) are recorded run by run.
  {func}`cuperiod.prewhiten.suggest_partition` proposes the obvious split of a sampling.
  Per-point label columns can be given to the constructor (`labels={"site": ...}`,
  aligned with the input rows): `partitions=["site"]` turns one into a partition at
  creation, and `Partition.from_labels(session.data.labels["site"], name="site")`
  later.
- `exclude_subsets(name, subsets, excluded=True)` leaves whole subsets (indices or
  labels) out of the fit. It is independent of deleted points: re-including a subset
  never resurrects points deleted inside it, and restoring points never re-includes an
  excluded subset.
- Three **roles** take a partition each, independently: `use_offsets(name)` (a zero
  point per subset instead of one mean; needs `fit_mean=True`), `use_weights(name)`
  (the subset weights multiply the inverse-variance weights; set them with
  `set_subset_weights`, and get 1/rms² values from `subset_rms_weights`) and
  `use_variation(name)` (see below). `None` switches a role off. A partition in use by a
  role, or with excluded subsets, cannot be redefined or dropped.
- Zero points come back in `result.subsets` (a {class}`~cuperiod.prewhiten.SubsetSolution`:
  `offset`, `offset_error` and labels per subset) and the number of deleted points in
  `result.n_deleted`.
- `configure(settings=None, **updates)` replaces the requested settings (a manual
  edit). Changing the grid settings (`minimum_frequency`, `maximum_frequency`,
  `nyquist_factor`, `samples_per_peak`) recomputes the grid from the points in use;
  `backend` and `device` cannot change.

A change of the points in use or their weights rebuilds the spectrum engine for the new
sampling (two are kept, so an undo finds the previous one again). Zero points per subset
cost about as much as a single mean even with hundreds of subsets, because they are
projected out of every solve rather than fitted as extra columns.

### Amplitude and phase from subset to subset

```python
seasons = cup.Partition.from_chunks(session.data.time, length=20.0, name="season")
session.define_partition(seasons)

var = session.variations(["F1", "F2"], by="season")     # a query: nothing changes
print(var.subset_labels, var.amplitude.round(5))
# ('C1', 'C2') [[0.01198 0.01199]
#               [0.00701 0.0069 ]]
print(var.amplitude_p_value.round(3), var.phase_p_value.round(3))
# [0.919 0.248] [0.189 0.91 ]

session.use_variation("season")                         # the persistent version
session.set_variation("F2", "amplitude")                # F2 gets one amplitude per season
```

{meth}`~cuperiod.PreWhitenSession.variations` fits the selected components with one
amplitude and/or phase per subset (`mode="both"`, `"amplitude"` or `"phase"`) and returns
a {class}`~cuperiod.VariationResult`: the per-subset values and errors, the common
values, a Wald constancy test per component (χ² with S−1 degrees of freedom, and its
p-value) and the BIC change. The frequencies are held unless
`improve_frequencies=True`, and the errors are **conditional on the held frequencies**: a
slightly wrong frequency shows up as a phase drift, so improve the frequencies first.

`set_variation(ids, mode)` puts per-subset amplitudes and/or phases into the session's
model, on the partition chosen by `use_variation`. `"phase"` and `"both"` also hold the
frequency (`hold_frequency=True`), because a free frequency next to a free phase per
subset is barely determined. A varying component's headline amplitude and phase are the
**coherent** (phasor-mean) values — what the spectrum sees — while its significance
(S/N, `A/Asp`, the `!` mark) uses its **RMS amplitude** `amplitude_rms`, since the
coherent mean shrinks when the phases wander; varying components are never pruned.
Per-subset values are in `result.subsets.amplitude` / `.phase`.

### Undo, redo and the history

```python
session.undo()                 # one action, or one group
session.redo()
for entry in session.history[-3:]:
    print(entry.seq, entry.op, "|", entry.report.message)
# 18 improve | improved all 3 frequencies
# 19 exclude_subsets | excluded N7 of 'night'
# 20 define_partition | defined partition 'season' (2 subsets)

session.goto(3)                                  # the state after log entry 3 (0 = start)
session.goto(session.log_entries[-1].seq)        # and back to the newest
```

`history` is the effective log up to the current state; `log_entries` also holds the
redo tail and `history_cursor` says where the current state is. `group()` makes the
actions inside one undo step (as `run()` and `finalize()` do). The undo depth is
`SessionSettings.undo_depth` (200 states); `reachable(seq)` says whether a state is still
in memory — after loading a project only the stored state is, and the rest of the log is
re-created by replaying it. Doing anything after an undo discards the redo tail.

### Finishing

```python
result = session.finalize()
print(result.summary())
```

```text
Pre-whitening: 3 components from 3391 points over 39.28 d  (backend=numpy)
  stopped: interactive session (not run to a stopping rule)
  residual rms 0.00196855   reduced chi2 0.9601   D=1.01   errors: covariance

  ID     frequency (1/d)         +/-     amplitude         +/-    phase     S/N   A/Asp  note
  F1          12.3455891    5.28e-05     0.0119896    4.79e-05   2.0201  192.21    1.00
  F2          17.8100563    8.63e-05    0.00695561    4.79e-05   3.4400  118.91    0.99
  F3          30.1556454    9.56e-05    0.00299675    4.79e-05   6.0755   55.04    0.95  F3 = F1 + F2
```

With one zero point per night the reduced χ² drops from 4.9 to 0.96, the residuals are
white again (`D` from 2.1 to 1.0) and the uncertainties shrink threefold.

The result is the usual {class}`~cuperiod.PreWhitenResult`. Each
{class}`~cuperiod.Sinusoid` also carries `fixed` (which quantities were held), `tied`,
`variation`, `amplitude_rms` and `flags` — `fixed_f`, `fixed_a`, `fixed_phi`, `tied`,
`negative_amplitude`, `degenerate`, `at_bound`, `out_of_band`, `unresolved`,
`low_frequency`, `weak_frequency`, `short_subset`. A parameter that cannot be determined
reports `NaN`. `to_table(constraints=None)` adds the constraint columns only when a
component carries a constraint, so an automatic solution keeps the v1.3 table layout.

### Saving, verifying and exporting

```python
session.save("star.cupw.json")
again = cup.PreWhitenSession.load("star.cupw.json", verify=True)

session.export_script("star_session.py")         # plus star_session.data.csv here
copy = cup.PreWhitenSession.replay((time, mag, err), session.history, settings=settings)
```

- `save(path, embed_data="auto")` writes the project atomically. With `"auto"` the data
  file is *referenced* when it sits in the project's folder and still holds exactly
  these data (and the session has no label columns), otherwise the data are *embedded*;
  `"always"` / `"never"` force either.
- `load(path, verify=False, data=None, backend=None, allow_modified_data=False)` opens a
  project at its stored state (no replay). `verify=True` first replays the log on a fresh
  session and raises {class}`~cuperiod.ProjectError` unless the stored state follows
  from it (frequencies to 1e-9 relative, amplitudes and phases to 1e-6). `data=` supplies
  the light curve when the referenced file has moved; it is still checked against the
  stored content hash, unless `allow_modified_data=True`, which re-fits the stored
  solution to the new data. `backend=None` uses the recorded backend when this machine
  has it, else the best CPU backend (with a {class}`~cuperiod.SessionWarning`).
- `replay(data, log, **create)` creates a new session and applies a log entry by entry;
  an entry whose outcome differs from the log raises {class}`~cuperiod.ProjectError`
  naming it. Steps reuse their logged peak seeds, so a replay is **bitwise identical on
  any spectrum backend** — a session run on `finufft` verifies on `numpy`.
- `to_script()` returns, and `export_script(path)` writes, a Python script that uses only
  the public API: it loads the data (from the source file, or a `*.data.csv` written
  next to the script when the data have no re-readable source), checks their SHA-256,
  repeats every action — consecutive steps fold into one `session.run(seeds=[...])` —
  prints the solution and says whether it reproduced the session. Every value in the
  script is a Python literal and every free text (notes, titles, labels) is sanitised
  into comments, so a project's content can never become code.
  {meth}`~cuperiod.PreWhitenSession.from_csv` reads such a data file back.
- `snapshot()` and `restore(snapshot, data)` keep and re-attach a session in memory —
  the head and the undo states exactly as they were, without a re-run. This is how the
  GUI shows a cached solution together with its session.

## The project file

A `*.cupw.json` project is versioned JSON (`"format": "cuperiod-prewhiten-project"`,
`"version": 1`) holding:

- `data` — the light curve, **embedded** (time, value, error and label columns) or
  **referenced** by a path relative to the project (with the column mapping), and in
  both cases a SHA-256 of the arrays the session used;
- `creation` — what the session was created with: the backend, `t_ref`, the frozen
  grid, the requested settings, the {class}`~cuperiod.SessionSettings` and the root
  partitions;
- `state` — the current solution: components and their constraints, zero points, the
  point mask, partitions, excluded subsets, roles, settings and status;
- `log` — the effective action log, each entry with the arguments it really used (a
  step's peak seed and grid index, a snapped frequency) and its outcome;
- `title`, `saved`, `created_with` and a free-form `ui` object for the GUI.

The undo stack is not stored: after `load` the stored state is the only one in memory,
and `verify` (or the GUI's *Verify project*) re-creates the rest by replaying the log.

**Opening a project never executes anything from it**, so a project from someone else
is safe to open:

- it is read as JSON only — no pickle, no `eval`; `NaN`, `Infinity`, numbers that
  overflow a float and duplicate keys are refused;
- its size is checked before it is read (`SessionSettings.max_project_bytes`, 64 MiB),
  the log length is capped (`max_actions`, 100 000) and so are the number of points and
  the grid size;
- every object is validated by a strict model (unknown fields refused, no type
  coercion), and every cross reference — component ids, ties, partitions, roles, the
  point mask — is checked against the data before the session exists;
- a data reference must stay inside the project's folder;
- the data must match the stored content hash;
- tie expressions are parsed, never evaluated, and exported scripts carry free text only
  as sanitised comments;
- whatever is wrong surfaces as {class}`~cuperiod.ProjectError` naming the field, and as
  nothing else:

```python
# a project whose first frequency was replaced by a string of code
cup.PreWhitenSession.load("bad.cupw.json")
# ProjectError: bad.cupw.json: state.components.0.frequency: Input should be a valid number
```

Older schema versions are migrated on reading; a migration that meets a missing
settings field writes that version's default, never a value from the environment.

## From the command line

`cuperiod prewhiten` can keep its automatic run as a project, or as a script:

```bash
cuperiod prewhiten star.csv --fmax 50 --save-project star.cupw.json --export-script star_session.py
```

and the `cuperiod session` group works on saved projects — written by the CLI, the
library or the GUI:

```bash
cuperiod session verify star.cupw.json
# Verified star.cupw.json: its 5 logged action(s) reproduce the stored solution
# (3 active component(s), backend numpy).

cuperiod session replay star.cupw.json -o replayed.cupw.json   # re-create it, print the solution
cuperiod session script star.cupw.json -o star_session.py      # export a reproducing script
```

`verify` and `replay` exit with status 1 and a one-line error when the project cannot
be read or its log does not reproduce the stored solution. Each
command takes `--data FILE` to use a light-curve file in place of the project's own data
(the content hash is still checked), and `verify` / `replay` take `--backend`. See
{doc}`cli` for every option.

## Settings of a session

{class}`~cuperiod.SessionSettings` (environment prefix `CUPERIOD_SESSION_`) decides how a
session reacts to edits and what it keeps — never a number of an automatic run:

| Setting | Default | What it controls |
| --- | --- | --- |
| `recalculate_on_edit` | `True` | Re-solve amplitudes, phases and zero points after every edit. |
| `protect_manual` | `True` | `prune` never removes manual, fixed, tied or tie-parent components. With `False`, pruning a tie parent dissolves its children's ties. |
| `undo_depth` | 200 | States kept in memory for undo. |
| `derived_cache` / `engine_cache` | 8 / 2 | Fits and residual spectra, and spectrum engines, kept per session. |
| `collision_rayleigh` | 0.25 | A manual frequency closer than this (Rayleigh widths) to another is refused. |
| `snap_rayleigh` | 1.0 | Search radius of a snapped manual frequency. |
| `min_subset_points` | 10 | Fewest points a variation subset may have. |
| `embed_data_max_points` | 200 000 | Largest light curve a project embeds automatically. |
| `max_actions`, `max_project_bytes` | 100 000, 64 MiB | Limits of what a project may hold. |

Pass them as `PreWhitenSession(..., session_settings=cup.SessionSettings(...))`.

## Also good to know

- **Multi-band data.** {meth}`PreWhitenSession.from_bands(mblc) <cuperiod.PreWhitenSession.from_bands>`
  pre-whitens all bands of a {class}`~cuperiod.MultiBandLightCurve` merged — each band's
  mean removed first, exactly as the GUI's stacked view does — with a root `band`
  partition, so `use_offsets("band")` fits a zero point per band. The merge itself is
  `cuperiod.core.lightcurve.stack_bands`.
- **Threads.** A session is used from one thread at a time: a call from a second
  thread while one is running raises {class}`~cuperiod.SessionError`. The GUI runs each
  session on its worker thread.
- **GPUs.** `session.release()` drops device-side plans and arrays; they are rebuilt on
  demand.
- **Spectra.** `session.spectrum()` (the data spectrum of the points in use, zero points
  removed), `residual_spectrum()` and `window()` return
  {class}`~cuperiod.AmplitudeSpectrum` objects; `AmplitudeSpectrum.noise_curve()` gives
  the local noise level at every grid frequency, which times `snr_threshold` is the S/N
  limit the GUI draws.
- **Fits without a session.** The constraint model is available directly:
  {func}`~cuperiod.fit_multisine` takes `constraints=` (a {class}`~cuperiod.FitConstraints`
  of {class}`~cuperiod.ComponentConstraint` and {class}`~cuperiod.FrequencyTie` records,
  zero-point and variation group codes, and relative point weights); unconstrained calls
  are unchanged.

```python
nights = cup.Partition.from_gaps(time, min_gap=0.3)
constraints = cup.FitConstraints(
    components=(
        cup.ComponentConstraint(),                 # F1 free
        cup.ComponentConstraint(),                 # F2 free
        cup.ComponentConstraint(frequency=cup.FrequencyTie.from_text("F1 + F2")),
    ),
    offset_groups=nights.codes,                    # one zero point per night
)
fit = cup.fit_multisine(time, mag, err, np.array([12.3456, 17.81, 30.1556]),
                        constraints=constraints)
fit.detail.flags          # (frozenset(), frozenset(), frozenset({'tied'}))
fit.detail.subset_offset  # the nightly zero points
```

## References

- Lenz, P., & Breger, M. 2005, CoAst 146, 53 — Period04.
- Breger, M., et al. 1993, A&A 271, 482 — the S/N ≥ 4 criterion.
- Montgomery, M. H., & O'Donoghue, D. 1999, DSSN 13, 28 — analytic uncertainties.
