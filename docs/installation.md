# Installation

cuPeriod runs on Python 3.11+ and installs from PyPI. The base install is pure-CPU and
cross-platform; optional extras add GPU acceleration and a faster CPU box search.

## Base install (CPU)

```bash
pip install cuperiod
```

This pulls in the core dependencies — `numpy`, `scipy`, `astropy`, `finufft`,
`array-api-compat`, `pydantic`, `pydantic-settings`, `typer`, and `pyarrow` (at least
14.0.1: earlier releases unpickle extension-type metadata while reading Parquet,
CVE-2023-47248, and cuPeriod opens user-supplied `.parquet` files) — and gives you every
method on the CPU, the command line, and batch processing over a process pool.

## Optional extras

| Extra | Command | Adds |
| --- | --- | --- |
| **gpu** | `pip install "cuperiod[gpu]"` | CUDA-12 GPU backends (`cupy-cuda12x`, `cufinufft`, the NVIDIA runtime wheels) |
| **torch** | `pip install "cuperiod[torch]"` | the portable **PyTorch** backend — runs every method on AMD (ROCm), Intel (XPU), Apple (MPS), and a CPU path |
| **fast** | `pip install "cuperiod[fast]"` | multicore `numba` CPU kernels for BLS, PDM, CE, String-Length, MHAOV, TLS, and SuperSmoother — the CPU default when installed, and much faster than the fallback CPU paths for all of them but MHAOV |
| **gui** | `pip install "cuperiod[gui]"` | the interactive desktop GUI, `cuperiod-gui` (PySide6 + pyqtgraph) — see {doc}`guide/gui`. On Python 3.11 it keeps PySide6 below 6.12 (see the note below) |
| **plot** | `pip install "cuperiod[plot]"` | matplotlib, for the period-change diagnostic figures: {func}`cuperiod.pdot.plots.pdot_figure` and the rest of `cuperiod.pdot.plots`, the `plots=` option of {func}`~cuperiod.batch_pdot`, and `cuperiod pdot-batch --plots` — see {doc}`guide/period-change` |
| **pandas** | `pip install "cuperiod[pandas]"` | pandas `DataFrame` ingestion |
| **nested** | `pip install "cuperiod[nested]"` | nested-pandas `NestedFrame` light curves, one row per object — see {doc}`guide/interop` |
| **lsdb** | `pip install "cuperiod[lsdb]"` | the same adapter over lazy, dask-partitioned lsdb HATS catalogs |

Extras combine, e.g. `pip install "cuperiod[gpu,fast]"` or `"cuperiod[gui,torch]"`.

:::{note}
**The GUI on Python 3.11.** PySide6 6.12 drops a reference to `None` whenever a Qt call
returns a null pointer. From Python 3.12 `None` is immortal and nothing happens; on 3.11
its reference count reaches zero within seconds of normal use and the interpreter aborts
("deallocating None"). The `[gui]` extra therefore installs `PySide6<6.12` on Python 3.11
(any PySide6 from 6.7 on 3.12+), and `cuperiod-gui` on 3.11 refuses to start under a newer
PySide6 already in the environment, naming the fix: `pip install 'PySide6<6.12'`, or run
it on Python 3.12 or newer.
:::

:::{tip}
The `[fast]` extra is worth installing even without a GPU: it gives BLS, PDM, CE,
String-Length, MHAOV, TLS, and SuperSmoother multicore JIT kernels that return the
fallback CPU paths' results, much sooner for the box and fold methods. When present they
become the default CPU backends automatically.
:::

### What the `[fast]` extra changes

Installing `numba` flips `backend="cpu"` (and the CPU fallback of `"auto"`) from the
fallback implementations — astropy's `BoxLeastSquares` for BLS, the vectorized numpy
kernels for PDM, CE, String-Length, MHAOV, TLS, and SuperSmoother — to in-house multicore
JIT kernels. Nothing else about your code changes: the results match the fallbacks (to
round-off, and within ~1e-10 for MHAOV; BLS still matches astropy), they just arrive
sooner. How much sooner depends on the method. On the {doc}`benchmarks` page the numba
BLS runs 89× faster than astropy's `BoxLeastSquares`; PDM, CE, String-Length, TLS and
SuperSmoother gain the most over their vectorized numpy paths, while MHAOV's numpy path
already sums its harmonics with a NUFFT on the automatic (uniform) grids and runs at
about the numba kernel's speed. (Note that `numba` currently caps `numpy < 2.5`, so installing it may
downgrade numpy slightly; the extra also asks for `scipy >= 1.13`, because the OpenBLAS
bundled with older scipy can crash the process at exit on many-core machines once numba's
thread pool is running.) The kernels are compiled once per install and cached on disk;
files named `supersmoother._numba_ss_kernel…` left in `cuperiod/methods/__pycache__` by
earlier versions, which recompiled in every process, can be deleted.

:::{note}
**numba's cache on a long Windows path.** numba writes its cache next to the installed
package (`cuperiod\methods\__pycache__`), and cuPeriod's kernel names are long. Installed
under a deep folder (a long project or OneDrive path) with Windows long paths disabled,
those file names pass the 260-character limit, and the first call of each numba kernel
in a process fails with `FileNotFoundError` naming a `*.nbi.tmp` or `*.nbc.tmp` file — in
a batch, every worker hits it. Set `NUMBA_CACHE_DIR` to a short
folder (for example `C:\numba-cache`), install into a shorter path, or enable Windows long
paths (the `LongPathsEnabled` registry value, or the *Enable Win32 long paths* policy).
:::

Backends are probed once per process. If you install a GPU or torch extra into a running
session (a notebook, say), call `cuperiod.core.backend.refresh_backends()` to probe again.

## GPU requirements

GPU acceleration needs:

- an **NVIDIA GPU** with the **CUDA 12** runtime,
- the `[gpu]` extra, which installs `cupy-cuda12x`, `cufinufft`, and the
  `nvidia-*-cu12` runtime wheels (no system CUDA toolkit required).

All eight methods have a GPU backend. With the extra installed and a device present,
`backend="auto"` (the default) uses the GPU and falls back to the CPU otherwise — so the
same code runs on both. See {doc}`guide/backends`.

:::{note}
On Windows, the cufinufft wheel needs the CUDA-12 runtime DLLs on the search path.
cuPeriod adds the `nvidia-*-cu12` wheel directories automatically when a GPU backend is
first used — only directories of the installed wheels, never a folder that merely happens
to be called `nvidia` — so no manual `PATH` editing is needed. cupy may print a benign
*"CUDA path could not be detected"* warning when it uses the pip wheels rather than a
system toolkit — this is harmless.
:::

## Portable GPU backend (PyTorch)

The `[torch]` extra adds a **PyTorch backend** that runs every method beyond NVIDIA — on
AMD (ROCm), Intel (XPU), and Apple-Silicon (MPS) GPUs, and on a CPU path everywhere (handy
even with no GPU at all):

```bash
pip install "cuperiod[torch]"
```

The plain wheel above is **CPU-only**. To use a GPU, install the PyTorch build matching
your accelerator from the [official index](https://pytorch.org/get-started/locally/) — we
deliberately don't pin a hardware-specific wheel:

| Hardware | PyTorch build |
| --- | --- |
| NVIDIA (CUDA) | the CUDA wheel — or just use the `[gpu]` extra's faster cufinufft/cupy paths |
| AMD (ROCm) | the ROCm wheel (Linux only) |
| Intel (XPU) | the XPU wheel (`torch.xpu`) |
| Apple Silicon | the standard macOS wheel (MPS is built in) |
| CPU only | the default wheel |

Select it with `backend="torch"` (or `"torch:cpu"`, `"torch:cuda"`, `"torch:mps"`,
`"torch:xpu"`); `backend="auto"` reaches a torch GPU on non-NVIDIA machines after the
cufinufft/cupy fast paths. See {doc}`guide/backends`.

:::{note}
**Apple MPS** cannot compute in float64 (a Metal limitation), so the Mac-GPU path uses
float32. `precision="auto"` (the default) keeps float64 everywhere it is supported and
drops to float32 only where the device forces it (MPS); an explicit
`precision="float64"` on MPS raises rather than silently downgrading.
:::

:::{warning}
**Windows OpenMP clash.** PyTorch and NumPy/SciPy (MKL) each ship an OpenMP runtime, and
importing torch after numpy can abort with *"OMP: Error #15 … libiomp5md.dll already
initialized."* cuPeriod does **not** set a workaround for you — it can silently affect
numerical results. If you hit this running a torch workload on Windows, set
`KMP_DUPLICATE_LIB_OK=TRUE` in your environment, or install torch and numpy builds that
share one OpenMP runtime. (`cuperiod doctor` sets it only for its own read-only device
probe.)
:::

## Verifying the install

List the registered methods and every backend each one can run:

```bash
cuperiod methods
```

Check whether a GPU is visible and how many batch workers it would suggest:

```bash
cuperiod gpu-info
```

For the full picture — every installed backend, the available torch devices and the
precision each will use, and what `backend="auto"` resolves to for every method — run:

```bash
cuperiod doctor
```

If no CUDA device is present, `gpu-info` says so and exits cleanly — the CPU paths still
work. A backend asked for by name that cannot run here says which extra provides it:
`backend="numba"` without numba names `pip install 'cuperiod[fast]'`, and `"cupy"` or
`"cufinufft"` without a CUDA device or the GPU packages name `pip install 'cuperiod[gpu]'`.
From Python:

```python
import cuperiod as cup

print(cup.__version__)
print([m.name for m in cup.list_methods()])
print(cup.gpu_info())          # a GpuInfo, or None if there's no usable GPU
```

## Development install

To work on cuPeriod itself (tests, linting, type-checking) the project uses
[uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q -m "not gpu"   # the CPU suite; never touches a GPU
uv run pytest -q                # everything; GPU tests skip without a device
```

GPU tests carry the `gpu` marker. `-m "not gpu"` deselects them **and** hides the GPUs
from the rest of the run (`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES=-1` and CPU-only
device discovery), so tests that use the default `backend="auto"` or a bare `"torch"` run
on the CPU and the run never loads a GPU that is busy with other work. A plain `pytest`
lets those tests use a visible GPU; with the GPU stack installed but no device, the GPU
tests skip instead of failing.

To build these docs locally, install the `docs` extra and run Sphinx (the build treats
every warning as an error, as Read the Docs does):

```bash
uv pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

The GUI's keyboard-shortcut tables are generated from its command registry at build
time; the `gui` extra is not needed for that.

### Releasing

1. Cut the version's metadata and merge. Every file that names the version changes
   together:
   - `version` in `pyproject.toml`, and the source-tree fallback `__version__` in
     `src/cuperiod/__init__.py`;
   - `uv.lock`, refreshed with `uv lock` so its `cuperiod` entry carries the new version;
   - `version` and `date-released` in `CITATION.cff`;
   - `CHANGELOG.md`: retitle `[Unreleased]` to `[<version>] - <date>` and add its
     compare link (`[<version>]: …/compare/v<previous>...v<version>`) at the bottom;
   - the README's citation note, which says what each release added.
2. Push the tag `v<version>`. The release workflow builds the package and refuses to
   publish when the tag does not equal the built version (tagging `v1.3.0` while
   `pyproject.toml` still says `1.3.0.dev0`, for example).
3. For a dry run, start the workflow by hand (`workflow_dispatch`): a manual run only
   ever publishes to TestPyPI, even when started on a tag.

The `pypi` deployment environment should require a reviewer and allow only `v*` tags.
Every GitHub Action is pinned to a full commit SHA and Dependabot moves those pins; the
rest is bumped by hand: the uv pin in `release.yml` is two inputs that change together —
`version` and `checksum`, the SHA-256 of `uv-x86_64-unknown-linux-gnu.tar.gz` for that
version (published beside it as `….tar.gz.sha256`), keeping the trailing
`# uv-x86_64-unknown-linux-gnu.tar.gz <version>` comment that `tests/test_supply_chain.py`
checks — and the twine pin with its `--exclude-newer` date and the hash-pinned hatchling
build backend are refreshed with the `uv pip compile` command in the header of
`.github/build-constraints.txt`. A stale checksum fails the release build; it never
passes silently.

Next: the {doc}`quickstart`.
