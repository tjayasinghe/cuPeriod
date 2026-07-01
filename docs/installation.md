# Installation

cuPeriod runs on Python 3.11+ and installs from PyPI. The base install is pure-CPU and
cross-platform; optional extras add GPU acceleration and a faster CPU box search.

## Base install (CPU)

```bash
pip install cuperiod
```

This pulls in the core dependencies — `numpy`, `scipy`, `astropy`, `finufft`,
`array-api-compat`, `pydantic`, `typer`, and `pyarrow` — and gives you every method on the
CPU, the command line, and batch processing over a process pool.

## Optional extras

| Extra | Command | Adds |
| --- | --- | --- |
| **gpu** | `pip install "cuperiod[gpu]"` | CUDA-12 GPU backends (`cupy-cuda12x`, `cufinufft`, the NVIDIA runtime wheels) |
| **torch** | `pip install "cuperiod[torch]"` | the portable **PyTorch** backend — runs every method on AMD (ROCm), Intel (XPU), Apple (MPS), and a CPU path |
| **fast** | `pip install "cuperiod[fast]"` | a multicore `numba` box search — BLS's CPU default, ~20× faster than astropy's compiled `BoxLeastSquares` |
| **gui** | `pip install "cuperiod[gui]"` | the interactive desktop GUI, `cuperiod-gui` (PySide6 + pyqtgraph) — see {doc}`guide/gui` |
| **pandas** | `pip install "cuperiod[pandas]"` | pandas `DataFrame` ingestion |

Extras combine, e.g. `pip install "cuperiod[gpu,fast]"` or `"cuperiod[gui,torch]"`.

:::{tip}
The `[fast]` extra is worth installing even without a GPU: it makes BLS an order of
magnitude faster on the CPU while matching astropy to floating point. When present it
becomes BLS's default CPU backend automatically; otherwise BLS falls back to astropy.
:::

### What the `[fast]` extra changes

Installing `numba` flips `backend="cpu"` for BLS from astropy's `BoxLeastSquares` to the
in-house multicore box search. Nothing else about your code changes — the results match
astropy to round-off; they just arrive ~20× sooner. (Note that `numba` currently caps
`numpy < 2.5`, so installing it may downgrade numpy slightly.)

## GPU requirements

GPU acceleration needs:

- an **NVIDIA GPU** with the **CUDA 12** runtime,
- the `[gpu]` extra, which installs `cupy-cuda12x`, `cufinufft`, and the
  `nvidia-*-cu12` runtime wheels (no system CUDA toolkit required).

All seven methods have a GPU backend. With the extra installed and a device present,
`backend="auto"` (the default) uses the GPU and falls back to the CPU otherwise — so the
same code runs on both. See {doc}`guide/backends`.

:::{note}
On Windows, the cufinufft wheel needs the CUDA-12 runtime DLLs on the search path.
cuPeriod adds the `nvidia-*-cu12` wheel directories automatically when a GPU backend is
first used, so no manual `PATH` editing is needed. cupy may print a benign
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
drops to float32 only where the device forces it (MPS, and some Intel GPUs); an explicit
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

List the registered methods and the backends available in your environment:

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
work. From Python:

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
uv run pytest -q          # GPU tests auto-skip without a device
```

To build these docs locally, install the `docs` extra and run Sphinx:

```bash
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Next: the {doc}`quickstart`.
