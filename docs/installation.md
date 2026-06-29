# Installation

cuPeriod runs on Python 3.11+ and installs from PyPI. The base install is pure-CPU and
cross-platform; optional extras add GPU acceleration and a faster CPU box search.

## Base install (CPU)

```bash
pip install cuperiod
```

This pulls in the core dependencies — `numpy`, `scipy`, `astropy`, `finufft`,
`pydantic`, `typer`, and `pyarrow` — and gives you every method on the CPU, the command
line, and batch processing over a process pool.

## Optional extras

| Extra | Command | Adds |
| --- | --- | --- |
| **gpu** | `pip install "cuperiod[gpu]"` | CUDA-12 GPU backends (`cupy-cuda12x`, `cufinufft`, the NVIDIA runtime wheels) |
| **fast** | `pip install "cuperiod[fast]"` | a multicore `numba` box search — BLS's CPU default, ~20× faster than astropy's compiled `BoxLeastSquares` |
| **pandas** | `pip install "cuperiod[pandas]"` | pandas `DataFrame` ingestion |

Extras combine, e.g. `pip install "cuperiod[gpu,fast]"`.

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

## Verifying the install

List the registered methods and the backends available in your environment:

```bash
cuperiod methods
```

Check whether a GPU is visible and how many batch workers it would suggest:

```bash
cuperiod gpu-info
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
