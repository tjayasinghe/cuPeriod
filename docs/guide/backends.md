# Backends & performance

Every method has more than one *backend* — a concrete implementation of the same math.
You select among them with the `backend=` argument; the result is identical to
floating-point round-off regardless of which one runs.

## Selecting a backend

```python
pg = cup.periodogram(lc, "GLS", backend="auto")    # default
```

The selectors:

| `backend=` | Meaning |
| --- | --- |
| `"auto"` *(default)* | The NVIDIA CUDA fast path when a CUDA device and the `[gpu]` extra are present; then the portable **torch** backend on another GPU (AMD/Intel/Mac) when the `[torch]` extra sees one; otherwise the best CPU backend. The same code runs on any machine. |
| `"cpu"` | Force the best CPU backend for this method. |
| `"gpu"` | Force a GPU backend — the CUDA fast path, or torch on a non-NVIDIA GPU. Raises {exc}`~cuperiod.BackendUnavailableError` if no GPU is usable. |
| `"torch"` / `"torch:<device>"` | Force the portable PyTorch backend; `<device>` is `cpu`, `cuda`, `mps`, or `xpu` (bare `"torch"` picks the best present). Needs the `[torch]` extra. |
| a concrete name | Force a specific implementation, e.g. `"finufft"`, `"astropy"`, `"numpy"`, `"cupy"`, `"cufinufft"`, `"numba"`. |

## What each method can run

```{list-table}
:header-rows: 1
:widths: 16 30 26 28

* - Method
  - CPU backends
  - GPU backend
  - `cpu` resolves to
* - GLS
  - `finufft`, `astropy`
  - `cufinufft`
  - `finufft`
* - BLS
  - `numba` *(with `[fast]`)*, `astropy`, `numpy`†
  - `cupy`
  - `numba` if installed, else `astropy`
* - PDM
  - `numba` *(with `[fast]`)*, `numpy`
  - `cupy`
  - `numba` if installed, else `numpy`
* - CE
  - `numba` *(with `[fast]`)*, `numpy`
  - `cupy`
  - `numba` if installed, else `numpy`
* - String-Length
  - `numba` *(with `[fast]`)*, `numpy`
  - `cupy`
  - `numba` if installed, else `numpy`
* - MHAOV
  - `numba` *(with `[fast]`)*, `numpy`
  - `cupy`
  - `numba` if installed, else `numpy`
* - TLS
  - `numba` *(with `[fast]`)*, `numpy`
  - `cupy`
  - `numba` if installed, else `numpy`
```

† BLS's `numpy` backend is a GPU-parity *reference*, not the product path — it shares one
array-module-generic source with the CUDA kernel so the two validate to floating point,
but it is slow (it trades memory traffic for the parallelism that makes the GPU fast).
For CPU BLS use `numba` (the default with `[fast]`) or `astropy`, **not** `numpy`.

In addition, **every method has a portable `torch` backend** (the `[torch]` extra) that
runs the same array-API code on any torch device — see below.

## The portable PyTorch backend

`backend="torch"` runs a method through the [array API](https://data-apis.org/array-api/)
on whichever torch device you have — CUDA, AMD **ROCm**, Apple **MPS**, Intel **XPU**, or
**CPU** — so the accelerated code is no longer NVIDIA-only and works even with no GPU at
all. The NVIDIA fast paths (cufinufft for GLS, the cupy kernels for BLS/PDM/CE/TLS) are
untouched and remain what `"auto"`/`"gpu"` pick on CUDA; torch is the cross-vendor path
for everything else.

```python
pg = cup.periodogram(lc, "GLS", backend="torch")        # best torch device
pg = cup.periodogram(lc, "BLS", backend="torch:xpu")    # pin the Intel GPU
pg = cup.periodogram(lc, "PDM", backend="torch:cpu", settings=cup.PDMSettings(device="cpu"))
```

Two orthogonal settings tune it (both environment-overridable, e.g. `CUPERIOD_GLS_DEVICE`):

- **`device`** — `"auto"` (default; the best present), `"cpu"`, `"cuda"`, `"mps"`, `"xpu"`.
  A `"torch:<device>"` backend string overrides it.
- **`precision`** — `"auto"` (default) is float64 everywhere it is supported and float32
  only where the device forces it (Apple MPS; some Intel GPUs). `"float64"` and `"float32"`
  force it; `precision="float64"` on MPS raises rather than silently downgrading. Results
  are always returned as float64 numpy arrays regardless of the device precision.

:::{note}
The portable path is correct everywhere (it matches the CPU reference to round-off — float
methods bit-for-bit, MHAOV to ~1e-10 from its linear solve) but it is **not** the speed
champion on CPU, where finufft (GLS) and the numba box search (BLS) are faster. Its value
is reaching GPUs the CUDA fast paths can't — AMD, Intel, and Apple. On those devices its
massively-parallel evaluation is the win. Run `cuperiod doctor` to see what you have.
:::

Inspect the live picture for your install:

```python
for info in cup.list_methods():
    print(info.name, "| all:", info.all_backends, "| available:", info.available_backends)
```

## What's fast where

The headline from the {doc}`../benchmarks` (single light curve, ~900 points; an
RTX 5070 Ti vs the CPU backends):

```{list-table}
:header-rows: 1
:widths: 15 18 14 14 14 15

* - Method
  - CPU backend
  - CPU time
  - GPU time
  - torch:cuda
  - GPU speed-up
* - GLS
  - finufft
  - 0.016 s
  - 0.007 s
  - 0.013 s
  - ~2×
* - BLS
  - numba
  - 0.195 s
  - 0.092 s
  - 0.391 s
  - ~2×
* - PDM
  - numpy
  - 1.89 s
  - 0.010 s
  - 0.016 s
  - **186×**
* - CE
  - numpy
  - 0.585 s
  - 0.012 s
  - 0.011 s
  - 49×
* - String-Length
  - numpy
  - 1.41 s
  - 0.027 s
  - 0.011 s
  - 51×
* - MHAOV
  - numpy
  - 3.30 s
  - 0.062 s
  - 0.048 s
  - 53×
* - TLS
  - numpy
  - 4.78 s
  - 0.044 s
  - 2.42 s
  - 108×
```

How to read this:

- **GLS and BLS already have specialized CPU backends** (finufft; the multicore numba box
  search). They're so fast on the CPU that the GPU's marginal gain on a single curve is
  modest — but the GPU still wins decisively for **large grids and big catalogs**
  ({doc}`batch`).
- The **CPU times in the table are the vectorized numpy paths**. With the `[fast]` extra
  installed, PDM, CE, String-Length, MHAOV, and TLS now run multicore **numba** kernels
  on `"cpu"`/`"auto"` — one to two orders of magnitude faster than the numpy column
  (e.g. PDM ~300×, CE ~135×, MHAOV ~57× on a 3k-point curve) — which puts a warm CPU
  within reach of the GPU for a *single* curve. The GPU remains the throughput champion
  for large grids and catalogs.
- On **consumer NVIDIA cards** (GeForce), whose float64 throughput is 1/64 of float32,
  the opt-in `precision="float32"` runs the BLS/TLS CUDA kernels ~8-9× faster at
  detection-grade accuracy; float64 stays the default.
- cuPeriod's **CPU** path already beats the established reference tools it was checked
  against (GLS ~3× astropy, PDM ~4× PyAstronomy, BLS ~18× astropy's `BoxLeastSquares`).

:::{tip}
Write `backend="auto"` and let cuPeriod choose. Reach for `"cpu"`/`"gpu"` or a concrete
name only to benchmark, to pin a reference implementation, or to keep a long-running job
off a busy GPU.
:::

## When the GPU isn't available

`backend="auto"` silently falls back to the CPU, so portable code "just works."
`backend="gpu"` is explicit and raises {exc}`~cuperiod.BackendUnavailableError` when no
CUDA device or the `[gpu]` extra is missing — use it when you *want* to fail loudly rather
than run on the CPU by accident.

```python
try:
    pg = cup.periodogram(lc, "TLS", backend="gpu")
except cup.BackendUnavailableError as e:
    print("no GPU:", e)
```

---

Next: {doc}`tuning` — settings and custom grids.
