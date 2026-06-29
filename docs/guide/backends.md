# Backends & performance

Every method has more than one *backend* — a concrete implementation of the same math.
You select among them with the `backend=` argument; the result is identical to
floating-point round-off regardless of which one runs.

## Selecting a backend

```python
pg = cup.periodogram(lc, "GLS", backend="auto")    # default
```

The four selectors:

| `backend=` | Meaning |
| --- | --- |
| `"auto"` *(default)* | Use the GPU when a CUDA device and the `[gpu]` extra are present; otherwise the best CPU backend. The same code runs on any machine. |
| `"cpu"` | Force the best CPU backend for this method. |
| `"gpu"` | Force the GPU backend. Raises {exc}`~cuperiod.BackendUnavailableError` if no GPU is usable. |
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
  - `numpy`
  - `cupy`
  - `numpy`
* - CE
  - `numpy`
  - `cupy`
  - `numpy`
* - String-Length
  - `numpy`
  - `cupy`
  - `numpy`
* - MHAOV
  - `numpy`
  - `cupy`
  - `numpy`
* - TLS
  - `numpy`
  - `cupy`
  - `numpy`
```

† BLS's `numpy` backend is a GPU-parity *reference*, not the product path — it shares one
array-module-generic source with the CUDA kernel so the two validate to floating point,
but it is slow (it trades memory traffic for the parallelism that makes the GPU fast).
For CPU BLS use `numba` (the default with `[fast]`) or `astropy`, **not** `numpy`.

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
:widths: 16 24 20 20 20

* - Method
  - CPU backend
  - CPU time
  - GPU time
  - GPU speed-up
* - GLS
  - finufft
  - 0.019 s
  - 0.007 s
  - ~3×
* - BLS
  - numba
  - 0.187 s
  - 0.091 s
  - ~2×
* - PDM
  - numpy
  - 1.79 s
  - 0.010 s
  - **178×**
* - CE
  - numpy
  - 0.53 s
  - 0.012 s
  - 45×
* - String-Length
  - numpy
  - 0.92 s
  - 0.023 s
  - 39×
* - MHAOV
  - numpy
  - 3.56 s
  - 0.111 s
  - 32×
* - TLS
  - numpy
  - 4.49 s
  - 0.042 s
  - 106×
```

How to read this:

- **GLS and BLS already have specialized CPU backends** (finufft; the multicore numba box
  search). They're so fast on the CPU that the GPU's marginal gain on a single curve is
  modest — but the GPU still wins decisively for **large grids and big catalogs**
  ({doc}`batch`).
- **PDM, CE, String-Length, MHAOV, TLS run on numpy on the CPU**, so the GPU's
  data-parallelism delivers 30–180× on a single curve. If you're searching many periods or
  many stars with these, the GPU is a large win.
- cuPeriod's **CPU** path already beats the established reference tools it was checked
  against (GLS ~3× astropy, PDM ~4× PyAstronomy, BLS ~20× astropy's `BoxLeastSquares`).

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
