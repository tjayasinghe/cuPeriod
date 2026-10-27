# Backends & performance

Every method has more than one *backend* — a concrete implementation of the same math.
You select among them with the `backend=` argument; the results agree to floating-point
round-off whichever one runs — GLS to its NUFFT tolerance (`nufft_eps`, 1e-9 relative;
≤ 5e-9 absolute on the validation set) — and the CPU, CUDA and torch backends picked the
same best period on every validation star ({doc}`../benchmarks`).

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

Selectors ignore case and surrounding spaces (`"CPU"`, `" Auto "`).

**The settings' backend is the default.** Every settings model has a `backend` field
(`CUPERIOD_<METHOD>_BACKEND` in the environment). `cup.periodogram(..., backend="auto")`
— or `None` — defers to it, and any other `backend=` overrides it; the batch drivers, the
LINCC adapters, {func}`~cuperiod.pdot_search`, {func}`~cuperiod.sliding_periodogram`,
{func}`~cuperiod.multiband_fap` and the GUI follow the same rule. A settings pin of
`"gpu"` on a machine without one raises {exc}`~cuperiod.BackendUnavailableError` instead
of silently running on the CPU, and bare `"torch"` counts as a GPU backend only when torch
actually has a GPU (or `settings.device` pins one).

Backend discovery is memoised per process (resolving a backend costs about a
microsecond), and an explicit CPU request never imports cupy. After installing an extra
into a running session, `cuperiod.core.backend.refresh_backends()` probes again. Batch
CPU pools resolve `"auto"` to CPU backends only, so they never open a GPU ({doc}`batch`).

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
* - SuperSmoother
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
all. The NVIDIA fast paths (cufinufft for GLS, the cupy kernels for the other methods)
are untouched and remain what `"auto"`/`"gpu"` pick on CUDA; torch is the cross-vendor path
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
  only where the device forces it (Apple MPS). `"float64"` and `"float32"`
  force it; `precision="float64"` on MPS raises rather than silently downgrading. Results
  are always returned as float64 numpy arrays regardless of the device precision.

:::{note}
The portable path is correct everywhere: it matches the CPU reference to round-off (GLS
to its NUFFT tolerance, MHAOV to ~1e-10 from its linear solve; the {doc}`../benchmarks`
list each method) and picked the CPU's best period on every validation star. It is
**not** the speed champion on CPU, where finufft (GLS) and the numba box search (BLS)
are faster. Its value is reaching GPUs the CUDA fast paths can't — AMD, Intel, and
Apple. On those devices its
massively-parallel evaluation is the win. Run `cuperiod doctor` to see what you have.
:::

Inspect the live picture for your install:

```python
for info in cup.list_methods():
    print(info.name, "| all:", info.all_backends, "| available:", info.available_backends)
```

## What's fast where

The headline from the {doc}`../benchmarks` (single light curve, ~900 points; an RTX 5070 Ti
vs. the `[fast]`-extra CPU backends, on a 32-thread machine):

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
  - 0.006 s
  - 0.002 s
  - 0.008 s
  - ~3×
* - BLS
  - numba
  - 0.039 s
  - 0.090 s
  - 0.382 s
  - ~0.4× (GPU slower)
* - PDM
  - numba
  - 0.003 s
  - 0.005 s
  - 0.011 s
  - ~0.6× (GPU slower)
* - CE
  - numba
  - 0.005 s
  - 0.004 s
  - 0.044 s
  - ~1.3×
* - String-Length
  - numba
  - 0.009 s
  - 0.012 s
  - 0.009 s
  - ~0.8× (GPU slower)
* - MHAOV
  - numba
  - 0.026 s
  - 0.037 s
  - 0.031 s
  - ~0.7× (GPU slower)
* - TLS
  - numba
  - 0.126 s
  - 0.133 s
  - 7.047 s
  - ~0.9× (a wash)
* - SuperSmoother
  - numba
  - 0.064 s
  - 0.271 s
  - 0.214 s
  - ~0.2× (GPU slower)
```

How to read this:

- **The CPU backend for every method but GLS is now the multicore `numba` kernel** (with the
  `[fast]` extra installed — one to two orders of magnitude faster than the vectorized
  numpy fallback, see {doc}`../installation`). On this 32-thread machine, that CPU tier is
  now fast enough that a single-curve GPU run is only a modest win for GLS (~3×), a
  near-wash for CE and TLS (~0.9-1.3×), and the GPU is actually *slower* than the CPU for
  String-Length, MHAOV, PDM and BLS (~0.4-0.8×) and SuperSmoother (~0.2×) at this curve
  size — fixed dispatch and host↔device transfer overhead no longer amortizes when the CPU
  kernel itself takes milliseconds. GLS keeps its edge because its CPU path is finufft,
  not a numba kernel.
- **TLS** searches a default grid 2.5× denser than in 1.2 (380,614 trial periods here,
  enough to resolve the shortest searched transit) in the same CPU time — 2.6× less per
  trial period. Its corrected statistic fits the out-of-transit level jointly with the depth;
  the numba and cupy kernels absorbed that cost through their own speed-ups, but the
  portable `torch:cuda` path takes about 30% longer per trial period than in 1.2, which
  with the denser grid is why its column reads 7 s.
- The scaling sweep (up to 30k points and a 100k-frequency grid) adds nuances. GLS's GPU
  edge grows with the grid: with small finufft transforms running single-threaded, the CPU
  is faster up to 3k frequencies (0.6× at 1k, 0.9× at 3k), while the GPU is 3.1× faster at
  30k and 6.6× at 100k, and 2.9–4.5× across curve lengths. MHAOV is a near-wash across
  grid sizes (0.3–1.1×), but across curve lengths the numba path stays near 0.03–0.04 s
  from 100 to 30,000 points (on uniform grids it sums the harmonics by NUFFT) while the
  GPU's time grows with the point count — 2.2× faster than the CPU at 100 points, 25×
  slower at 30,000. SuperSmoother's GPU only approaches CPU parity once curves reach
  several thousand points. Expect a wider GPU margin on a narrower CPU. The GPU's clear,
  reproducible win is **catalog throughput** — many curves in flight at once — and reaching
  non-NVIDIA hardware via the portable torch backend, not single-curve latency on the
  CPU-tier methods ({doc}`batch`).
- Without the `[fast]` extra, PDM/CE/String-Length/MHAOV/TLS/SuperSmoother fall back to the
  **vectorized numpy paths** on the CPU — one to two orders of magnitude slower than the
  numba column above (e.g. PDM ~300×, CE ~135×, MHAOV ~57× on a 3k-point curve, and
  SuperSmoother ~130× on a separate 600-point curve over a 20 000-frequency grid).
- On **consumer NVIDIA cards** (GeForce), whose float64 throughput is 1/64 of float32,
  the opt-in `precision="float32"` runs the BLS/TLS CUDA kernels ~8-9× faster at
  detection-grade accuracy; float64 stays the default.
- cuPeriod's **CPU** path already beats the established reference tools it was checked
  against — GLS ~6× astropy, BLS ~89× astropy's `BoxLeastSquares`, and PDM's numba kernel
  ~2,000× PyAstronomy's pure-Python `pyPDM` (the numpy PDM path alone is already ~4×).
- Small finufft transforms — `n_trans × N + n_modes ≤ 150 000`, single GLS curves,
  `pdot_search` rows, pre-whitening spectra — run on one thread, where thread start-up
  would otherwise dominate (GLS 4–11× faster, pre-whitening spectra up to ~7× on many-core
  CPUs); larger ones use every core (`OMP_NUM_THREADS`). The crossover was measured on a
  32-thread Windows desktop and is machine-dependent.

:::{tip}
Write `backend="auto"` and let cuPeriod choose. Reach for `"cpu"`/`"gpu"` or a concrete
name only to benchmark, to pin a reference implementation, or to keep a long-running job
off a busy GPU.
:::

## When the GPU isn't available

`backend="auto"` silently falls back to the CPU, so portable code "just works."
`backend="gpu"` is explicit and raises {exc}`~cuperiod.BackendUnavailableError` when no
GPU path is usable — neither a CUDA device with the `[gpu]` extra nor a GPU that the
`[torch]` extra sees (AMD, Intel, Apple, or an NVIDIA card without `[gpu]`) — use it when
you *want* to fail loudly rather than run on the CPU by accident.

```python
try:
    pg = cup.periodogram(lc, "TLS", backend="gpu")
except cup.BackendUnavailableError as e:
    print("no GPU:", e)
```

---

Next: {doc}`tuning` — settings and custom grids.
