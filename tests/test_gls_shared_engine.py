"""One-off GPU GLS periodograms reuse a process-wide engine (PERF-METHODS-12).

``cup.periodogram(..., backend="cufinufft")`` and the multi-band path used to build
and destroy two cufinufft plans per call -- 40-70% of a GPU GLS periodogram. They
now share :func:`shared_cufinufft_engine`: a bounded, lock-protected plan cache
that :func:`cuperiod.free_gpu_memory` gives back. These tests pin the results to
the CPU path and the plan-per-call path, the plan reuse, the LRU bound, the release
and concurrent use from several threads.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import cuperiod as cup
from conftest import requires_gpu
from cuperiod.methods import gls as gls_module
from synth import synthetic_multiband_sine, synthetic_sine


def test_free_gpu_memory_does_not_import_gls_or_cupy(monkeypatch) -> None:
    # Freeing memory must stay import-free: without GLS loaded nothing is released.
    import sys

    monkeypatch.delitem(sys.modules, "cuperiod.methods.gls", raising=False)
    cup.free_gpu_memory()  # must not raise, must not need a device


@requires_gpu
def test_one_off_gls_uses_the_shared_engine_and_matches_the_cpu() -> None:
    gls_module.release_shared_engines()
    t, mag, err = synthetic_sine()
    cpu = cup.periodogram((t, mag, err), "GLS", backend="finufft")
    gpu = cup.periodogram((t, mag, err), "GLS", backend="cufinufft")
    assert gpu.backend == "cufinufft"
    assert float(np.max(np.abs(cpu.power - gpu.power))) < 1e-7
    engine = gls_module.shared_cufinufft_engine(cup.GLSSettings().nufft_eps)
    plans = dict(engine._plans)
    assert plans  # the call planned through the shared engine
    again = cup.periodogram((t, mag, err), "GLS", backend="cufinufft")
    # Same plans, reused (no new keys, same objects), and the same numbers.
    assert dict(engine._plans) == plans
    np.testing.assert_array_equal(again.power, gpu.power)


@requires_gpu
def test_shared_engine_matches_the_plan_per_call_transform() -> None:
    t, mag, err = synthetic_sine(n=900, seed=3)
    lc = cup.LightCurve.from_arrays(t, mag, err)
    settings = cup.GLSSettings(fap_method="none")
    method = cup.get_method("GLS")
    grid = method.default_grid(lc, settings)
    f0, df, nf = grid.uniform_frequency_params()
    oneshot = gls_module.lombscargle_power(
        t, mag, err, f0, df, nf, backend="cufinufft"
    )
    shared = method.power(grid, lc, settings, "cufinufft").power
    assert float(np.max(np.abs(oneshot - shared))) < 1e-8


@requires_gpu
def test_free_gpu_memory_releases_the_shared_plans() -> None:
    t, mag, err = synthetic_sine(n=300, seed=5)
    cup.periodogram((t, mag, err), "GLS", backend="cufinufft")
    eps = cup.GLSSettings().nufft_eps
    engine = gls_module.shared_cufinufft_engine(eps)
    assert engine._plans
    cup.free_gpu_memory()
    assert not engine._plans
    assert eps not in gls_module._SHARED_ENGINES
    # The next call simply builds a fresh engine.
    pg = cup.periodogram((t, mag, err), "GLS", backend="cufinufft")
    assert pg.power.size


@requires_gpu
def test_plan_cache_is_bounded_least_recently_used() -> None:
    engine = gls_module.CufinufftGLS(nf_bucket=1024, max_cached_modes=6 * 1024)
    t, mag, err = synthetic_sine(n=200, seed=7)
    for nf in (1000, 2000, 3000):  # plan sizes 1024, 2048, 3072 (n_trans 2 and 1)
        engine.power(t, mag, err, 0.05, 0.001, nf)
    total = sum(m * k for m, k, _ in engine._plans)
    assert total <= 6 * 1024
    # The newest plans survive.
    assert (3072, 1, "sorted") in engine._plans
    assert (1024, 2, "sorted") not in engine._plans


@requires_gpu
def test_shared_engine_is_safe_across_threads() -> None:
    curves = [synthetic_sine(n=400 + 50 * k, seed=k) for k in range(6)]
    expected = [
        cup.periodogram(c, "GLS", backend="cufinufft").power for c in curves
    ]
    results: dict[int, np.ndarray] = {}
    errors: list[BaseException] = []

    def work(k: int) -> None:
        try:
            for _ in range(3):
                results[k] = cup.periodogram(
                    curves[k], "GLS", backend="cufinufft"
                ).power
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(len(curves))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    for k, power in enumerate(expected):
        np.testing.assert_allclose(results[k], power, rtol=0.0, atol=1e-12)


@requires_gpu
@pytest.mark.parametrize("model", ["offsets", "perband", "flex"])
def test_multiband_one_off_uses_the_shared_engine(model: str) -> None:
    from cuperiod.multiband.gls_mb import gls_multiband_power

    gls_module.release_shared_engines()
    bands = synthetic_multiband_sine()
    mb = cup.MultiBandLightCurve.from_light_curves(
        {name: cup.LightCurve.from_arrays(t, m, e) for name, (t, m, e) in bands.items()}
    )
    grid = cup.uniform_frequency_grid(
        float(np.ptp(np.concatenate([b[0] for b in bands.values()]))),
        maximum_frequency=4.0,
    )
    settings = cup.GLSSettings(mb_model=model)  # type: ignore[arg-type]
    gpu = gls_multiband_power(grid, mb, settings, "cufinufft").power
    cpu = gls_multiband_power(grid, mb, settings, "finufft").power
    assert np.allclose(gpu, cpu, atol=1e-8)
    assert gls_module.shared_cufinufft_engine(settings.nufft_eps)._plans
