"""Tests for the cache keys and the LRU result cache."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.config import GLSSettings
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.gui.models import (
    ResultCache,
    ResultKey,
    data_fingerprint,
    result_nbytes,
    settings_hash,
)


def _periodogram() -> Periodogram:
    freq = np.linspace(0.1, 1.0, 50)
    power = np.random.default_rng(0).random(50)
    return Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )


def test_settings_hash_order_independent() -> None:
    a = GLSSettings(samples_per_peak=7, nyquist_factor=3)
    b = GLSSettings(nyquist_factor=3, samples_per_peak=7)
    assert settings_hash(a) == settings_hash(b)


def test_settings_hash_changes_with_value() -> None:
    a = GLSSettings(samples_per_peak=7)
    b = GLSSettings(samples_per_peak=8)
    assert settings_hash(a) != settings_hash(b)


def test_result_cache_lru_eviction() -> None:
    cache = ResultCache(maxsize=2)
    keys = [ResultKey(f"s{i}", "GLS", "h", "auto") for i in range(3)]
    pg = _periodogram()
    cache.put(keys[0], pg)
    cache.put(keys[1], pg)
    cache.get(keys[0])  # touch keys[0] so keys[1] becomes least-recently-used
    cache.put(keys[2], pg)  # evicts keys[1]
    assert keys[0] in cache
    assert keys[1] not in cache
    assert keys[2] in cache
    assert len(cache) == 2


def test_result_cache_get_marks_recent() -> None:
    cache = ResultCache(maxsize=8)
    key = ResultKey("s", "GLS", "h", "auto")
    assert cache.get(key) is None
    pg = _periodogram()
    cache.put(key, pg)
    assert cache.get(key) is pg


# -- GUI-A-1: the cache key identifies the data, not just the file stem -------------


def _curve(seed: int, n: int = 100) -> LightCurve:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 50.0, n))
    return LightCurve.from_arrays(time, rng.normal(size=n), np.full(n, 0.1))


def test_result_key_keeps_its_four_field_constructor() -> None:
    key = ResultKey("s", "GLS", "h", "auto")
    assert key.data_id == ""
    assert key != ResultKey("s", "GLS", "h", "auto", "abc")


def test_data_fingerprint_tracks_the_numbers_not_the_object() -> None:
    a, b = _curve(1), _curve(2)
    assert data_fingerprint(a) != data_fingerprint(b)
    # the same data loaded again (a new object) keeps its fingerprint
    assert data_fingerprint(a) == data_fingerprint(_curve(1))
    # one changed sample (a file edited on disk) changes it
    edited = LightCurve.from_arrays(a.time, a.value + np.eye(1, a.n)[0], a.error)
    assert data_fingerprint(edited) != data_fingerprint(a)
    # errors and the domain are part of the data too
    no_err = LightCurve.from_arrays(a.time, a.value)
    assert data_fingerprint(no_err) != data_fingerprint(a)
    flux = LightCurve.from_arrays(a.time, a.value, a.error, domain="flux")
    assert data_fingerprint(flux) != data_fingerprint(a)
    assert data_fingerprint(None) == ""


def test_data_fingerprint_multiband_depends_on_band_names() -> None:
    a, b = _curve(1), _curve(2)
    gr = MultiBandLightCurve.from_light_curves({"g": a, "r": b})
    gi = MultiBandLightCurve.from_light_curves({"g": a, "i": b})
    assert data_fingerprint(gr) != data_fingerprint(gi)
    assert data_fingerprint(gr) != data_fingerprint(a)
    same = MultiBandLightCurve.from_light_curves({"g": _curve(1), "r": _curve(2)})
    assert data_fingerprint(gr) == data_fingerprint(same)


def test_settings_hash_exclude_drops_fields() -> None:
    a = GLSSettings(peak_separation_rayleigh=3.0)
    b = GLSSettings(peak_separation_rayleigh=30.0)
    assert settings_hash(a) != settings_hash(b)
    peak_only = {"peak_separation_rayleigh"}
    assert settings_hash(a, exclude=peak_only) == settings_hash(b, exclude=peak_only)


# -- GUI-A-11 / SEC-14: the caches are bounded by bytes, not only by count ----------


def _sized(n: int) -> Periodogram:
    freq = np.linspace(0.1, 1.0, n)
    return Periodogram.from_spectrum(
        method="GLS",
        backend="numpy",
        frequency=freq,
        power=np.ones(n),
        objective_sense="max",
        n_samples=100,
        baseline=90.0,
    )


def test_result_nbytes_counts_each_buffer_once() -> None:
    pg = _sized(1000)
    # frequency, period and power: three float64 arrays of 1000
    assert result_nbytes(pg) == pytest.approx(3 * 8000, rel=0.01)
    shared = np.zeros(500)
    assert result_nbytes({"a": shared, "b": shared[:10], "c": (shared,)}) == 4000


def test_result_cache_evicts_by_bytes_in_lru_order() -> None:
    entry = result_nbytes(_sized(1000))
    cache: ResultCache[Periodogram] = ResultCache(maxsize=64, max_bytes=3 * entry)
    keys = [ResultKey(f"s{i}", "GLS", "h", "auto") for i in range(5)]
    for key in keys[:3]:
        cache.put(key, _sized(1000))
    cache.get(keys[0])  # keys[1] is now least recently used
    cache.put(keys[3], _sized(1000))
    assert len(cache) == 3 and keys[1] not in cache and keys[0] in cache
    cache.put(keys[4], _sized(1000))
    assert len(cache) == 3 and keys[2] not in cache
    assert cache.nbytes <= 3 * entry


def test_result_cache_keeps_an_entry_larger_than_the_budget() -> None:
    cache: ResultCache[Periodogram] = ResultCache(max_bytes=100)
    small, big = ResultKey("a", "GLS", "h", "auto"), ResultKey("b", "GLS", "h", "auto")
    cache.put(small, _sized(10))
    cache.put(big, _sized(10_000))
    assert big in cache and small not in cache and len(cache) == 1
    cache.clear()
    assert cache.nbytes == 0 and len(cache) == 0


def test_result_cache_refresh_does_not_double_count() -> None:
    cache: ResultCache[Periodogram] = ResultCache(max_bytes=10**9)
    key = ResultKey("a", "GLS", "h", "auto")
    cache.put(key, _sized(1000))
    first = cache.nbytes
    cache.put(key, _sized(1000))
    assert cache.nbytes == first and len(cache) == 1
