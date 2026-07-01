"""Tests for the cache keys and the LRU result cache."""

from __future__ import annotations

import numpy as np

from cuperiod.core.config import GLSSettings
from cuperiod.core.result import Periodogram
from cuperiod.gui.models import ResultCache, ResultKey, settings_hash


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
