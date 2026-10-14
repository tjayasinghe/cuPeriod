"""Method registry and backend resolution."""

from __future__ import annotations

from typing import Any

import pytest

from cuperiod.core.errors import BackendUnavailableError, UnknownMethodError
from cuperiod.methods.base import get_method, list_methods, method_names


def test_lookup_is_case_insensitive() -> None:
    assert get_method("gls") is get_method("GLS")


def test_lookup_ignores_separators() -> None:
    # The documented "String-Length" spelling (and underscores) must resolve.
    sl = get_method("STRINGLENGTH")
    assert get_method("String-Length") is sl
    assert get_method("StringLength") is sl
    assert get_method("string_length") is sl


def test_multiresult_getitem_normalizes() -> None:
    import numpy as np

    import cuperiod as cup

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 80, 300))
    y = 12 + 0.3 * np.sin(2 * np.pi * t / 2.0) + 0.02 * rng.standard_normal(300)
    res = cup.periodogram((t, y, np.full(300, 0.02)), ["GLS", "String-Length"])
    assert res["String-Length"].method == "STRINGLENGTH"
    assert res["stringlength"] is res["String-Length"]


def test_unknown_method_raises() -> None:
    with pytest.raises(UnknownMethodError):
        get_method("not-a-method")


def test_registry_contains_phase1_methods() -> None:
    names = method_names()
    assert {"GLS", "BLS"} <= set(names)
    assert {m.name for m in list_methods()} == set(names)


def test_resolve_cpu_backend() -> None:
    from cuperiod.core.backend import available_backends

    assert get_method("GLS").resolve_backend("cpu") == "finufft"
    # BLS prefers the multicore numba box search on the CPU when installed,
    # falling back to astropy's compiled BoxLeastSquares otherwise.
    expected = "numba" if "numba" in available_backends() else "astropy"
    assert get_method("BLS").resolve_backend("cpu") == expected


def test_resolve_gpu_without_device_raises() -> None:
    from cuperiod.core.backend import cuda_available

    if cuda_available():
        pytest.skip("a GPU is present")
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("gpu")


def test_resolve_unknown_backend_raises() -> None:
    with pytest.raises(BackendUnavailableError):
        get_method("GLS").resolve_backend("nonsense")


def _fake_env(
    monkeypatch: pytest.MonkeyPatch, *, backends: set[str], cuda: bool, torch_gpu: bool
) -> None:
    """Pretend this machine has exactly ``backends`` and these device kinds."""
    import cuperiod.methods.base as base

    monkeypatch.setattr(base, "available_backends", lambda: set(backends))
    monkeypatch.setattr(base, "cuda_available", lambda: cuda)
    monkeypatch.setattr(base, "torch_gpu_available", lambda: torch_gpu)


def test_auto_skips_cuda_path_when_its_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cupy sees a GPU, but GLS's fast path (cufinufft) is not installed: "auto"
    # must fall through to the CPU rather than pick a backend that will raise
    # ImportError once it actually runs.
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy"},
        cuda=True,
        torch_gpu=False,
    )
    assert get_method("GLS").resolve_backend("auto") == "finufft"


def test_auto_prefers_torch_gpu_over_cpu_when_cuda_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy", "torch"},
        cuda=True,
        torch_gpu=True,
    )
    assert get_method("GLS").resolve_backend("auto") == "torch"


def test_auto_takes_cuda_path_when_fully_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy", "cufinufft"},
        cuda=True,
        torch_gpu=False,
    )
    assert get_method("GLS").resolve_backend("auto") == "cufinufft"


def test_gpu_request_names_the_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_env(
        monkeypatch,
        backends={"numpy", "finufft", "astropy", "cupy"},
        cuda=True,
        torch_gpu=False,
    )
    with pytest.raises(BackendUnavailableError, match="cufinufft"):
        get_method("GLS").resolve_backend("gpu")


# --- backend selector spelling (CORE-B-15) -------------------------------------------


@pytest.mark.parametrize("raw", ["CPU", "Cpu", " cpu ", "cpu\n"])
def test_backend_selector_is_case_and_space_insensitive(raw: str) -> None:
    for name in method_names():
        m = get_method(name)
        assert m.resolve_backend(raw) == m.resolve_backend("cpu")


def test_concrete_backend_names_are_case_insensitive() -> None:
    assert get_method("GLS").resolve_backend("FINUFFT") == "finufft"
    assert get_method("BLS").resolve_backend(" NumPy ") == "numpy"


def test_unknown_backend_message_lists_the_generic_selectors() -> None:
    with pytest.raises(BackendUnavailableError) as info:
        get_method("GLS").resolve_backend("nonsense")
    message = str(info.value)
    for selector in ("'auto'", "'cpu'", "'gpu'", "torch", "finufft", "cufinufft"):
        assert selector in message


# --- is_gpu_backend for the bare torch name (METH-A-20) ------------------------------


def test_bare_torch_is_gpu_only_when_torch_sees_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuperiod as cup
    import cuperiod.methods.base as base

    gls = get_method("GLS")
    monkeypatch.setattr(base, "torch_gpu_available", lambda: False)
    assert gls.is_gpu_backend("torch") is False
    monkeypatch.setattr(base, "torch_gpu_available", lambda: True)
    assert gls.is_gpu_backend("torch") is True
    # A settings device pin decides where bare "torch" runs.
    assert gls.is_gpu_backend("torch", cup.GLSSettings(device="cpu")) is False
    # Device-qualified names are unaffected.
    assert gls.is_gpu_backend("torch:cuda") is True
    assert gls.is_gpu_backend("torch:cpu") is False
    assert gls.is_gpu_backend("cufinufft") is True
    assert gls.is_gpu_backend("finufft") is False


# --- cached default settings (PERF-METHODS-13) ----------------------------------------


def test_default_settings_equal_a_fresh_instance() -> None:
    for name in method_names():
        m = get_method(name)
        a = m.coerce_settings(None)
        b = m.coerce_settings(None)
        assert a == m.settings_cls()
        assert a is not b
        # Mutating one copy must not leak into the next default.
        a.min_detections = 999  # type: ignore[attr-defined]
        assert m.coerce_settings(None) == m.settings_cls()


def test_default_settings_follow_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    gls = get_method("GLS")
    monkeypatch.delenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", raising=False)
    assert gls.coerce_settings(None).samples_per_peak == 5  # type: ignore[attr-defined]
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "7")
    assert gls.coerce_settings(None).samples_per_peak == 7  # type: ignore[attr-defined]
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "9")
    assert gls.coerce_settings(None).samples_per_peak == 9  # type: ignore[attr-defined]
    monkeypatch.delenv("CUPERIOD_GLS_SAMPLES_PER_PEAK")
    assert gls.coerce_settings(None).samples_per_peak == 5  # type: ignore[attr-defined]
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "0")
    with pytest.raises(ValidationError):
        gls.coerce_settings(None)


def test_default_settings_are_built_once_per_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cache is the point of the fix: without it every call re-reads the
    # environment and re-validates every field (0.3-0.5 ms). Count constructions.
    import cuperiod.methods.base as base
    from cuperiod.core.config import GLSSettings

    built: list[int] = []
    real_init = GLSSettings.__init__

    def counting_init(self: GLSSettings, *args: object, **kwargs: object) -> None:
        built.append(1)
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(GLSSettings, "__init__", counting_init)
    monkeypatch.delenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", raising=False)
    cache: dict[Any, Any] = base._DEFAULT_SETTINGS
    monkeypatch.delitem(cache, GLSSettings, raising=False)
    gls = get_method("GLS")
    for _ in range(3):
        gls.coerce_settings(None)
    assert len(built) == 1
    # A changed CUPERIOD_GLS_* variable rebuilds the template exactly once.
    monkeypatch.setenv("CUPERIOD_GLS_SAMPLES_PER_PEAK", "7")
    copies = [gls.coerce_settings(None) for _ in range(3)]
    assert len(built) == 2
    assert all(c.samples_per_peak == 7 for c in copies)  # type: ignore[attr-defined]
