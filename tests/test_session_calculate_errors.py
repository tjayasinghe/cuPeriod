"""The explicit Calculate solves with the covariance; an implicit one stays light.

Section 4.1 of the design ("View cost"): a light view shows only the errors a
covariance already in the head's fit provides. The explicit Calculate (F8) and
Improve solve with the covariance, so the frequency table gets frequency, amplitude
and phase errors at once (exactly 0 for a fixed quantity); the implicit Calculate
that follows every edit skips it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.prewhiten.constraints import ComponentConstraint, FitConstraints
from cuperiod.prewhiten.fit import MultiSineFit, fit_multisine
from cuperiod.prewhiten.session import PreWhitenSession
from cuperiod.prewhiten.session.report import build_result, rows
from cuperiod.prewhiten.session.state import SessionState
from cuperiod.prewhiten.uncertainty import component_uncertainties

FREQS = (3.1, 7.45, 11.2)
ERRORS = ("frequency_error", "amplitude_error", "phase_error")


def _light_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    t = np.sort(rng.uniform(0.0, 20.0, 360))
    y = (
        0.05 * np.sin(2 * np.pi * FREQS[0] * t + 0.3)
        + 0.02 * np.sin(2 * np.pi * FREQS[1] * t + 1.2)
        + 0.012 * np.sin(2 * np.pi * FREQS[2] * t + 2.0)
        + rng.normal(0.0, 0.003, t.size)
    )
    return t, y, np.full(t.size, 0.003)


def _empty() -> PreWhitenSession:
    return PreWhitenSession(
        _light_curve(),
        settings=cup.PreWhitenSettings(backend="numpy"),
        grid=(0.05, 0.005, 3000),
    )


def _session() -> PreWhitenSession:
    """Three components added at their true frequencies (held: ``refine="none"``)."""
    session = _empty()
    for frequency in FREQS:
        assert session.add(frequency, snap=False, refine="none").kind == "edited"
    return session


def _direct(session: PreWhitenSession, **kw: Any) -> MultiSineFit:
    """The fit Calculate stands for: every frequency held, covariance on."""
    t, y, e = _light_curve()
    return fit_multisine(
        t, y, e, np.array([c.frequency for c in session.state.active]),
        fit_mean=True, t_ref=session.t_ref, refine="none", covariance=True, **kw,
    )


def _assert_same_covariance(fit: MultiSineFit, direct: MultiSineFit) -> None:
    scale = float(np.abs(direct.covariance).max())
    np.testing.assert_allclose(
        fit.covariance, direct.covariance, rtol=1e-10, atol=1e-14 * scale
    )
    for name in ERRORS:
        np.testing.assert_allclose(
            getattr(fit, name), getattr(direct, name), rtol=1e-10
        )


def _table_errors(session: PreWhitenSession) -> dict[str, np.ndarray]:
    """The errors of the live (light) table, which never re-fits."""
    table = rows(session.kernel, session.state, session.derived)
    return {name: np.array([getattr(r, name) for r in table]) for name in ERRORS}


def _spy(
    session: PreWhitenSession, monkeypatch: pytest.MonkeyPatch
) -> list[bool]:
    """Record the ``covariance`` of every ``kernel.recalculate`` call."""
    calls: list[bool] = []
    kernel = session.kernel
    original = kernel.recalculate

    def recalculate(
        state: SessionState, *, covariance: bool = False
    ) -> Any:
        calls.append(covariance)
        return original(state, covariance=covariance)

    monkeypatch.setattr(kernel, "recalculate", recalculate)
    return calls


def test_calculate_gives_the_errors_of_a_direct_covariance_fit() -> None:
    session = _session()
    assert session.fit.covariance.size == 0  # the adds solved without one
    assert all(np.isnan(v).all() for v in _table_errors(session).values())
    report = session.calculate()
    assert report.kind == "edited"
    fit = session.fit
    np.testing.assert_array_equal(fit.frequency, FREQS)  # held
    direct = _direct(session)
    _assert_same_covariance(fit, direct)
    for name in ERRORS:
        values = getattr(fit, name)
        assert np.all(np.isfinite(values)) and np.all(values > 0.0), name
    # The table shows them as the configured covariance estimator reports them.
    t, _, e = _light_curve()
    expected = component_uncertainties(
        t, e, direct, method="covariance",
        correlation_correction=session.settings.correlation_correction,
    )
    shown = _table_errors(session)
    np.testing.assert_allclose(shown["frequency_error"], expected.frequency, rtol=1e-10)
    np.testing.assert_allclose(shown["amplitude_error"], expected.amplitude, rtol=1e-10)
    np.testing.assert_allclose(shown["phase_error"], expected.phase, rtol=1e-10)
    kernel, head, derived = session.kernel, session.state, session.derived
    light = build_result(kernel, head, derived, light=True)
    assert light.uncertainty_method == "covariance"
    assert np.isfinite(light.offset_error)
    # The full result needs no second covariance fit.
    assert kernel.with_covariance(head, derived) is derived


def test_calculate_errors_are_zero_for_fixed_quantities() -> None:
    session = _session()
    assert session.fix(1, frequency=True).kind == "edited"
    assert session.fix(2, amplitude=True, phase=True).kind == "edited"
    assert session.calculate().kind == "edited"
    fit = session.fit
    assert fit.detail is not None  # the constrained kernel solved it
    c2 = session.state.component(2)
    direct = _direct(
        session,
        constraints=FitConstraints(
            components=(
                ComponentConstraint(frequency="fixed"),
                ComponentConstraint(amplitude=c2.amplitude, phase=c2.phase),
                ComponentConstraint(),
            )
        ),
    )
    _assert_same_covariance(fit, direct)
    fixed = {("frequency_error", 0), ("amplitude_error", 1), ("phase_error", 1)}
    shown = _table_errors(session)
    for name in ERRORS:
        for i in range(len(FREQS)):
            for values in (getattr(fit, name), shown[name]):
                if (name, i) in fixed:
                    assert values[i] == 0.0, (name, i)
                else:  # free, even a held frequency
                    assert np.isfinite(values[i]) and values[i] > 0.0, (name, i)


def test_calculate_without_components_gives_the_offset_error() -> None:
    session = _empty()
    assert session.calculate().kind == "edited"
    fit = session.fit
    assert fit.n_components == 0 and np.isfinite(fit.offset_error)
    t, y, e = _light_curve()
    direct = fit_multisine(
        t, y, e, np.zeros(0), fit_mean=True, t_ref=session.t_ref, refine="none"
    )
    assert fit.offset_error == pytest.approx(direct.offset_error, rel=1e-12)


def test_an_implicit_calculate_stays_light(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    calls = _spy(session, monkeypatch)
    assert session.calculate().kind == "edited"
    assert calls == [True] and session.fit.covariance.size
    # Toggling a row re-solves through the implicit Calculate, without a covariance:
    # the table goes back to "no errors yet" rather than paying for one per edit.
    assert session.set_active(3, False).kind == "edited"
    assert calls == [True, False]
    assert session.fit.covariance.size == 0
    assert all(np.isnan(v).all() for v in _table_errors(session).values())
    light = build_result(session.kernel, session.state, session.derived, light=True)
    assert light.uncertainty_method == "none"
    assert session.set_active(3, True).kind == "edited"
    assert calls == [True, False, False]
    assert session.fit.covariance.size == 0


def test_improve_selected_reports_the_errors_of_the_persistent_model() -> None:
    session = _session()
    assert session.improve([2]).kind == "edited"
    fit = session.fit
    # The covariance of a Calculate at the improved frequencies, with every free
    # frequency free (the improving fit held F1 and F3, the errors do not).
    _assert_same_covariance(fit, _direct(session))
    assert np.all(fit.frequency_error > 0.0)
