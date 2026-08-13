"""Multi-band periodogram variants (GLS, BLS, MHAOV, PDM, CE, string-length).

One module per method, each exporting the single entry point the owning
:class:`~cuperiod.methods.base.PeriodogramMethod` calls from ``multiband_power``:
:func:`~cuperiod.multiband.gls_mb.gls_multiband_power`,
:func:`~cuperiod.multiband.bls_mb.bls_multiband_power`,
:func:`~cuperiod.multiband.mhaov_mb.mhaov_multiband_power`,
:func:`~cuperiod.multiband.pdm_mb.pdm_multiband_theta`,
:func:`~cuperiod.multiband.conditional_entropy_mb.ce_multiband_entropy` and
:func:`~cuperiod.multiband.string_length_mb.string_length_multiband`.

The submodules are imported lazily by their methods (never here), so a single-band run
never pays for the multi-band code paths — hence the empty ``__all__``.
"""

from __future__ import annotations

__all__: list[str] = []
