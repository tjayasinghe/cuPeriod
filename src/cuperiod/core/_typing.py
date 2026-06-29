"""Shared NumPy array type aliases used across cuPeriod.

Centralizing these keeps the public signatures terse and consistent: a periodogram
is built from ``float64`` time/value/error arrays, returns ``float64`` power, and
stores downsampled spectra as ``float32`` to keep batch outputs compact.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: A contiguous ``float64`` array (times, magnitudes, power spectra).
FloatArray = npt.NDArray[np.float64]

#: A ``float32`` array (compact storage for downsampled periodograms).
Float32Array = npt.NDArray[np.float32]

#: A ``int64`` index array (peak indices, bin counts).
IntArray = npt.NDArray[np.int64]

__all__ = ["FloatArray", "Float32Array", "IntArray"]
