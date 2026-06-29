"""Batch processing: scale periodograms over many light curves."""

from __future__ import annotations

from cuperiod.batch.runner import BatchSummary, batch_periodograms

__all__ = ["BatchSummary", "batch_periodograms"]
