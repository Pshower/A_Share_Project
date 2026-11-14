"""Newey-West standard errors, in one place.

Moved here when the alpha-code research module was retired: the Bartlett kernel is the only
piece of it the live line still uses, and a 30-line helper should not drag a 455-line study
toolchain along with it.
"""
from __future__ import annotations

import math

import numpy as np


def nw_se(values, lag):
    """Newey-West standard error of the mean with a Bartlett kernel of ``lag`` lags."""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 4:
        return float("nan")
    lag = int(min(max(int(lag), 0), n - 2))
    centred = series - series.mean()
    variance = float(centred @ centred) / n
    for step in range(1, lag + 1):
        covariance = float(centred[:-step] @ centred[step:]) / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return math.sqrt(max(variance, 0.0) / n)
