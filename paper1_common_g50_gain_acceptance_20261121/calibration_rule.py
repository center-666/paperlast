"""Finite-sample MARGINAL healthy path false-alarm calibration for future use.

This rule is not applied retroactively to the legacy study and is NOT enabled
by the common-G raw42 compatibility runner. Paths must be exchangeable and a
fresh path independent of the calibration sample; dependence within a path
and between the six monitored streams is allowed.
"""
from fractions import Fraction
import numpy as np


def path_fwer_cutoff(healthy_path_maxima, alpha=.05):
    """Return strict-alarm cutoff and rank bound; no conditional-FWER claim."""
    maxima = np.asarray(healthy_path_maxima, dtype=float)
    if maxima.ndim != 1 or len(maxima) == 0 or not np.all(np.isfinite(maxima)):
        raise ValueError('A nonempty finite vector of calibration path maxima is required.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0,1).')
    n = len(maxima)
    rank = (n + 1) * (1 - Fraction(str(alpha)))
    k = -(-rank.numerator // rank.denominator)  # Exact ceil for decimal alpha.
    cutoff = float(np.partition(maxima, k - 1)[k - 1]) if k <= n else float('inf')
    return {'threshold': cutoff, 'calibration_paths': n, 'one_based_rank': k,
            'alarm_rule': 'fresh_path_max > threshold',
            'marginal_path_fwer_bound': (n + 1 - k) / (n + 1),
            'warning': 'Marginal over calibration plus fresh path; not a guarantee conditional on a fixed realized calibration set.'}
