"""Generic stats helpers shared across the notebooks in this folder -- kept
here rather than in `bench/validation/` because they're plain utility
functions (p-value correction, significance labels), not data-loading or
analysis logic tied to one notebook's dataset.
"""
from __future__ import annotations

import numpy as np


def holm_adjust(pvals):
    """Holm-Bonferroni step-down correction for a family of p-values."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    for i, idx in enumerate(order):
        adj[idx] = min(pvals[idx] * (len(pvals) - i), 1.0)
    for i in range(1, len(order)):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    return adj


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
