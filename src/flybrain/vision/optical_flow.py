"""Compute an approximate motion signal ΔI = I(t) - I(t-dt)."""
from __future__ import annotations
import numpy as np


def frame_difference(prev: np.ndarray | None, curr: np.ndarray) -> np.ndarray:
    """Return a non-negative scalar motion image (H, W) in [0, 1]."""
    if prev is None or prev.shape != curr.shape:
        return np.zeros(curr.shape[:2], dtype=np.float32)
    a = curr.astype(np.float32).mean(axis=-1) / 255.0
    b = prev.astype(np.float32).mean(axis=-1) / 255.0
    return np.abs(a - b)
