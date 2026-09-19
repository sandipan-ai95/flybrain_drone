"""Synaptic dynamics.

Milestone 1 uses instantaneous synapses (current = W @ spikes). This module
holds richer synapse implementations for later milestones so the simulator
doesn't need to change.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class InstantaneousSynapses:
    """I_post = W @ spikes (delta synapses). Fastest and simplest."""
    def __init__(self, W: sp.csr_matrix) -> None:
        self.W = W

    def propagate(self, spikes: np.ndarray) -> np.ndarray:
        return self.W @ spikes


class ExponentialSynapses:
    """Postsynaptic current decays with time constant tau_s:

        ds/dt = -s / tau_s + W @ spikes
        I_post = s
    """
    def __init__(self, W: sp.csr_matrix, tau_s: float = 5.0) -> None:
        self.W = W
        self.tau_s = tau_s
        self.s = np.zeros(W.shape[0], dtype=np.float32)

    def propagate(self, spikes: np.ndarray, dt: float = 1.0) -> np.ndarray:
        self.s += -self.s * (dt / self.tau_s) + (self.W @ spikes)
        return self.s

    def reset(self) -> None:
        self.s[:] = 0.0
