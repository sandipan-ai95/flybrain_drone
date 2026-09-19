"""Neuron models.

The :class:`NeuralModel` interface is the extension point required by spec §3:
the neural model must be swappable (LIF today; Izhikevich, rate, conductance
tomorrow) *without* changing the simulator or the rest of the pipeline.

Scientific assumptions (Milestone 1):
    * All neurons share identical LIF parameters. Real neurons don't — future
      milestones will read per-type parameters from annotations.
    * Synaptic transmission is instantaneous current injection (no synaptic
      time constants). Add exponential synapses in :mod:`flybrain.neuroscience.synapses`.
    * Excitatory-only weights (all positive). Real connectomes require sign
      annotations (e.g. FlyWire's neurotransmitter prediction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


class NeuralModel(Protocol):
    """Minimal interface every neuron model must implement."""
    num_neurons: int

    def step(self, inputs: np.ndarray, dt: float) -> np.ndarray:
        """Advance one timestep. Returns a spike (or activity) vector."""
        ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------- LIF
@dataclass
class LIFParams:
    v_rest: float = -65.0     # mV
    v_reset: float = -70.0    # mV
    v_thresh: float = -50.0   # mV
    tau_m: float = 20.0       # ms
    r_m: float = 10.0         # MΩ (scales input current to mV)
    refractory_ms: float = 2.0


class LIFModel:
    """Vectorised Leaky Integrate-and-Fire population.

        dv/dt = (-(v - v_rest) + R * I) / tau_m
        if v >= v_thresh: spike; v = v_reset; enter refractory
    """
    def __init__(self, num_neurons: int, params: LIFParams | None = None) -> None:
        self.num_neurons = int(num_neurons)
        self.p = params or LIFParams()
        self.v = np.full(self.num_neurons, self.p.v_rest, dtype=np.float32)
        self.refractory_left = np.zeros(self.num_neurons, dtype=np.float32)
        self.last_spikes = np.zeros(self.num_neurons, dtype=np.float32)

    def reset(self) -> None:
        self.v[:] = self.p.v_rest
        self.refractory_left[:] = 0.0
        self.last_spikes[:] = 0.0

    def step(self, inputs: np.ndarray, dt: float) -> np.ndarray:
        p = self.p
        active = self.refractory_left <= 0.0
        # Euler update on active neurons only.
        dv = (-(self.v - p.v_rest) + p.r_m * inputs) * (dt / p.tau_m)
        self.v = np.where(active, self.v + dv, self.v)
        spikes = (self.v >= p.v_thresh) & active
        # Reset & refractory
        self.v = np.where(spikes, p.v_reset, self.v)
        self.refractory_left = np.where(spikes, p.refractory_ms, self.refractory_left - dt)
        self.refractory_left = np.maximum(self.refractory_left, 0.0)
        self.last_spikes = spikes.astype(np.float32)
        return self.last_spikes


# --------------------------------------------------------------------------- rate
class RateModel:
    """Simple leaky rate model as a lightweight alternative to LIF for debugging."""
    def __init__(self, num_neurons: int, tau: float = 20.0, gain: float = 1.0) -> None:
        self.num_neurons = int(num_neurons)
        self.tau = tau
        self.gain = gain
        self.r = np.zeros(self.num_neurons, dtype=np.float32)

    def reset(self) -> None:
        self.r[:] = 0.0

    def step(self, inputs: np.ndarray, dt: float) -> np.ndarray:
        self.r += (dt / self.tau) * (-self.r + self.gain * np.tanh(inputs))
        return self.r
