"""Closed-loop neural simulator.

Per spec §6 the per-timestep loop is:

    1. receive sensory input
    2. inject current into sensory neurons
    3. update active neurons
    4. generate spikes
    5. propagate spikes through synaptic edges
    6. update downstream neurons
    7. record activity

This module only handles steps 3–7 (given already-injected currents). Vision
and motor decoding wrap around it in :mod:`flybrain.demo.hover_loop`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from flybrain.connectome.graph import Connectome
from flybrain.neuroscience.neuron_models import NeuralModel
from flybrain.neuroscience.synapses import InstantaneousSynapses


@dataclass
class SimulationState:
    t: float = 0.0
    step_idx: int = 0
    spikes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    total_spikes: int = 0


class NeuralSimulator:
    def __init__(
        self,
        connectome: Connectome,
        model: NeuralModel,
        dt_ms: float = 1.0,
    ) -> None:
        assert connectome.num_neurons == model.num_neurons, "connectome/model size mismatch"
        self.connectome = connectome
        self.model = model
        self.dt = float(dt_ms)
        self.synapses = InstantaneousSynapses(connectome.W)
        self.state = SimulationState(spikes=np.zeros(model.num_neurons, np.float32))

    def reset(self) -> None:
        self.model.reset()
        self.state = SimulationState(spikes=np.zeros(self.model.num_neurons, np.float32))

    def step(self, external_current: np.ndarray) -> np.ndarray:
        """One neural timestep. Returns the spike vector.

        `external_current` is the sensory / bias current injected this step
        (shape ``(N,)``). It's added to the synaptic current from the previous
        spike vector.
        """
        syn_current = self.synapses.propagate(self.state.spikes)
        total_current = syn_current + external_current
        spikes = self.model.step(total_current, self.dt)
        self.state.spikes = spikes
        self.state.t += self.dt
        self.state.step_idx += 1
        self.state.total_spikes += int(spikes.sum())
        return spikes
