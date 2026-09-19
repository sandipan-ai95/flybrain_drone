"""Neuron populations & annotations.

Real datasets provide cell-type labels (e.g. FlyWire's ``super_class``,
``cell_type``); those loaders will populate a :class:`NeuronAnnotations` for us.
For Milestone 1 we generate synthetic labels aligned with the synthetic graph.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Population:
    name: str
    neuron_indices: np.ndarray  # int32

    def size(self) -> int:
        return int(self.neuron_indices.size)


@dataclass
class NeuronAnnotations:
    num_neurons: int
    populations: dict[str, Population] = field(default_factory=dict)
    # Optional: cell-type string per neuron (from real datasets).
    cell_type: np.ndarray | None = None
    # Optional: xyz coords (µm) per neuron (from real datasets).
    xyz: np.ndarray | None = None

    def add_population(self, name: str, indices: np.ndarray) -> None:
        self.populations[name] = Population(name, np.asarray(indices, dtype=np.int32))

    def get(self, name: str) -> Population:
        return self.populations[name]

    @staticmethod
    def synthetic(n_visual: int, n_inter: int, n_motor: int) -> "NeuronAnnotations":
        N = n_visual + n_inter + n_motor
        ann = NeuronAnnotations(num_neurons=N)
        ann.add_population("visual", np.arange(0, n_visual))
        ann.add_population("interneuron", np.arange(n_visual, n_visual + n_inter))
        ann.add_population("motor", np.arange(n_visual + n_inter, N))
        # Sub-populations of motor for the decoder (see decoding/motor_decoder.py).
        # NOTE: these labels are computational — we do NOT claim these are real
        # biological motor identities.
        motor_start = n_visual + n_inter
        chunk = max(1, n_motor // 4)
        ann.add_population("motor_thrust", np.arange(motor_start, motor_start + chunk))
        ann.add_population("motor_pitch",  np.arange(motor_start + chunk, motor_start + 2*chunk))
        ann.add_population("motor_roll",   np.arange(motor_start + 2*chunk, motor_start + 3*chunk))
        ann.add_population("motor_yaw",    np.arange(motor_start + 3*chunk, N))
        return ann
