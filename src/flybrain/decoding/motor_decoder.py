"""Population motor decoder.

Spec §7 — we deliberately keep this **separate** from the connectome. It is an
*experimental* mapping from neural population activity to drone commands, not
a claim about biological motor identity.

We maintain low-pass filtered firing rates for four labeled sub-populations
(``motor_thrust``, ``motor_pitch``, ``motor_roll``, ``motor_yaw``) and map
them to :class:`DroneCommand` fields via configurable gains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.drone.physics import DroneCommand


@dataclass
class MotorDecoderConfig:
    tau_ms: float = 50.0
    thrust_gain: float = 40.0
    pitch_gain:  float = 6.0
    roll_gain:   float = 6.0
    yaw_gain:    float = 4.0
    thrust_bias: float = 0.0     # baseline hover offset (m/s^2 above g)
    populations: dict[str, str] = field(default_factory=lambda: {
        "thrust": "motor_thrust",
        "pitch":  "motor_pitch",
        "roll":   "motor_roll",
        "yaw":    "motor_yaw",
    })


class PopulationMotorDecoder:
    def __init__(self, annotations: NeuronAnnotations, config: MotorDecoderConfig | None = None):
        self.ann = annotations
        self.cfg = config or MotorDecoderConfig()
        self.rates = {k: 0.0 for k in self.cfg.populations}

    def reset(self) -> None:
        for k in self.rates: self.rates[k] = 0.0

    def decode(self, spikes: np.ndarray, dt_ms: float) -> DroneCommand:
        alpha = dt_ms / self.cfg.tau_ms
        for key, pop_name in self.cfg.populations.items():
            idx = self.ann.get(pop_name).neuron_indices
            inst = float(spikes[idx].mean()) if idx.size else 0.0
            self.rates[key] += alpha * (inst - self.rates[key])

        # Symmetric interpretations: thrust is offset from bias; pitch/roll/yaw
        # are signed by comparing to a neutral 0.5 firing rate. For LIF spikes
        # instantaneous rate is 0/1, so we use the *filtered* rate directly.
        return DroneCommand(
            thrust=self.cfg.thrust_bias + self.cfg.thrust_gain * (self.rates["thrust"] - 0.05),
            pitch=self.cfg.pitch_gain * (self.rates["pitch"] - 0.05),
            roll=self.cfg.roll_gain  * (self.rates["roll"]  - 0.05),
            yaw=self.cfg.yaw_gain    * (self.rates["yaw"]   - 0.05),
        )
