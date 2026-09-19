"""Drone state, command dataclasses, physics primitives."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DroneState:
    x: float = 0.0; y: float = 0.0; z: float = 1.0
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0
    wx: float = 0.0; wy: float = 0.0; wz: float = 0.0
    t: float = 0.0
    collided: bool = False


@dataclass
class DroneCommand:
    thrust: float = 0.0   # net vertical accel above gravity, m/s^2
    roll: float = 0.0     # target angular accel around x
    pitch: float = 0.0    # target angular accel around y (forward tilt)
    yaw: float = 0.0      # target angular accel around z


THRUST_LIMIT = 15.0
ANG_LIMIT = 5.0


def clamp_command(cmd: DroneCommand) -> DroneCommand:
    def _c(v, lim): return max(-lim, min(lim, float(v)))
    return DroneCommand(
        thrust=_c(cmd.thrust, THRUST_LIMIT),
        roll=_c(cmd.roll, ANG_LIMIT),
        pitch=_c(cmd.pitch, ANG_LIMIT),
        yaw=_c(cmd.yaw, ANG_LIMIT),
    )
