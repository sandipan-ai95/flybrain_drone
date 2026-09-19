"""Drone simulator.

Milestone 1 provides :class:`SimplePhysicsSim` — a lightweight point-mass
quadrotor with gravity, drag, thrust, and rotational dynamics driven directly
by commanded angular accelerations. Milestone 4 swaps this for a PyBullet
implementation exposing the same :class:`DroneSimulator` interface.
"""
from __future__ import annotations
from typing import Protocol
import math

from .physics import DroneState, DroneCommand, clamp_command


GRAVITY = 9.81
DRAG_LIN = 0.15
DRAG_ANG = 1.5


class DroneSimulator(Protocol):
    state: DroneState
    def step(self, cmd: DroneCommand, dt: float) -> DroneState: ...
    def reset(self) -> None: ...


class SimplePhysicsSim:
    """Point-mass quadrotor. Pitch tilts the thrust vector to produce
    horizontal acceleration; roll does the same laterally. Yaw rotates the
    heading. Very approximate but sufficient for closed-loop control demos."""

    def __init__(self, initial: DroneState | None = None) -> None:
        self._initial = initial or DroneState()
        self.state = DroneState(**vars(self._initial))
        self._bounds = ((-20.0, 20.0), (-20.0, 20.0), (0.0, 10.0))

    def reset(self) -> None:
        self.state = DroneState(**vars(self._initial))

    def step(self, cmd: DroneCommand, dt: float) -> DroneState:
        cmd = clamp_command(cmd)
        s = self.state
        # Angular dynamics
        s.wx += (cmd.roll  - DRAG_ANG * s.wx) * dt
        s.wy += (cmd.pitch - DRAG_ANG * s.wy) * dt
        s.wz += (cmd.yaw   - DRAG_ANG * s.wz) * dt
        s.roll  += s.wx * dt
        s.pitch += s.wy * dt
        s.yaw   += s.wz * dt

        # Translational: thrust along body-z; small-angle tilt gives horizontal accel.
        thrust = GRAVITY + cmd.thrust           # command is *net* thrust
        # Body → world using yaw + small-angle roll/pitch
        cy, sy = math.cos(s.yaw), math.sin(s.yaw)
        ax_body =  thrust * math.sin(s.pitch)
        ay_body = -thrust * math.sin(s.roll)
        ax = cy * ax_body - sy * ay_body
        ay = sy * ax_body + cy * ay_body
        az = thrust * math.cos(s.roll) * math.cos(s.pitch) - GRAVITY

        s.vx += (ax - DRAG_LIN * s.vx) * dt
        s.vy += (ay - DRAG_LIN * s.vy) * dt
        s.vz += (az - DRAG_LIN * s.vz) * dt

        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt

        # Boundaries / ground collision.
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._bounds
        s.collided = False
        if s.z <= zmin:
            s.z = zmin; s.vz = max(s.vz, 0.0); s.collided = True
        if not (xmin <= s.x <= xmax) or not (ymin <= s.y <= ymax) or s.z >= zmax:
            s.collided = True

        s.t += dt
        return s
