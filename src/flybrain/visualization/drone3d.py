"""Simple 3-D quadrotor mesh drawn with matplotlib.

Rendered as:
    * a central body sphere
    * 4 arms (X configuration) as thick line segments
    * 4 rotor discs whose color pulses with the current thrust
    * an orientation axis triad so roll/pitch/yaw are visible

Positions/orientations are updated every frame from a :class:`DroneState`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class Drone3DMesh:
    """Persistent 3D mesh handle. Call :meth:`update` each frame."""
    ax: object       # Axes3D
    arm_len: float = 0.45
    rotor_radius: float = 0.18
    body_color: str = "#f0f2ff"
    rotor_color: str = "#22ffcc"

    def __post_init__(self) -> None:
        # Body point
        self._body, = self.ax.plot([0], [0], [0], "o", color=self.body_color,
                                    markersize=9, zorder=5)
        # Arms (X-frame): 4 segments from centre to each rotor.
        self._arms = Line3DCollection(
            np.zeros((4, 2, 3), dtype=np.float32),
            colors="#c8c8f0", linewidths=2.2,
        )
        self.ax.add_collection3d(self._arms)
        # Rotor discs (approximated by a small triangle-fan line)
        theta = np.linspace(0, 2 * np.pi, 18)
        self._rotor_template = np.stack(
            [np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1
        ) * self.rotor_radius
        # Two spinning "blade" line segments per rotor for a visible propeller.
        self._blade_template = np.array([
            [-self.rotor_radius, 0.0, 0.0],
            [+self.rotor_radius, 0.0, 0.0],
        ], dtype=np.float32)
        self._rotor_lines = [
            self.ax.plot([], [], [], "-", color=self.rotor_color, lw=1.4)[0]
            for _ in range(4)
        ]
        self._blade_lines = [
            self.ax.plot([], [], [], "-", color="#ffffff", lw=2.0, alpha=0.9)[0]
            for _ in range(4)
        ]
        self._blade_phase = 0.0
        # Shadow ellipse on the ground plane so the drone looks *airborne*.
        self._shadow, = self.ax.plot([], [], [], "-", color=(0, 0, 0, 0.35),
                                     lw=2.5, zorder=1)
        # Small heading arrow so yaw is visible
        self._heading, = self.ax.plot([], [], [], "-", color="#ff5a5a", lw=2.2)

        # Precompute body-frame arm endpoints (X pattern)
        L = self.arm_len
        self._arm_endpoints_body = np.array([
            [+L, +L, 0.0],
            [-L, +L, 0.0],
            [-L, -L, 0.0],
            [+L, -L, 0.0],
        ], dtype=np.float32)

    # ----------------------------------------------------------------- update
    def update(self, x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float,
               thrust: float = 0.0) -> None:
        R = _rotation_matrix(roll, pitch, yaw)
        centre = np.array([x, y, z], dtype=np.float32)

        # Arms
        world_ends = (R @ self._arm_endpoints_body.T).T + centre
        segs = np.stack([np.tile(centre, (4, 1)), world_ends], axis=1)
        self._arms.set_segments(segs)

        # Body
        self._body.set_data([x], [y])
        self._body.set_3d_properties([z])

        # Heading arrow (body +x rotated to world)
        head = centre + (R @ np.array([0.9, 0.0, 0.0]))
        self._heading.set_data([centre[0], head[0]], [centre[1], head[1]])
        self._heading.set_3d_properties([centre[2], head[2]])

        # Rotor discs — pulse color with thrust
        t01 = float(np.clip((thrust + 10.0) / 20.0, 0.0, 1.0))
        rotor_rgb = (0.13 + 0.8 * t01, 1.0 - 0.3 * t01, 0.6 + 0.3 * t01)
        # Spin the blades faster with thrust.
        self._blade_phase = (self._blade_phase + 0.6 + 1.8 * t01) % (2 * np.pi)
        c, s = np.cos(self._blade_phase), np.sin(self._blade_phase)
        spin = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for i in range(4):
            world_ring = (R @ self._rotor_template.T).T + world_ends[i]
            self._rotor_lines[i].set_data(world_ring[:, 0], world_ring[:, 1])
            self._rotor_lines[i].set_3d_properties(world_ring[:, 2])
            self._rotor_lines[i].set_color(rotor_rgb)

            blade_world = (R @ (spin @ self._blade_template.T)).T + world_ends[i]
            self._blade_lines[i].set_data(blade_world[:, 0], blade_world[:, 1])
            self._blade_lines[i].set_3d_properties(blade_world[:, 2])

        # Ground shadow (ellipse squashed by altitude)
        alt = max(z, 0.05)
        shadow_r = 0.6 + 0.05 * alt
        theta = np.linspace(0, 2 * np.pi, 24)
        sx = x + shadow_r * np.cos(theta)
        sy = y + shadow_r * np.sin(theta) * 0.7
        sz = np.zeros_like(sx)
        self._shadow.set_data(sx, sy)
        self._shadow.set_3d_properties(sz)

