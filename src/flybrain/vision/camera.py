"""Camera abstraction.

Milestone 1: :class:`SyntheticCamera` procedurally renders a scene of colored
targets that depend on drone pose. Milestone 4 will swap in a PyBullet camera
implementing the same :class:`Camera` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class CameraFrame:
    rgb: np.ndarray          # (H, W, 3) uint8
    t: float


class Camera(Protocol):
    def capture(self, drone_state) -> CameraFrame: ...


class SyntheticCamera:
    """Renders a bright target on a dark background.

    Coordinates: the target is a fixed point in world space. Its projected
    position on the camera plane depends on drone x/y (yaw ignored for now).
    """
    def __init__(self, resolution=(32, 32), target_xy=(3.0, 0.0)) -> None:
        self.h, self.w = int(resolution[0]), int(resolution[1])
        self.target_xy = np.asarray(target_xy, dtype=np.float32)

    def capture(self, drone_state) -> CameraFrame:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Vector from drone to target in xy plane; project to a horizontal FOV.
        dx = self.target_xy[0] - drone_state.x
        dy = self.target_xy[1] - drone_state.y
        # Yaw: rotate world vector into camera frame.
        c, s = np.cos(-drone_state.yaw), np.sin(-drone_state.yaw)
        fx = c * dx - s * dy   # forward (out of camera)
        rx = s * dx + c * dy   # right
        if fx <= 0.1:
            return CameraFrame(rgb=img, t=drone_state.t)  # target behind
        # Perspective: angle relative to forward.
        angle = np.arctan2(rx, fx)                 # radians
        fov = np.deg2rad(80)
        u = int((0.5 - angle / fov) * self.w)      # target column
        # Target size grows as we get closer.
        dist = float(np.hypot(fx, rx))
        radius = max(1, int(min(self.h, self.w) * 0.5 / max(dist, 0.5)))
        v = self.h // 2 + int((drone_state.z - 1.0) * (self.h * 0.2))
        # Rasterise a filled circle.
        yy, xx = np.ogrid[:self.h, :self.w]
        mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2
        img[mask] = (255, 200, 60)
        return CameraFrame(rgb=img, t=drone_state.t)
