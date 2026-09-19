"""Task evaluation metrics (spec §13). Milestones 8–10.

Provides objective metrics used across all controller variants:
    * task success rate
    * collision rate
    * path length
    * reaction time
    * control smoothness
    * energy consumption
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RunMetrics:
    success: bool
    collisions: int
    path_length: float
    mean_speed: float
    control_smoothness: float
    total_energy: float


def summarise(trajectory: np.ndarray, commands: np.ndarray, collided: np.ndarray,
              success: bool) -> RunMetrics:
    """
    trajectory: (T, 3) xyz.
    commands:   (T, 4) thrust/roll/pitch/yaw.
    collided:   (T,) bool.
    """
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    path_length = float(dists.sum())
    mean_speed = float(dists.mean()) if dists.size else 0.0
    cmd_diffs = np.diff(commands, axis=0)
    smoothness = -float(np.abs(cmd_diffs).mean()) if cmd_diffs.size else 0.0
    energy = float((commands ** 2).sum())
    return RunMetrics(success=success,
                      collisions=int(collided.sum()),
                      path_length=path_length, mean_speed=mean_speed,
                      control_smoothness=smoothness, total_energy=energy)
