"""Trajectory logging."""
from __future__ import annotations
import csv
from pathlib import Path
from flybrain.drone.physics import DroneState


class TrajectoryLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["t", "x", "y", "z", "vx", "vy", "vz",
                          "roll", "pitch", "yaw", "collided"])

    def log(self, s: DroneState) -> None:
        self._w.writerow([s.t, s.x, s.y, s.z, s.vx, s.vy, s.vz,
                          s.roll, s.pitch, s.yaw, int(s.collided)])

    def close(self) -> None:
        self._fh.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
