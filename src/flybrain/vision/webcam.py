"""OpenCV webcam capture (spec §20).

Safety: this input only drives the *simulated* drone. Never wired to hardware.

macOS AVFoundation quirks handled here:
  * First cap.read() often returns (False, None) or a black frame — warm up.
  * VideoCapture(0) may open but hand back only None frames; probe more indices.
  * **Continuity Camera** (iPhone) often grabs index 0 and returns a tall
    portrait frame at very high resolution. When ``prefer_builtin=True`` we
    probe several indices, score each by aspect ratio + resolution, and pick
    the one that looks like a landscape laptop cam.
"""
from __future__ import annotations

import time
from typing import Iterable
import numpy as np


def probe_devices(max_index: int = 5, warmup_s: float = 1.0):
    """Return list of ``(index, width, height)`` for every opened camera."""
    import cv2
    out = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release(); continue
        frame = None
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            ok, bgr = cap.read()
            if ok and bgr is not None and np.any(bgr):
                frame = bgr; break
            time.sleep(0.03)
        cap.release()
        if frame is not None:
            out.append((i, frame.shape[1], frame.shape[0]))
    return out


class WebcamCapture:
    def __init__(
        self,
        device: int | None = None,
        resolution: tuple[int, int] = (480, 640),
        warmup_timeout_s: float = 3.0,
        try_devices: Iterable[int] | None = None,
        prefer_builtin: bool = True,
    ):
        try:
            import cv2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "opencv-python not installed. `pip install -e '.[gesture]'`"
            ) from e
        import cv2
        self._cv2 = cv2

        # ---- Build the candidate list --------------------------------------
        if device is not None:
            candidates = [device]
        else:
            candidates = list(try_devices) if try_devices is not None else [0, 1, 2, 3]

        # ---- Probe every candidate, keep the ones that produce a frame -----
        probed: list[tuple[int, int, int, np.ndarray, "cv2.VideoCapture"]] = []
        errors: list[str] = []
        for idx in candidates:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                errors.append(f"device {idx}: VideoCapture failed to open")
                continue
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[1])
            frame = self._warmup(cap, warmup_timeout_s)
            if frame is None:
                cap.release()
                errors.append(
                    f"device {idx}: opened but never produced a valid frame "
                    f"within {warmup_timeout_s:.1f}s"
                )
                continue
            probed.append((idx, frame.shape[1], frame.shape[0], frame, cap))

        if not probed:
            joined = "\n  ".join(errors)
            raise RuntimeError(
                "Cannot open any webcam. Tried devices: "
                f"{candidates}\n  {joined}\n"
                "On macOS: System Settings → Privacy & Security → Camera → "
                "enable your terminal / VS Code."
            )

        # ---- Score & pick the best camera ----------------------------------
        # Penalise portrait (h > w) frames and very-high-resolution feeds
        # — those are almost always the iPhone via Continuity Camera.
        def _score(item):
            idx, w, h, _, _ = item
            portrait_penalty = 100.0 if h > w else 0.0
            hires_penalty    = max(0.0, (w * h - 1_280 * 720) / 1e5)
            device_penalty   = 0.0 if idx == device else 0.5 * idx
            return portrait_penalty + hires_penalty + device_penalty

        print("[webcam] available devices:")
        for idx, w, h, _, _ in probed:
            tag = "  ← PORTRAIT (likely iPhone/Continuity)" if h > w else ""
            print(f"[webcam]   device {idx}: {w}x{h}{tag}")

        chosen = min(probed, key=_score) if prefer_builtin else probed[0]
        for idx, _, _, _, cap in probed:
            if cap is not chosen[4]:
                cap.release()

        self.cap = chosen[4]
        self.device = chosen[0]
        print(f"[webcam] ✅ using device {self.device} — {chosen[1]}x{chosen[2]}")

    def _warmup(self, cap, timeout_s: float):
        cv2 = self._cv2
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ok, bgr = cap.read()
            if ok and bgr is not None and np.any(bgr):
                return cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
            time.sleep(0.05)
        return None

    def read(self) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return self._cv2.cvtColor(self._cv2.flip(bgr, 1), self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self.cap.release()

    def __enter__(self): return self
    def __exit__(self, *a): self.release()
