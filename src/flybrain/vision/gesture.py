"""Hand-gesture recognition.

Uses MediaPipe Hands when available; otherwise a keyboard/scripted fallback so
the pipeline runs on any machine (e.g. CI, headless servers).

Recognized gestures (mapped 1:1 onto sensory sub-populations of the gesture
connectome, see :func:`flybrain.connectome.loader.load_gesture`):

    UP        — palm open, pointing up
    DOWN      — palm open, pointing down
    LEFT      — palm open, pointing left
    RIGHT     — palm open, pointing right
    ROTATE_CW — index finger only (fist + index up), rotate right
    ROTATE_CCW— pinky finger only, rotate left
    NONE      — no confident gesture

Scientific note: this classification is a *heuristic* on landmark geometry.
It is not a neuroscience claim — it just labels what your hand is doing so we
can drive the connectome's sensory channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class Gesture(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ROTATE_CW = "rot_cw"
    ROTATE_CCW = "rot_ccw"
    NONE = "none"


ALL_GESTURES = [Gesture.UP, Gesture.DOWN, Gesture.LEFT, Gesture.RIGHT,
                Gesture.ROTATE_CW, Gesture.ROTATE_CCW]


@dataclass
class GestureResult:
    gesture: Gesture
    confidence: float
    landmarks: Optional[np.ndarray] = None   # (21, 2) normalized xy, if available


# ---------------------------------------------------------------- MediaPipe backend
class MediaPipeGestureRecognizer:
    """Recognizes gestures from an RGB frame using MediaPipe Hands.

    Compatible with both APIs:
      * legacy ``mediapipe.solutions.hands`` (mediapipe 0.10.x)
      * new    ``mediapipe.tasks.vision.HandLandmarker`` (mediapipe ≥ 0.10.9 / 1.x)
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mediapipe not installed. `pip install -e '.[gesture]'`"
            ) from e
        import mediapipe as mp
        self._mp = mp
        self._backend = None

        # ---- Try the *new* Tasks API first (mediapipe 1.x)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            model_path = self._ensure_hand_landmarker_model()
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._tasks = mp_vision.HandLandmarker.create_from_options(options)
            self._backend = "tasks"
            print("[gesture] backend: mediapipe.tasks HandLandmarker")
            return
        except Exception as e_new:
            new_err = repr(e_new)

        # ---- Fall back to the legacy Solutions API (mediapipe 0.10.x)
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
            )
            self._backend = "solutions"
            print("[gesture] backend: mediapipe.solutions.hands (legacy)")
            return
        except Exception as e_old:
            raise RuntimeError(
                f"MediaPipe hand detection unavailable.\n"
                f"  new API (tasks) failed: {new_err}\n"
                f"  old API (solutions) failed: {e_old!r}\n"
                f"Installed mediapipe version: {getattr(mp, '__version__', '?')}"
            ) from e_old

    # -------- model download helper for the tasks backend
    def _ensure_hand_landmarker_model(self):
        """Return a Path to hand_landmarker.task, downloading it if needed."""
        from pathlib import Path
        import urllib.request
        cache_dir = Path.home() / ".cache" / "flybrain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "hand_landmarker.task"
        if not model_path.exists():
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/"
                   "hand_landmarker.task")
            print(f"[gesture] downloading MediaPipe model → {model_path}")
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def close(self) -> None:
        if self._backend == "tasks":
            try: self._tasks.close()
            except Exception: pass
        elif self._backend == "solutions":
            try: self._hands.close()
            except Exception: pass

    def detect(self, rgb: np.ndarray) -> GestureResult:
        if rgb is None:
            return GestureResult(Gesture.NONE, 0.0, None)

        # ---- new Tasks API
        if self._backend == "tasks":
            mp = self._mp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._tasks.detect(image)
            if not res.hand_landmarks:
                return GestureResult(Gesture.NONE, 0.0, None)
            lm = res.hand_landmarks[0]
            pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
            g, conf = classify_landmarks(pts)
            return GestureResult(g, conf, pts)

        # ---- legacy Solutions API
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return GestureResult(Gesture.NONE, 0.0, None)
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        g, conf = classify_landmarks(pts)
        return GestureResult(g, conf, pts)


def classify_landmarks(pts: np.ndarray) -> tuple[Gesture, float]:
    """Rule-based gesture classifier on 21 hand landmarks (normalized).

    Landmark indices follow MediaPipe:
        0 wrist; 4 thumb_tip; 8 index_tip; 12 middle_tip; 16 ring_tip; 20 pinky_tip;
        5 index_mcp; 9 middle_mcp; 13 ring_mcp; 17 pinky_mcp.
    """
    wrist = pts[0]
    # A finger is "extended" if its tip is farther from wrist than its PIP joint.
    def extended(tip_i, pip_i):
        return np.linalg.norm(pts[tip_i] - wrist) > 1.05 * np.linalg.norm(pts[pip_i] - wrist)

    idx_ext   = extended(8, 6)
    mid_ext   = extended(12, 10)
    ring_ext  = extended(16, 14)
    pinky_ext = extended(20, 18)

    # Rotation gestures: single-finger poses.
    if idx_ext and not (mid_ext or ring_ext or pinky_ext):
        return Gesture.ROTATE_CW, 0.9
    if pinky_ext and not (idx_ext or mid_ext or ring_ext):
        return Gesture.ROTATE_CCW, 0.9

    # Open palm: index + middle + ring extended → direction from wrist to middle_tip.
    if idx_ext and mid_ext and ring_ext:
        v = pts[12] - wrist                    # (dx, dy) in image coords (y grows down)
        dx, dy = float(v[0]), float(v[1])
        if abs(dy) > abs(dx):
            return (Gesture.UP, 0.85) if dy < 0 else (Gesture.DOWN, 0.85)
        else:
            return (Gesture.RIGHT, 0.85) if dx > 0 else (Gesture.LEFT, 0.85)

    return Gesture.NONE, 0.0


# ------------------------------------------------------------------- fallback
class ScriptedGestureRecognizer:
    """Cycles through a fixed gesture sequence — used when no webcam is
    available (CI, headless demos, unit tests)."""

    def __init__(self, sequence: list[Gesture] | None = None, hold_frames: int = 60):
        self.sequence = sequence or [Gesture.UP, Gesture.RIGHT, Gesture.LEFT,
                                     Gesture.ROTATE_CW, Gesture.DOWN, Gesture.NONE]
        self.hold_frames = hold_frames
        self._i = 0

    def close(self) -> None:  # for API parity
        pass

    def detect(self, rgb: np.ndarray | None = None) -> GestureResult:  # ignores rgb
        g = self.sequence[(self._i // self.hold_frames) % len(self.sequence)]
        self._i += 1
        return GestureResult(g, 1.0, None)
