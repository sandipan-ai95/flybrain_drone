"""Gesture → sensory-neuron current injector.

Given a detected :class:`Gesture` and the gesture-connectome annotations,
return a full-length external-current vector that stimulates the matching
sensory sub-population.
"""
from __future__ import annotations
import numpy as np

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.vision.gesture import Gesture


GESTURE_TO_POP = {
    Gesture.UP:         "sensory_up",
    Gesture.DOWN:       "sensory_down",
    Gesture.LEFT:       "sensory_left",
    Gesture.RIGHT:      "sensory_right",
    Gesture.ROTATE_CW:  "sensory_rot_cw",
    Gesture.ROTATE_CCW: "sensory_rot_ccw",
}


class GestureStimulus:
    def __init__(self, ann: NeuronAnnotations, drive_current: float = 20.0):
        self.ann = ann
        self.drive = float(drive_current)

    def encode(self, gesture: Gesture, confidence: float = 1.0) -> np.ndarray:
        cur = np.zeros(self.ann.num_neurons, dtype=np.float32)
        pop_name = GESTURE_TO_POP.get(gesture)
        if pop_name is None or pop_name not in self.ann.populations:
            return cur
        idx = self.ann.get(pop_name).neuron_indices
        cur[idx] = self.drive * float(confidence)
        return cur
