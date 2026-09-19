"""Tests for the gesture connectome + gesture-driven closed loop."""
import numpy as np

from flybrain.connectome.loader import load_gesture
from flybrain.vision.gesture import Gesture, ScriptedGestureRecognizer, classify_landmarks
from flybrain.vision.stimulus import GestureStimulus
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator


def test_gesture_connectome_populations_present():
    conn, ann = load_gesture(seed=0)
    for name in ["sensory_up", "sensory_down", "sensory_left", "sensory_right",
                 "sensory_rot_cw", "sensory_rot_ccw",
                 "motor_thrust", "motor_descent", "motor_roll_l", "motor_roll_r",
                 "motor_yaw_cw", "motor_yaw_ccw", "interneuron"]:
        assert name in ann.populations, f"missing population {name}"
    assert conn.W.nnz > 0


def test_scripted_recognizer_cycles():
    r = ScriptedGestureRecognizer(sequence=[Gesture.UP, Gesture.DOWN], hold_frames=1)
    assert r.detect().gesture == Gesture.UP
    assert r.detect().gesture == Gesture.DOWN


def test_stimulus_targets_correct_population():
    _, ann = load_gesture(seed=0)
    stim = GestureStimulus(ann, drive_current=20.0)
    cur = stim.encode(Gesture.UP)
    up_idx = ann.get("sensory_up").neuron_indices
    other_idx = ann.get("sensory_down").neuron_indices
    assert np.all(cur[up_idx] == 20.0)
    assert np.all(cur[other_idx] == 0.0)


def test_up_gesture_drives_thrust_population_more_than_descent():
    """Structural sanity: after enough steps, sensory_up drives motor_thrust
    more strongly than motor_descent (the gesture wiring biases them)."""
    conn, ann = load_gesture(seed=0)
    sim = NeuralSimulator(conn, LIFModel(conn.num_neurons), dt_ms=1.0)
    stim = GestureStimulus(ann, drive_current=20.0)
    cur = stim.encode(Gesture.UP)
    thrust_idx  = ann.get("motor_thrust").neuron_indices
    descent_idx = ann.get("motor_descent").neuron_indices
    thrust_spikes = descent_spikes = 0
    for _ in range(400):
        s = sim.step(cur)
        thrust_spikes  += int(s[thrust_idx].sum())
        descent_spikes += int(s[descent_idx].sum())
    assert thrust_spikes > descent_spikes, \
        f"UP gesture failed to bias thrust>descent (thrust={thrust_spikes}, descent={descent_spikes})"


def test_classify_landmarks_shape_ok():
    # 21 random landmarks — just verify the classifier returns something sensible.
    rng = np.random.default_rng(0)
    pts = rng.random((21, 2)).astype(np.float32)
    g, conf = classify_landmarks(pts)
    assert isinstance(g, Gesture)
    assert 0.0 <= conf <= 1.0
