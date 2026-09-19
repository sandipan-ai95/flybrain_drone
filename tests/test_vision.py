import numpy as np
from flybrain.vision.camera import SyntheticCamera
from flybrain.vision.encoder import VisualEncoder, EncoderConfig
from flybrain.vision.optical_flow import frame_difference
from flybrain.drone.physics import DroneState


def test_camera_returns_image():
    cam = SyntheticCamera(resolution=(16, 16))
    f = cam.capture(DroneState())
    assert f.rgb.shape == (16, 16, 3)


def test_encoder_maps_to_sensory_neurons():
    N = 40
    sens = np.arange(10)
    enc = VisualEncoder(EncoderConfig(resolution=(8, 8)), num_neurons=N, sensory_indices=sens)
    img = np.full((16, 16, 3), 255, np.uint8)
    out = enc.encode(img)
    assert out.shape == (N,)
    assert (out[10:] == 0).all()   # non-sensory untouched
    assert out[:10].sum() > 0      # bright image produces positive current


def test_frame_difference_zero_for_identical():
    a = np.zeros((4, 4, 3), np.uint8)
    assert frame_difference(a, a).sum() == 0
