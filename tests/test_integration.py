"""End-to-end integration test: image → encoder → sim → decoder → drone."""
import numpy as np
from flybrain.connectome.loader import load_synthetic
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator
from flybrain.vision.camera import SyntheticCamera
from flybrain.vision.encoder import VisualEncoder, EncoderConfig
from flybrain.decoding.motor_decoder import PopulationMotorDecoder
from flybrain.drone.simulator import SimplePhysicsSim
from flybrain.drone.physics import DroneState


def test_closed_loop_runs():
    conn, ann = load_synthetic(seed=1)
    sim = NeuralSimulator(conn, LIFModel(conn.num_neurons), dt_ms=1.0)
    enc = VisualEncoder(EncoderConfig(resolution=(8, 8)),
                        num_neurons=conn.num_neurons,
                        sensory_indices=ann.get("visual").neuron_indices,
                        rng=np.random.default_rng(1))
    cam = SyntheticCamera(resolution=(16, 16))
    dec = PopulationMotorDecoder(ann)
    drone = SimplePhysicsSim(DroneState(z=1.0))

    for _ in range(20):
        frame = cam.capture(drone.state)
        cur = enc.encode(frame.rgb)
        for _ in range(5):
            sim.step(cur)
        cmd = dec.decode(sim.state.spikes, dt_ms=5.0)
        drone.step(cmd, dt=0.01)

    # Simulation ran without errors and produced finite state.
    assert np.isfinite([drone.state.x, drone.state.y, drone.state.z]).all()
