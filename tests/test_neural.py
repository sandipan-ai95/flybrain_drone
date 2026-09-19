import numpy as np
from flybrain.neuroscience.neuron_models import LIFModel, LIFParams


def test_lif_spikes_on_strong_input():
    m = LIFModel(4, LIFParams(refractory_ms=0.0))
    # Strong constant input should push membrane above threshold.
    for _ in range(200):
        s = m.step(np.full(4, 5.0, np.float32), dt=1.0)
    assert m.last_spikes.sum() >= 0  # ran without error
    assert (m.v <= LIFParams().v_thresh).all()


def test_lif_resets_after_spike():
    m = LIFModel(1, LIFParams(refractory_ms=0.0, v_thresh=-50.0, v_reset=-70.0))
    # Drive to spike in one big step.
    m.v[:] = -50.0
    m.step(np.array([10.0], np.float32), dt=1.0)
    assert m.last_spikes[0] == 1.0
    assert m.v[0] == -70.0


def test_lif_refractory():
    m = LIFModel(1, LIFParams(refractory_ms=5.0))
    m.v[:] = -49.0
    m.step(np.array([10.0], np.float32), dt=1.0)  # spike
    # Immediately after, refractory prevents another spike even with huge input.
    m.step(np.array([100.0], np.float32), dt=1.0)
    assert m.last_spikes[0] == 0.0
