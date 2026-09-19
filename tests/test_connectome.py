import numpy as np
from flybrain.connectome.graph import Connectome, SynapticEdges


def test_connectome_construction_and_propagation():
    edges = SynapticEdges(
        pre=np.array([0, 0, 1], np.int32),
        post=np.array([1, 2, 2], np.int32),
        weights=np.array([1.0, 0.5, 2.0], np.float32),
    )
    c = Connectome(3, edges)
    s = np.array([1.0, 0.0, 0.0], np.float32)
    out = c.propagate(s)
    # Neuron 1 gets 1.0 from 0; neuron 2 gets 0.5 from 0.
    assert np.allclose(out, [0.0, 1.0, 0.5])


def test_edge_direction_and_degrees():
    edges = SynapticEdges(np.array([0, 0], np.int32),
                          np.array([1, 2], np.int32),
                          np.array([1.0, 1.0], np.float32))
    c = Connectome(3, edges)
    assert c.out_degree()[0] == 2
    assert c.in_degree()[1] == 1


def test_synthetic_visual_motor_sizes():
    c = Connectome.synthetic_visual_motor(n_visual=8, n_inter=16, n_motor=4)
    assert c.num_neurons == 28
    assert c.W.shape == (28, 28)
    assert c.edges.pre.size > 0
