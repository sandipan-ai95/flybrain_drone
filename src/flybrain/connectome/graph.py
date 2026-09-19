"""Directed synaptic graph G = (V, E).

Design goals (from spec §2):
    * sparse representation (SciPy CSR) — avoid dense N×N matrices
    * memory-mapped / chunked friendly (arrays only; no Python-object edges)
    * pluggable: synthetic graph today, FlyWire/hemibrain tomorrow
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class SynapticEdges:
    """Directed weighted edges. `weights[i]` is the synaptic strength of
    edge ``(pre[i] -> post[i])``. In real connectomes this is typically the
    synapse count between the neuron pair."""
    pre: np.ndarray   # int32, shape (E,)
    post: np.ndarray  # int32, shape (E,)
    weights: np.ndarray  # float32, shape (E,)

    def __post_init__(self) -> None:
        assert self.pre.shape == self.post.shape == self.weights.shape, "edge arrays mismatch"


class Connectome:
    """Sparse directed connectome.

    The core data structure is a CSR matrix ``W`` of shape ``(N, N)`` where
    ``W[post, pre]`` is the synaptic weight from ``pre`` to ``post``. This
    orientation lets us compute the postsynaptic input current for a spike
    vector ``s`` as ``W @ s`` in one sparse mat-vec.
    """

    def __init__(
        self,
        num_neurons: int,
        edges: SynapticEdges,
        neuron_ids: np.ndarray | None = None,
    ) -> None:
        self.num_neurons = int(num_neurons)
        self.edges = edges
        self.neuron_ids = (
            neuron_ids if neuron_ids is not None else np.arange(num_neurons, dtype=np.int64)
        )
        # W[post, pre]
        self.W: sp.csr_matrix = sp.csr_matrix(
            (edges.weights.astype(np.float32),
             (edges.post.astype(np.int32), edges.pre.astype(np.int32))),
            shape=(self.num_neurons, self.num_neurons),
        )

    # ------------------------------------------------------------------ ops
    def propagate(self, spikes: np.ndarray) -> np.ndarray:
        """Return postsynaptic input currents for a binary/float spike vector."""
        return self.W @ spikes

    def out_degree(self) -> np.ndarray:
        # Column sums of W[post, pre] == number of postsynaptic targets per pre.
        return np.asarray((self.W != 0).sum(axis=0)).ravel()

    def in_degree(self) -> np.ndarray:
        return np.asarray((self.W != 0).sum(axis=1)).ravel()

    # ---------------------------------------------------------------- factory
    @staticmethod
    def synthetic_visual_motor(
        n_visual: int = 64,
        n_inter: int = 128,
        n_motor: int = 8,
        density: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> "Connectome":
        """A tiny 3-layer feed-forward random connectome for Milestone 1.

        Not biological. Used purely to bring up the closed loop end-to-end so
        the rest of the pipeline can be developed independently. Replace with
        :func:`flybrain.connectome.loader.load_flywire_subgraph` in Milestone 2.
        """
        rng = rng or np.random.default_rng(0)
        N = n_visual + n_inter + n_motor
        pre_l, post_l, w_l = [], [], []

        def _connect(pre_range, post_range, p, w_scale=1.0):
            for pre in pre_range:
                mask = rng.random(len(post_range)) < p
                posts = np.asarray(post_range)[mask]
                if posts.size == 0:
                    continue
                pre_l.append(np.full(posts.size, pre, dtype=np.int32))
                post_l.append(posts.astype(np.int32))
                w_l.append((rng.random(posts.size).astype(np.float32) * w_scale))

        visual = range(0, n_visual)
        inter = range(n_visual, n_visual + n_inter)
        motor = range(n_visual + n_inter, N)

        _connect(visual, inter, density * 4, w_scale=1.0)
        _connect(inter, inter, density,     w_scale=0.5)   # recurrence
        _connect(inter, motor, density * 4, w_scale=1.5)

        edges = SynapticEdges(
            pre=np.concatenate(pre_l) if pre_l else np.zeros(0, np.int32),
            post=np.concatenate(post_l) if post_l else np.zeros(0, np.int32),
            weights=np.concatenate(w_l) if w_l else np.zeros(0, np.float32),
        )
        return Connectome(N, edges)

    # ------------------------------------------------------------------ gesture
    @staticmethod
    def gesture_wired(
        n_per_sensory: int = 12,
        n_inter: int = 96,
        n_per_motor: int = 10,
        density: float = 0.05,
        bias_strength: float = 5.0,
        rng: np.random.Generator | None = None,
    ) -> tuple["Connectome", dict[str, np.ndarray]]:
        """Six-channel gesture connectome.

        Sensory sub-populations: ``up, down, left, right, rot_cw, rot_ccw``.
        Motor sub-populations:   ``thrust, descent, roll_l, roll_r, yaw_cw, yaw_ccw``.
        Each sensory channel projects strongly to its matching motor channel
        via a shared random interneuron pool. This is a **hand-wired
        computational mapping**, not biological — see §3 in the README.

        Returns ``(connectome, population_indices_dict)``.
        """
        rng = rng or np.random.default_rng(0)
        sensory_names = ["up", "down", "left", "right", "rot_cw", "rot_ccw"]
        motor_names   = ["thrust", "descent", "roll_l", "roll_r", "yaw_cw", "yaw_ccw"]
        assert len(sensory_names) == len(motor_names)

        pops: dict[str, np.ndarray] = {}
        cursor = 0
        for name in sensory_names:
            pops[f"sensory_{name}"] = np.arange(cursor, cursor + n_per_sensory, dtype=np.int32)
            cursor += n_per_sensory
        pops["interneuron"] = np.arange(cursor, cursor + n_inter, dtype=np.int32)
        cursor += n_inter
        for name in motor_names:
            pops[f"motor_{name}"] = np.arange(cursor, cursor + n_per_motor, dtype=np.int32)
            cursor += n_per_motor
        N = cursor

        pre_l, post_l, w_l = [], [], []

        # Partition interneurons into one "channel" pool per gesture (+ a shared pool).
        inter = pops["interneuron"]
        n_channels = len(sensory_names)
        channel_pools = np.array_split(inter[: (len(inter) // n_channels) * n_channels], n_channels)

        def _connect(pre_ids, post_ids, p, w_scale):
            for pre in pre_ids:
                mask = rng.random(len(post_ids)) < p
                posts = np.asarray(post_ids)[mask]
                if posts.size == 0:
                    continue
                pre_l.append(np.full(posts.size, pre, dtype=np.int32))
                post_l.append(posts.astype(np.int32))
                w_l.append(rng.random(posts.size).astype(np.float32) * w_scale)

        # Sensory → its channel interneurons (strong) and to others (weak) + recurrence.
        for i, s_name in enumerate(sensory_names):
            s_ids = pops[f"sensory_{s_name}"]
            own_inter = channel_pools[i]
            other_inter = np.concatenate([channel_pools[j] for j in range(n_channels) if j != i])
            _connect(s_ids, own_inter,   p=0.6,     w_scale=bias_strength)
            _connect(s_ids, other_inter, p=density, w_scale=0.3)
        # Interneuron recurrence (weak).
        _connect(inter, inter, p=density, w_scale=0.3)
        # Channel interneurons → matching motor sub-pop (strong); crosstalk (weak).
        for i, m_name in enumerate(motor_names):
            m_ids = pops[f"motor_{m_name}"]
            own_inter = channel_pools[i]
            other_inter = np.concatenate([channel_pools[j] for j in range(n_channels) if j != i])
            _connect(own_inter,   m_ids, p=0.7,     w_scale=bias_strength * 1.5)
            _connect(other_inter, m_ids, p=density, w_scale=0.2)

        edges = SynapticEdges(
            pre=np.concatenate(pre_l), post=np.concatenate(post_l),
            weights=np.concatenate(w_l),
        )
        return Connectome(N, edges), pops
