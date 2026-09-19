"""Connectome ablation experiments (spec §14). Milestone 8.

Utilities to remove a fraction of neurons (random / hub-targeted / by
population) from a :class:`Connectome` and produce an ablated copy.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from flybrain.connectome.graph import Connectome, SynapticEdges


def ablate_random(conn: Connectome, fraction: float, rng: np.random.Generator | None = None) -> Connectome:
    rng = rng or np.random.default_rng(0)
    N = conn.num_neurons
    n_drop = int(N * fraction)
    if n_drop <= 0:
        return conn
    drop = rng.choice(N, size=n_drop, replace=False)
    return _drop_neurons(conn, drop)


def ablate_hubs(conn: Connectome, fraction: float) -> Connectome:
    deg = conn.in_degree() + conn.out_degree()
    n_drop = int(conn.num_neurons * fraction)
    if n_drop <= 0:
        return conn
    drop = np.argsort(-deg)[:n_drop]
    return _drop_neurons(conn, drop)


def ablate_population(conn: Connectome, indices: np.ndarray) -> Connectome:
    return _drop_neurons(conn, np.asarray(indices, dtype=np.int64))


def _drop_neurons(conn: Connectome, drop: np.ndarray) -> Connectome:
    keep_mask = np.ones(conn.num_neurons, dtype=bool)
    keep_mask[drop] = False
    W = conn.W.tocoo()
    edge_keep = keep_mask[W.row] & keep_mask[W.col]
    pre = W.col[edge_keep]
    post = W.row[edge_keep]
    weights = W.data[edge_keep]
    # Zero-out dropped rows/cols but keep index space (simpler; performance-wise
    # fine for research runs). Reindexing is an option later.
    edges = SynapticEdges(pre=pre.astype(np.int32),
                          post=post.astype(np.int32),
                          weights=weights.astype(np.float32))
    return Connectome(conn.num_neurons, edges, conn.neuron_ids)
