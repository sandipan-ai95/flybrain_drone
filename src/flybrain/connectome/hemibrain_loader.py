"""Janelia hemibrain loader (neuprint).

The hemibrain v1.2.1 dataset (~25 k neurons, ~20 M synapses in half a fly
brain) is hosted by Janelia at https://neuprint.janelia.org and can be
queried without an account for read access. This is the fastest way to run
FlyBrain-Drone on a real Drosophila connectome without any manual downloads.

Requires the ``neuprint-python`` client:
    pip install neuprint-python
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.connectome.graph import Connectome, SynapticEdges


HEMIBRAIN_SERVER = "https://neuprint.janelia.org"
HEMIBRAIN_DATASET = "hemibrain:v1.2.1"


def load_hemibrain(
    max_neurons: Optional[int] = 5000,
    min_syn_weight: int = 5,
    token: Optional[str] = None,
) -> tuple[Connectome, NeuronAnnotations]:
    """Build a :class:`Connectome` from the Janelia hemibrain via neuprint.

    Parameters
    ----------
    max_neurons
        Take the top-N neurons by in-degree. 5000 is a comfortable laptop
        size (~250k edges after filtering).
    min_syn_weight
        Drop synaptic connections weaker than this many synapses.
    token
        Optional neuprint auth token (from your account page). Read queries
        work without a token but may be rate-limited.
    """
    try:
        from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
    except ImportError as e:
        raise ImportError(
            "neuprint-python not installed. `pip install neuprint-python`"
        ) from e

    print(f"[hemibrain] connecting to {HEMIBRAIN_SERVER} ({HEMIBRAIN_DATASET})")
    Client(HEMIBRAIN_SERVER, HEMIBRAIN_DATASET, token=token)

    # 1) Pull neurons with types + statuses of interest.
    print(f"[hemibrain] fetching top-{max_neurons} neurons by in-degree...")
    neurons_df, _ = fetch_neurons(NC(status="Traced"))
    neurons_df = neurons_df.sort_values("post", ascending=False)
    if max_neurons:
        neurons_df = neurons_df.head(max_neurons)
    neurons_df = neurons_df.reset_index(drop=True)
    print(f"[hemibrain]   kept {len(neurons_df):,} neurons")

    body_ids = neurons_df["bodyId"].to_numpy()
    id_to_ix = {int(v): i for i, v in enumerate(body_ids)}

    # 2) Adjacencies restricted to those bodies.
    print(f"[hemibrain] fetching adjacencies (syn≥{min_syn_weight})...")
    _, conn_df = fetch_adjacencies(
        sources=NC(bodyId=body_ids.tolist()),
        targets=NC(bodyId=body_ids.tolist()),
        min_total_weight=min_syn_weight,
    )
    conn_df = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    print(f"[hemibrain]   {len(conn_df):,} edges")

    pre = conn_df["bodyId_pre"].map(id_to_ix).to_numpy(dtype=np.int32)
    post = conn_df["bodyId_post"].map(id_to_ix).to_numpy(dtype=np.int32)
    w = conn_df["weight"].to_numpy(dtype=np.float32)

    conn = Connectome(
        num_neurons=len(body_ids),
        edges=SynapticEdges(pre=pre, post=post, weights=w),
        neuron_ids=body_ids.astype(np.int64),
    )
    ann = _build_annotations(neurons_df)
    print(f"[hemibrain] ✅ ready: N={conn.num_neurons:,}  edges={conn.W.nnz:,}")
    return conn, ann


def _build_annotations(neurons_df) -> NeuronAnnotations:
    ann = NeuronAnnotations(num_neurons=len(neurons_df))
    ct = neurons_df.get("type")
    if ct is not None:
        ann.cell_type = ct.fillna("").to_numpy()

    # Heuristic visual / descending / interneuron partition using hemibrain
    # cell-type name conventions:
    #   * visual neurons in the optic lobe often carry "L", "Mi", "Tm", "T4",
    #     "T5", "LC" prefixes.
    #   * descending neurons are labelled "DNa##", "DNb##", "DNp##".
    types = neurons_df.get("type", "").fillna("").to_numpy()
    is_desc   = np.array([t.startswith(("DNa", "DNb", "DNp", "DNg")) for t in types])
    is_visual = np.array([t.startswith(("L", "Mi", "Tm", "T4", "T5", "LC", "LPLC",
                                        "LPC", "MC")) for t in types])
    inter     = ~(is_desc | is_visual)

    visual_idx = np.where(is_visual)[0].astype(np.int32)
    desc_idx   = np.where(is_desc)[0].astype(np.int32)
    inter_idx  = np.where(inter)[0].astype(np.int32)

    if inter_idx.size:  ann.add_population("interneuron", inter_idx)
    if visual_idx.size: ann.add_population("visual", visual_idx)
    if desc_idx.size:   ann.add_population("motor",  desc_idx)

    # 6-way computational split for gesture channels (documented approximation,
    # same convention as flywire_loader.py).
    sensory_names = ["sensory_up", "sensory_down", "sensory_left",
                     "sensory_right", "sensory_rot_cw", "sensory_rot_ccw"]
    motor_names = ["motor_thrust", "motor_descent", "motor_roll_l",
                   "motor_roll_r", "motor_yaw_cw", "motor_yaw_ccw"]
    for name, part in zip(sensory_names, np.array_split(visual_idx, 6)):
        ann.add_population(name, np.asarray(part, np.int32))
    for name, part in zip(motor_names, np.array_split(desc_idx, 6)):
        ann.add_population(name, np.asarray(part, np.int32))
    return ann
