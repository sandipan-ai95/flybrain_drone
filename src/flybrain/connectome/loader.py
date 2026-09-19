"""Connectome data ingestion pipeline.

Milestone 1: stub that only returns the synthetic connectome.
Milestone 2 will add real loaders:

    * FlyWire (public CC BY 4.0 release, CSV/parquet from codex.flywire.ai)
    * hemibrain via neuprint-python
    * MANC for motor-neuron populations

Design: the loader returns ``(Connectome, NeuronAnnotations)`` so the rest of
the pipeline never depends on which dataset is behind it.
"""
from __future__ import annotations
from pathlib import Path

from .graph import Connectome
from .annotations import NeuronAnnotations


DEFAULT_DATA_DIR = Path("data")


def load_synthetic(n_visual=64, n_inter=128, n_motor=8, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    conn = Connectome.synthetic_visual_motor(n_visual, n_inter, n_motor, rng=rng)
    ann = NeuronAnnotations.synthetic(n_visual, n_inter, n_motor)
    return conn, ann


def load_gesture(seed: int = 0):
    """Six-channel gesture connectome + annotations. See :meth:`Connectome.gesture_wired`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    conn, pops = Connectome.gesture_wired(rng=rng)
    ann = NeuronAnnotations(num_neurons=conn.num_neurons)
    for name, idx in pops.items():
        ann.add_population(name, idx)
    # Convenience aliases used by the motor decoder and visual encoder.
    all_sensory = np.concatenate([pops[k] for k in pops if k.startswith("sensory_")])
    all_motor   = np.concatenate([pops[k] for k in pops if k.startswith("motor_")])
    ann.add_population("visual", all_sensory)   # so VisualEncoder targets sensory pool
    ann.add_population("motor", all_motor)
    return conn, ann


def load_flywire_subgraph(data_dir: Path = DEFAULT_DATA_DIR, cell_types: list[str] | None = None):
    """Load a real FlyWire visual-motor subgraph (Milestone 2).

    Delegates to :mod:`flybrain.connectome.flywire_loader`. Requires the user
    to have downloaded the FlyWire CSVs to ``data/raw/flywire/`` from
    https://codex.flywire.ai/api/download (free sign-in).
    """
    from .flywire_loader import load_flywire, FLYWIRE_DIR
    return load_flywire(FLYWIRE_DIR if data_dir is DEFAULT_DATA_DIR else data_dir / "raw/flywire")
