"""Neural activity panel (per-population spike counts, firing rates)."""
from __future__ import annotations
import numpy as np
from flybrain.connectome.annotations import NeuronAnnotations


def population_activity(spikes: np.ndarray, ann: NeuronAnnotations) -> dict[str, float]:
    """Mean instantaneous activity per population."""
    return {name: float(spikes[p.neuron_indices].mean()) if p.size() else 0.0
            for name, p in ann.populations.items()}
