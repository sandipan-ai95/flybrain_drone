"""Smoke test for the 3D brain viewer (no windows, no MP4)."""
from pathlib import Path
import numpy as np

from flybrain.connectome.loader import load_gesture
from flybrain.visualization.brain3d import (
    compute_3d_layout, Matplotlib3DBrain,
)


def test_layout_shape():
    conn, ann = load_gesture(seed=0)
    pos = compute_3d_layout(conn, ann)
    assert pos.shape == (conn.num_neurons, 3)
    # sensory should be on the left, motor on the right
    s = ann.get("sensory_up").neuron_indices
    m = ann.get("motor_thrust").neuron_indices
    assert pos[s, 0].mean() < pos[m, 0].mean()


def test_matplotlib_brain_updates_without_error(tmp_path: Path):
    conn, ann = load_gesture(seed=0)
    view = Matplotlib3DBrain(conn, ann, record_path=None, show=False, rotate=False)
    act = np.zeros(conn.num_neurons, np.float32)
    act[ann.get("sensory_up").neuron_indices] = 1.0
    view.update(act, title_suffix="test", info_lines=["hello"])
    view.close()


def test_plotly_html_written(tmp_path: Path):
    plotly = __import__("importlib").util.find_spec("plotly")
    if plotly is None:
        import pytest; pytest.skip("plotly not installed")
    from flybrain.visualization.brain3d import Plotly3DBrain
    conn, ann = load_gesture(seed=0)
    p = Plotly3DBrain(conn, ann)
    out = tmp_path / "conn.html"
    p.write_html(out, activity=np.zeros(conn.num_neurons, np.float32))
    assert out.exists() and out.stat().st_size > 1000
