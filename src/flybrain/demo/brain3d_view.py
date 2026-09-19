"""3D connectome viewer demo.

What this shows
---------------
* **Interactive HTML**: open in any browser → drag to rotate, scroll to zoom,
  hover a neuron for its ID + population + activity.
* **MP4 recording** (optional): a live matplotlib 3D view with a slowly
  rotating camera. A pulse of activity is injected into each sensory
  sub-population in turn (UP → DOWN → LEFT → RIGHT → ROT_CW → ROT_CCW), so
  you can literally *see* signals flow from sensory columns, through the
  interneuron cloud, into the matching motor column.

Run
---
    # 1. Static interactive HTML with a snapshot of activity (fast; no video)
    python -m flybrain.demo.brain3d_view --html-only

    # 2. Record an animated MP4 driving each gesture population in sequence
    python -m flybrain.demo.brain3d_view --record --steps 400 --no-show

    # 3. Live window + record simultaneously
    python -m flybrain.demo.brain3d_view --record --steps 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flybrain.connectome.loader import load_gesture
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator
from flybrain.vision.gesture import Gesture
from flybrain.vision.stimulus import GestureStimulus
from flybrain.visualization.brain3d import Plotly3DBrain, Matplotlib3DBrain


GESTURE_SEQUENCE = [
    Gesture.UP, Gesture.DOWN, Gesture.LEFT,
    Gesture.RIGHT, Gesture.ROTATE_CW, Gesture.ROTATE_CCW,
]


def run(
    out_dir: Path = Path("experiments/brain3d"),
    steps: int = 400,
    fps: int = 20,
    html_only: bool = False,
    record: bool = False,
    show: bool = True,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -- Build brain
    conn, ann = load_gesture(seed=seed)
    print(f"[brain3d] connectome: N={conn.num_neurons}  edges={conn.W.nnz}")

    # -- 1. Interactive HTML: run the brain briefly under an "UP" stimulus so
    # the snapshot shows real activity flowing sensory→motor.
    sim = NeuralSimulator(conn, LIFModel(conn.num_neurons), dt_ms=1.0)
    stim = GestureStimulus(ann, drive_current=20.0)
    ema = np.zeros(conn.num_neurons, dtype=np.float32)
    cur = stim.encode(Gesture.UP)
    for _ in range(120):
        s = sim.step(cur)
        ema = 0.9 * ema + 0.1 * s

    html_path = out_dir / "connectome_3d.html"
    try:
        viewer = Plotly3DBrain(conn, ann)
        viewer.write_html(html_path, activity=ema,
                          title="FlyBrain 3D — activity snapshot under 'UP' gesture")
        print(f"[brain3d] wrote interactive viewer: {html_path}")
        print(f"[brain3d] open with:  open '{html_path}'")
    except ImportError as e:
        print(f"[brain3d] Plotly not available: {e}")

    if html_only:
        return {"html": str(html_path)}

    # -- 2. Animated MP4 that walks through the gesture sequence.
    if not record and not show:
        return {"html": str(html_path)}

    sim.reset()
    ema[:] = 0.0
    mp4_path = out_dir / "connectome_3d.mp4" if record else None
    view = Matplotlib3DBrain(conn, ann, record_path=mp4_path, fps=fps,
                             rotate=True, show=show)

    steps_per_gesture = max(1, steps // len(GESTURE_SEQUENCE))
    try:
        for gi, g in enumerate(GESTURE_SEQUENCE):
            cur = stim.encode(g)
            for k in range(steps_per_gesture):
                s = sim.step(cur)
                ema = 0.9 * ema + 0.1 * s
                # Population activity summary for the on-screen overlay.
                active = int((ema > 0.05).sum())
                spikes = int(s.sum())
                view.update(
                    activity=ema,
                    title_suffix=f"gesture = {g.value.upper()}",
                    info_lines=[
                        f"gesture         : {g.value.upper()}",
                        f"neural step     : {sim.state.step_idx:5d}",
                        f"active neurons  : {active:4d}",
                        f"spikes this step: {spikes:4d}",
                        f"total spikes    : {sim.state.total_spikes:,}",
                    ],
                )
    finally:
        view.close()

    if mp4_path is not None:
        print(f"[brain3d] wrote animation: {mp4_path}")
    return {"html": str(html_path), "mp4": str(mp4_path) if mp4_path else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("experiments/brain3d"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--html-only", action="store_true",
                   help="only produce the interactive HTML file, skip animation")
    p.add_argument("--record", action="store_true", help="write connectome_3d.mp4")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="don't open a live window (headless)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(out_dir=a.out, steps=a.steps, fps=a.fps,
        html_only=a.html_only, record=a.record, show=a.show, seed=a.seed)


if __name__ == "__main__":
    main()
