"""3D connectome visualization (spec §10).

Two renderers on the same data:

* :class:`Plotly3DBrain`  – interactive HTML (zoom / pan / rotate in a browser).
* :class:`Matplotlib3DBrain` – live matplotlib figure that can record an MP4.

Both share the layout helper :func:`compute_3d_layout`, which places sensory
sub-populations on a left plane, interneurons in a central cloud, and motor
sub-populations on a right plane. Purely visual — NOT anatomical coordinates.

When Milestone 2 wires in the real FlyWire subgraph, we'll swap this layout
for the actual (x, y, z) neuron positions from the dataset annotations. The
renderer API stays identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import numpy as np

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.connectome.graph import Connectome


# ---------------------------------------------------------------------- layout
def compute_3d_layout(
    conn: Connectome,
    ann: NeuronAnnotations,
    jitter: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Return ``(N, 3)`` float32 positions.

    If the annotations carry **real anatomical coordinates** (``ann.xyz`` —
    populated by the FlyWire loader from ``coordinates.csv.gz``), we return
    those directly so the dashboard shows the true shape of the fly brain
    (optic lobes, central brain, VNC, etc.). Otherwise we fall back to a
    schematic sensory→motor column layout.

    Layout convention (schematic fallback):
      * x-axis  = information flow (sensory → interneurons → motor)
      * y, z    = spread within a population
    """
    N = conn.num_neurons

    # ---- Real FlyWire anatomy path -------------------------------------------------
    xyz = getattr(ann, "xyz", None)
    if xyz is not None and xyz.shape == (N, 3):
        pos = np.asarray(xyz, dtype=np.float32).copy()
        # Replace missing rows (NaN) with the population centroid, then jitter.
        rng = np.random.default_rng(seed)
        bad = ~np.isfinite(pos).all(axis=1)
        if bad.any():
            good_mean = np.nanmean(pos, axis=0)
            pos[bad] = good_mean + rng.normal(0.0, 1.0, (bad.sum(), 3)).astype(np.float32)
        # FlyWire coords are in nanometres; centre + rescale to a nice viewing box.
        pos -= pos.mean(axis=0, keepdims=True)
        scale = np.max(np.abs(pos)) + 1e-6
        pos = (pos / scale) * 3.0                     # roughly ±3 units
        # FlyWire y-axis points down (anatomical convention). Flip so dorsal is up.
        pos[:, 1] *= -1.0
        return pos.astype(np.float32)

    # ---- Schematic fallback (synthetic connectome) --------------------------------
    rng = np.random.default_rng(seed)
    pos = np.zeros((N, 3), dtype=np.float32)

    sensory_names = ["up", "down", "left", "right", "rot_cw", "rot_ccw"]
    motor_names   = ["thrust", "descent", "roll_l", "roll_r", "yaw_cw", "yaw_ccw"]

    def _place_column(idx: np.ndarray, x_base: float, z_offset: float):
        n = idx.size
        # Arrange along a short vertical line, jittered in y.
        zs = np.linspace(-0.6, 0.6, max(n, 2))[:n]
        ys = rng.normal(0.0, jitter, n)
        pos[idx, 0] = x_base + rng.normal(0.0, jitter * 0.5, n)
        pos[idx, 1] = ys
        pos[idx, 2] = zs + z_offset

    # Sensory on the left plane (x = -2), one column per gesture.
    for i, s in enumerate(sensory_names):
        key = f"sensory_{s}"
        if key in ann.populations:
            _place_column(ann.get(key).neuron_indices,
                          x_base=-2.0, z_offset=(i - 2.5) * 0.25)

    # Interneurons: 3D cloud in the middle.
    if "interneuron" in ann.populations:
        inter = ann.get("interneuron").neuron_indices
        n = inter.size
        pts = rng.normal(0.0, 0.6, (n, 3)).astype(np.float32)
        pts[:, 0] *= 0.7                      # keep them near x=0
        pos[inter] = pts

    # Motor on the right plane (x = +2).
    for i, m in enumerate(motor_names):
        key = f"motor_{m}"
        if key in ann.populations:
            _place_column(ann.get(key).neuron_indices,
                          x_base=+2.0, z_offset=(i - 2.5) * 0.25)

    # Any neurons still at origin (e.g. synthetic connectome) get a random cloud.
    unset = np.where((pos == 0).all(axis=1))[0]
    if unset.size:
        pos[unset] = rng.normal(0.0, 1.0, (unset.size, 3)).astype(np.float32)
    return pos


def _population_colors(ann: NeuronAnnotations, N: int) -> np.ndarray:
    """Return an ``(N, 4)`` RGBA base color per neuron (for inactive nodes)."""
    base = np.tile(np.array([0.35, 0.35, 0.45, 0.7], np.float32), (N, 1))
    palette = {
        "sensory_up":      (0.20, 0.85, 1.00),
        "sensory_down":    (0.20, 0.60, 1.00),
        "sensory_left":    (0.00, 0.90, 0.60),
        "sensory_right":   (0.60, 0.90, 0.20),
        "sensory_rot_cw":  (0.90, 0.90, 0.30),
        "sensory_rot_ccw": (0.90, 0.60, 0.20),
        "interneuron":     (0.55, 0.55, 0.70),
        "motor_thrust":    (1.00, 0.35, 0.35),
        "motor_descent":   (0.90, 0.30, 0.55),
        "motor_roll_l":    (1.00, 0.50, 0.20),
        "motor_roll_r":    (1.00, 0.65, 0.20),
        "motor_yaw_cw":    (1.00, 0.80, 0.30),
        "motor_yaw_ccw":   (0.95, 0.40, 0.70),
    }
    for name, rgb in palette.items():
        if name in ann.populations:
            idx = ann.get(name).neuron_indices
            base[idx, :3] = rgb
            base[idx, 3] = 0.85
    return base


# ---------------------------------------------------------------- edge sampling
def _sampled_edges(
    conn: Connectome, pos: np.ndarray, max_edges: int = 800, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(segments, weights)`` where ``segments`` has shape
    ``(E, 2, 3)`` — pairs of endpoints for a Line3DCollection."""
    coo = conn.W.tocoo()
    if coo.data.size == 0:
        return np.zeros((0, 2, 3), np.float32), np.zeros(0, np.float32)
    rng = np.random.default_rng(seed)
    n_take = min(max_edges, coo.data.size)
    idx = rng.choice(coo.data.size, size=n_take, replace=False)
    pre  = coo.col[idx]
    post = coo.row[idx]
    seg = np.stack([pos[pre], pos[post]], axis=1)          # (E, 2, 3)
    return seg.astype(np.float32), coo.data[idx].astype(np.float32)


# =============================================================== Plotly (HTML)
class Plotly3DBrain:
    """Export an interactive 3D connectome to standalone HTML.

    Usage::

        v = Plotly3DBrain(conn, ann)
        v.write_html("experiments/connectome_3d.html", activity=activity_vec)
    """

    def __init__(self, conn: Connectome, ann: NeuronAnnotations, max_edges: int = 800):
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError as e:
            raise ImportError("plotly not installed. `pip install -e '.[viz]'`") from e
        self.conn = conn
        self.ann = ann
        self.pos = compute_3d_layout(conn, ann)
        self.base_colors = _population_colors(ann, conn.num_neurons)
        self.segments, self.weights = _sampled_edges(conn, self.pos, max_edges=max_edges)

    def _figure(self, activity: Optional[np.ndarray] = None, title: str = ""):
        import plotly.graph_objects as go
        N = self.conn.num_neurons
        act = activity if activity is not None else np.zeros(N, np.float32)
        act = np.clip(act.astype(np.float32), 0.0, 1.0)

        # --- Edges: one Scatter3d with None separators (fast, single trace).
        if self.segments.size:
            E = self.segments.shape[0]
            xe = np.empty(E * 3, np.float32); ye = np.empty(E * 3, np.float32); ze = np.empty(E * 3, np.float32)
            xe[0::3] = self.segments[:, 0, 0]; xe[1::3] = self.segments[:, 1, 0]; xe[2::3] = np.nan
            ye[0::3] = self.segments[:, 0, 1]; ye[1::3] = self.segments[:, 1, 1]; ye[2::3] = np.nan
            ze[0::3] = self.segments[:, 0, 2]; ze[1::3] = self.segments[:, 1, 2]; ze[2::3] = np.nan
            edge_trace = go.Scatter3d(
                x=xe, y=ye, z=ze, mode="lines",
                line=dict(color="rgba(180,180,200,0.12)", width=1),
                hoverinfo="skip", name="synapses",
            )
        else:
            edge_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="synapses")

        # --- Nodes: color = base color when inactive, hot when active.
        # Encode activity into marker size + a red channel boost.
        base = self.base_colors.copy()
        boost = np.clip(act, 0, 1)[:, None] * np.array([1.0, 0.4, 0.0, 0.0])
        rgba = np.clip(base + boost, 0, 1)
        node_colors = [
            f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a:.2f})"
            for r, g, b, a in rgba
        ]
        sizes = 4.0 + 10.0 * act
        # Hover: neuron id + which population it belongs to.
        pop_of = np.array(["-"] * N, dtype=object)
        for name, p in self.ann.populations.items():
            pop_of[p.neuron_indices] = name
        hover = [f"id={i}<br>pop={pop_of[i]}<br>activity={act[i]:.2f}" for i in range(N)]

        node_trace = go.Scatter3d(
            x=self.pos[:, 0], y=self.pos[:, 1], z=self.pos[:, 2],
            mode="markers",
            marker=dict(size=sizes.tolist(), color=node_colors, line=dict(width=0)),
            text=hover, hoverinfo="text", name="neurons",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title or "FlyBrain connectome — 3D",
            paper_bgcolor="#0b0b12", font=dict(color="#e6e6e6"),
            scene=dict(
                bgcolor="#0b0b12",
                xaxis=dict(title="sensory → motor", color="#aaaacc",
                           showbackground=False, gridcolor="#22223a"),
                yaxis=dict(title="", color="#aaaacc",
                           showbackground=False, gridcolor="#22223a"),
                zaxis=dict(title="", color="#aaaacc",
                           showbackground=False, gridcolor="#22223a"),
                camera=dict(eye=dict(x=1.6, y=1.3, z=0.9)),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        return fig

    def show(self, activity: Optional[np.ndarray] = None, title: str = "") -> None:
        self._figure(activity, title).show()

    def write_html(
        self,
        path: str | Path,
        activity: Optional[np.ndarray] = None,
        title: str = "",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._figure(activity, title).write_html(str(path), include_plotlyjs="cdn")
        return path


# ============================================================ Matplotlib (MP4)
class Matplotlib3DBrain:
    """3D matplotlib scatter that can be updated frame-by-frame and recorded
    to an MP4 via ffmpeg (spec §21: "record to evaluate")."""

    def __init__(
        self,
        conn: Connectome,
        ann: NeuronAnnotations,
        record_path: Optional[Path] = None,
        fps: int = 20,
        rotate: bool = True,
        max_edges: int = 400,
        show: bool = True,
    ):
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        else:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.animation import FFMpegWriter

        self.conn = conn; self.ann = ann
        self.pos = compute_3d_layout(conn, ann)
        self.base_colors = _population_colors(ann, conn.num_neurons)
        self.segments, _ = _sampled_edges(conn, self.pos, max_edges=max_edges)
        self.rotate = rotate; self.fps = fps; self.show = show

        self.fig = plt.figure(figsize=(11, 8), facecolor="#0b0b12")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._style_axes()

        # Edges as a Line3DCollection
        self.edge_coll = Line3DCollection(
            self.segments, colors=(0.7, 0.7, 0.85, 0.06), linewidths=0.5,
        )
        self.ax.add_collection3d(self.edge_coll)

        # Nodes as a scatter
        self.scat = self.ax.scatter(
            self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
            c=self.base_colors, s=15, edgecolors="none", depthshade=True,
        )

        self._label_populations()

        # Title / info overlay
        self.title = self.ax.set_title("FlyBrain connectome — 3D",
                                       color="#e6e6e6", fontsize=13)
        self.info = self.fig.text(0.02, 0.02, "", color="#c8c8dc", fontsize=9,
                                  family="monospace")

        self._writer = None
        if record_path is not None:
            record_path = Path(record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._writer = FFMpegWriter(fps=fps, bitrate=2200)
                self._writer.setup(self.fig, str(record_path), dpi=110)
                print(f"[3d-brain] recording to {record_path}")
            except Exception as e:
                print(f"[3d-brain] MP4 recording disabled ({e}); install ffmpeg.")
                self._writer = None

        if show:
            plt.ion(); self.fig.canvas.draw(); plt.show(block=False)

        self._frame = 0

    # ---------------------------------------------------------------- private
    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor("#0b0b12")
        ax.xaxis.set_pane_color((0.05, 0.05, 0.10, 1.0))
        ax.yaxis.set_pane_color((0.05, 0.05, 0.10, 1.0))
        ax.zaxis.set_pane_color((0.05, 0.05, 0.10, 1.0))
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.line.set_color("#33334a"); a.label.set_color("#aaaacc")
        ax.tick_params(colors="#7d7d99", labelsize=8)
        ax.set_xlabel("sensory → motor"); ax.set_ylabel(""); ax.set_zlabel("")
        # Fixed bounds so activity animation stays stable.
        p = self.pos
        pad = 0.3
        ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
        ax.set_ylim(p[:, 1].min() - pad, p[:, 1].max() + pad)
        ax.set_zlim(p[:, 2].min() - pad, p[:, 2].max() + pad)

    def _label_populations(self):
        sensory_names = ["UP", "DOWN", "LEFT", "RIGHT", "ROT_CW", "ROT_CCW"]
        motor_names   = ["THRUST", "DESCENT", "ROLL_L", "ROLL_R", "YAW_CW", "YAW_CCW"]
        keys_s = ["sensory_up", "sensory_down", "sensory_left",
                  "sensory_right", "sensory_rot_cw", "sensory_rot_ccw"]
        keys_m = ["motor_thrust", "motor_descent", "motor_roll_l",
                  "motor_roll_r", "motor_yaw_cw", "motor_yaw_ccw"]
        for label, key in list(zip(sensory_names, keys_s)) + list(zip(motor_names, keys_m)):
            if key not in self.ann.populations: continue
            idx = self.ann.get(key).neuron_indices
            if idx.size == 0: continue
            c = self.pos[idx].mean(axis=0)
            self.ax.text(c[0], c[1], c[2] + 0.35, label,
                         color="#dddde8", fontsize=7, ha="center")

    # ---------------------------------------------------------------- update
    def update(
        self,
        activity: np.ndarray,
        title_suffix: str = "",
        info_lines: Iterable[str] = (),
    ) -> None:
        """Push one animation frame.

        `activity` — shape (N,) in [0, 1] (e.g. an EMA of the spike vector).
        """
        act = np.clip(activity.astype(np.float32), 0.0, 1.0)

        # Color = base RGBA + warm boost proportional to activity
        boost = act[:, None] * np.array([1.0, 0.4, 0.0, 0.0])
        rgba = np.clip(self.base_colors + boost, 0.0, 1.0)
        sizes = 15.0 + 60.0 * act
        self.scat.set_facecolors(rgba)
        self.scat.set_sizes(sizes)

        if self.rotate:
            self.ax.view_init(elev=22.0, azim=(self._frame * 0.8) % 360)

        if title_suffix:
            self.title.set_text(f"FlyBrain connectome — 3D   |   {title_suffix}")
        if info_lines:
            self.info.set_text("\n".join(info_lines))

        self.fig.canvas.draw_idle()
        if self.show:
            self.fig.canvas.flush_events()
        if self._writer is not None:
            self._writer.grab_frame()
        self._frame += 1

    def close(self) -> None:
        if self._writer is not None:
            try: self._writer.finish()
            except Exception: pass
        import matplotlib.pyplot as plt
        plt.close(self.fig)
