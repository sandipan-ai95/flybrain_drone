"""FlyBrain Drone — Real-time Neural Simulation dashboard.

Single-screen research UI inspired by the "FlyBrain Drone" product mockup.
The whole figure is also recorded to MP4 via FFMpegWriter.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Optional
import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.connectome.graph import Connectome
from flybrain.visualization.brain3d import compute_3d_layout, _sampled_edges
from flybrain.visualization.drone3d import Drone3DMesh


BG         = "#0b1220"
PANEL_BG   = "#0f172a"
PANEL_EDGE = "#233149"
TEXT       = "#e6edf7"
MUTED      = "#8ea0bd"
ACCENT     = "#22d3ee"

REGION_COLORS = {
    "region_optic_lobe":       (0.22, 0.75, 1.00),
    "region_central_complex":  (0.75, 0.35, 1.00),
    "region_mushroom_body":    (1.00, 0.85, 0.25),
    "region_antennal_lobe":    (0.35, 0.95, 0.55),
    "region_motor":            (1.00, 0.42, 0.35),
    "region_interneuron":      (0.55, 0.60, 0.75),
}
REGION_LABEL = {
    "region_optic_lobe":       "Optic Lobe\n(Visual Input)",
    "region_central_complex":  "Central Complex\n(Navigation)",
    "region_mushroom_body":    "Mushroom Body\n(Learning / Memory)",
    "region_antennal_lobe":    "Antennal Lobe\n(Olfaction)",
    "region_motor":            "Motor Circuits\n(Output)",
    "region_interneuron":      "Interneurons",
}

SENSORY_MAP = {
    "sensory_up": "region_optic_lobe", "sensory_down": "region_optic_lobe",
    "sensory_left": "region_optic_lobe", "sensory_right": "region_optic_lobe",
    "sensory_rot_cw": "region_optic_lobe", "sensory_rot_ccw": "region_optic_lobe",
    "interneuron": "region_interneuron",
    "motor_thrust": "region_motor", "motor_descent": "region_motor",
    "motor_roll_l": "region_motor", "motor_roll_r": "region_motor",
    "motor_yaw_cw": "region_motor", "motor_yaw_ccw": "region_motor",
}


def _resolve_regions(ann, N):
    regions = {k: [] for k in REGION_COLORS}
    have_real = any(k in ann.populations for k in REGION_COLORS)
    if have_real:
        for k in REGION_COLORS:
            if k in ann.populations:
                regions[k] = ann.get(k).neuron_indices.tolist()
    else:
        for pop_name, region_key in SENSORY_MAP.items():
            if pop_name in ann.populations:
                regions[region_key].extend(ann.get(pop_name).neuron_indices.tolist())
    assigned = np.zeros(N, dtype=bool)
    for idx in regions.values():
        if idx:
            assigned[idx] = True
    if not assigned.all():
        regions["region_interneuron"] = list(
            set(regions["region_interneuron"]) | set(np.where(~assigned)[0].tolist())
        )
    return {k: np.asarray(v, dtype=np.int32) for k, v in regions.items() if len(v)}


def _downsample(arr, out):
    h, w = arr.shape
    if h < out or w < out:
        return arr
    yy = np.linspace(0, h - 1, out).astype(np.int32)
    xx = np.linspace(0, w - 1, out).astype(np.int32)
    return arr[np.ix_(yy, xx)]


class UnifiedDashboard:
    """Real-time neural-simulation dashboard styled after the mockup."""

    def __init__(
        self,
        conn,
        ann,
        record_path=None,
        fps=15,
        show=True,
        raster_window=200,
        rotate_brain=True,
        max_edges=800,
    ):
        self.conn = conn
        self.ann = ann
        self.N = conn.num_neurons
        self.show = show
        self.rotate_brain = rotate_brain
        self.fps = fps

        self.pos = compute_3d_layout(conn, ann)
        self.regions = _resolve_regions(ann, self.N)
        self.base_colors = self._make_base_colors()
        segments, _ = _sampled_edges(conn, self.pos, max_edges=max_edges)
        self._segments = segments

        coo = conn.W.tocoo()
        if coo.data.size > 0 and segments.shape[0] > 0:
            rng = np.random.default_rng(0)
            k = min(max_edges, coo.data.size)
            idx = rng.choice(coo.data.size, size=k, replace=False)
            self._edge_pre = coo.col[idx].astype(np.int32)
        else:
            self._edge_pre = np.zeros(0, np.int32)
        self._edge_glow = np.zeros(self._edge_pre.size, dtype=np.float32)

        self._activity_hist = {k: deque([0.0] * 200, maxlen=200) for k in self.regions}
        self._time_hist = deque(maxlen=200)
        self._traj = []
        self._frame = 0
        self._t0 = None
        self._total_spikes = 0

        self.fig = plt.figure(figsize=(17.5, 10), facecolor=BG)
        gs = self.fig.add_gridspec(
            4, 3,
            width_ratios=[1.0, 2.0, 1.35],
            height_ratios=[0.35, 1.0, 1.0, 0.85],
            left=0.025, right=0.985, top=0.965, bottom=0.035,
            hspace=0.35, wspace=0.16,
        )
        self.ax_header    = self.fig.add_subplot(gs[0, :])
        self.ax_cam       = self.fig.add_subplot(gs[1, 0])
        self.ax_encoding  = self.fig.add_subplot(gs[2, 0])
        self.ax_retinal   = self.fig.add_subplot(gs[3, 0])
        self.ax_brain     = self.fig.add_subplot(gs[1:3, 1], projection="3d")
        self.ax_brain_bar = self.fig.add_subplot(gs[3, 1])
        self.ax_drone     = self.fig.add_subplot(gs[1, 2], projection="3d")
        self.ax_neural    = self.fig.add_subplot(gs[2, 2])
        self.ax_motor     = self.fig.add_subplot(gs[3, 2])

        self._init_header()
        self._init_camera_panel()
        self._init_encoding_panel()
        self._init_retinal_panel()
        self._init_brain_panel()
        self._init_brain_bar()
        self._init_drone_panel()
        self._init_neural_panel()
        self._init_motor_panel()

        self.writer = None
        if record_path is not None:
            record_path = Path(record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.writer = FFMpegWriter(fps=fps, bitrate=2600)
                self.writer.setup(self.fig, str(record_path), dpi=110)
                print(f"[dashboard] recording to {record_path}")
            except Exception as e:
                print(f"[dashboard] MP4 disabled ({e}); install ffmpeg.")

        if show:
            plt.ion()
            self.fig.canvas.draw()
            plt.show(block=False)

    def _make_base_colors(self):
        base = np.tile(np.array([0.30, 0.35, 0.50, 0.55], np.float32),
                       (self.N, 1))
        for key, idx in self.regions.items():
            rgb = REGION_COLORS.get(key, (0.6, 0.6, 0.8))
            base[idx, :3] = rgb
            base[idx, 3] = 0.90
        return base

    def _style_axes(self, ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(PANEL_BG)
        for s in ax.spines.values():
            s.set_color(PANEL_EDGE)
        ax.tick_params(colors=MUTED, labelsize=7)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=6)
        if xlabel:
            ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=MUTED, fontsize=8)

    def _init_header(self):
        ax = self.ax_header
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.005, 0.72, "FlyBrain Drone — Real-time Neural Simulation",
                color=TEXT, fontsize=15, fontweight="bold", va="center")
        ax.text(0.005, 0.28,
                "From Vision to Action • Inspired by Drosophila Connectome (FAFB v783)",
                color=MUTED, fontsize=9, va="center")
        ax.text(0.995, 0.72, "● SIMULATION RUNNING",
                color="#22c55e", fontsize=10, ha="right", va="center",
                fontweight="bold")
        steps = [
            ("1. Camera Input",    "Live / Simulated"),
            ("2. Visual Encoding", "Retinal-like features"),
            ("3. Fly Brain Sim.",  "Neural propagation"),
            ("4. Motor Decoder",   "Neural → Control"),
            ("5. Drone Control",   "3D flight"),
        ]
        n = len(steps); pad = 0.005
        w = (1.0 - 2 * pad) / n - 0.008
        y = 0.05; h = 0.18
        for i, (title, sub) in enumerate(steps):
            x = pad + i * ((1.0 - 2 * pad) / n)
            box = FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.005,rounding_size=0.02",
                                 linewidth=1.0,
                                 edgecolor=PANEL_EDGE, facecolor=PANEL_BG)
            ax.add_patch(box)
            ax.text(x + 0.012, y + h * 0.60, title, color=TEXT,
                    fontsize=8.5, fontweight="bold", va="center")
            ax.text(x + 0.012, y + h * 0.25, sub, color=MUTED,
                    fontsize=7, va="center")
            if i < n - 1:
                ax.annotate("", xy=(x + w + 0.004, y + h / 2),
                            xytext=(x + w - 0.001, y + h / 2),
                            arrowprops=dict(arrowstyle="->",
                                            color=ACCENT, lw=1.4))

    def _init_camera_panel(self):
        ax = self.ax_cam
        self._style_axes(ax, "Camera Input (Live View)")
        ax.set_xticks([]); ax.set_yticks([])
        self._cam_im = ax.imshow(np.zeros((10, 10, 3), np.uint8))
        self._cam_txt = ax.text(
            0.02, 0.05, "", transform=ax.transAxes,
            fontsize=9, color="#7cff9e", va="bottom", family="monospace",
            bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none"),
        )
        self._cam_gest = ax.text(
            0.98, 0.97, "", transform=ax.transAxes,
            fontsize=11, color=TEXT, va="top", ha="right", fontweight="bold",
            bbox=dict(facecolor=(0.13, 0.83, 0.54, 0.35),
                      pad=5, edgecolor="none",
                      boxstyle="round,pad=0.35"),
        )

    def _init_encoding_panel(self):
        ax = self.ax_encoding
        self._style_axes(ax, "Visual Encoding")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 3); ax.set_ylim(0, 1.15)
        self._enc_ims = []
        for i, label in enumerate(["Luminance", "Edges", "Motion / Optic Flow"]):
            sub = np.zeros((32, 32), dtype=np.float32)
            im = ax.imshow(sub, extent=(i + 0.05, i + 0.95, 0.15, 1.10),
                           cmap="gray" if i < 2 else "twilight",
                           vmin=0, vmax=1, aspect="auto")
            self._enc_ims.append(im)
            ax.text(i + 0.5, 0.06, label, color=MUTED, fontsize=8,
                    ha="center", va="center")

    def _init_retinal_panel(self):
        ax = self.ax_retinal
        self._style_axes(ax, "Neural Input (Retinal Map)")
        ax.set_xticks([]); ax.set_yticks([])
        self._retinal_im = ax.imshow(np.zeros((16, 16), np.float32),
                                     cmap="viridis", vmin=0, vmax=1,
                                     aspect="equal")

    def _init_brain_panel(self):
        ax = self.ax_brain
        self._style_axes(ax, "Fly Brain — 3D Neural Activity (FAFB v783)")
        ax.set_facecolor(PANEL_BG)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.06, 0.09, 0.16, 1.0))
            a.line.set_color(PANEL_EDGE)
        ax.tick_params(colors=MUTED, labelsize=6)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        pad = 0.3
        ax.set_xlim(self.pos[:, 0].min() - pad, self.pos[:, 0].max() + pad)
        ax.set_ylim(self.pos[:, 1].min() - pad, self.pos[:, 1].max() + pad)
        ax.set_zlim(self.pos[:, 2].min() - pad, self.pos[:, 2].max() + pad)
        ax.view_init(elev=18.0, azim=-70.0)
        self._edge_coll = Line3DCollection(
            self._segments, colors=(0.7, 0.75, 0.95, 0.04), linewidths=0.35)
        ax.add_collection3d(self._edge_coll)
        self._brain_scat = ax.scatter(
            self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
            c=self.base_colors, s=4, edgecolors="none", depthshade=True)
        for key, idx in self.regions.items():
            if idx.size < 20:
                continue
            label = REGION_LABEL.get(key, key)
            c = self.pos[idx].mean(axis=0)
            rgb = REGION_COLORS.get(key, (0.9, 0.9, 0.9))
            ax.text(c[0], c[1], c[2] + 0.35, label,
                    color=(rgb[0], rgb[1], rgb[2], 1.0),
                    fontsize=7.5, ha="center", va="bottom",
                    fontweight="bold")

    def _init_brain_bar(self):
        ax = self.ax_brain_bar
        ax.set_facecolor(PANEL_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(PANEL_EDGE)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        chips = [k for k in REGION_COLORS if k in self.regions]
        chip_w = 0.98 / max(len(chips), 1)
        for i, key in enumerate(chips):
            x = 0.01 + i * chip_w
            rgb = REGION_COLORS[key]
            ax.add_patch(Rectangle((x + 0.005, 0.55), 0.02, 0.3,
                                    facecolor=rgb, edgecolor="none"))
            ax.text(x + 0.032, 0.70,
                    key.replace("region_", "").replace("_", " ").title(),
                    color=TEXT, fontsize=8, va="center")
        self._brain_bar_txt = ax.text(
            0.01, 0.20, "", color=MUTED, fontsize=8.5,
            family="monospace", va="center")

    def _init_drone_panel(self):
        ax = self.ax_drone
        self._style_axes(ax, "Drone View (3D Environment)")
        ax.set_facecolor(PANEL_BG)
        ax.xaxis.set_pane_color((0.10, 0.14, 0.28, 1.0))
        ax.yaxis.set_pane_color((0.10, 0.14, 0.28, 1.0))
        ax.zaxis.set_pane_color((0.06, 0.08, 0.16, 1.0))
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.line.set_color(PANEL_EDGE)
        ax.tick_params(colors=MUTED, labelsize=6)
        self._world_half = 8.0
        ax.set_xlim(-self._world_half, self._world_half)
        ax.set_ylim(-self._world_half, self._world_half)
        ax.set_zlim(0, 10)
        grid_lines = []
        R = 20.0
        for g in np.arange(-R, R + 2.0, 2.0):
            grid_lines.append([(-R, g, 0), (R, g, 0)])
            grid_lines.append([(g, -R, 0), (g, R, 0)])
        ax.add_collection3d(Line3DCollection(
            grid_lines, colors=(0.32, 0.38, 0.55, 0.35), linewidths=0.5))
        theta = np.linspace(0, 2 * np.pi, 60)
        ax.plot(R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta),
                color=(0.4, 0.6, 1.0, 0.4), lw=0.7)
        pad_t = np.linspace(0, 2 * np.pi, 40)
        ax.plot(0.8 * np.cos(pad_t), 0.8 * np.sin(pad_t),
                np.zeros_like(pad_t), color="#ffcc44", lw=1.2)
        self._traj_line, = ax.plot([], [], [], "-", color=ACCENT,
                                    lw=1.4, alpha=0.85)
        self._drone_mesh = Drone3DMesh(ax)
        self._drone_txt = ax.text2D(
            0.02, 0.97, "", transform=ax.transAxes,
            color=TEXT, fontsize=8, family="monospace", va="top",
            bbox=dict(facecolor=(0, 0, 0, 0.5), pad=4, edgecolor="none",
                      boxstyle="round,pad=0.3"),
        )
        ax.view_init(elev=18.0, azim=-60.0)

    def _init_neural_panel(self):
        ax = self.ax_neural
        self._style_axes(ax, "Neural Activity (Real-time)",
                         xlabel="Time (s)", ylabel="Firing rate (Hz)")
        ax.set_xlim(-20, 0); ax.set_ylim(0, 100)
        self._neural_lines = {}
        for key in self.regions:
            rgb = REGION_COLORS.get(key, (0.7, 0.7, 0.7))
            line, = ax.plot([], [], "-", lw=1.4, color=rgb,
                            label=REGION_LABEL.get(key, key).split("\n")[0])
            self._neural_lines[key] = line
        leg = ax.legend(loc="upper right", fontsize=6.5, framealpha=0.35,
                        facecolor=PANEL_BG, edgecolor=PANEL_EDGE)
        for t in leg.get_texts():
            t.set_color(TEXT)

    def _init_motor_panel(self):
        ax = self.ax_motor
        self._style_axes(ax, "Motor Output (from Neural Activity)")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Yaw", "Pitch", "Roll", "Thrust"],
                           color=TEXT, fontsize=8)
        ax.axvline(0, color=PANEL_EDGE, lw=0.6)
        self._motor_bars = ax.barh(
            [3, 2, 1, 0], [0, 0, 0, 0],
            color=[(0.13, 0.83, 0.94), (0.35, 0.95, 0.55),
                   (1.00, 0.65, 0.20), (0.75, 0.35, 1.00)],
            edgecolor="none",
        )
        self._motor_txt = [
            ax.text(1.45, i, "0.00", color=TEXT, fontsize=8, family="monospace",
                    ha="right", va="center")
            for i in [3, 2, 1, 0]
        ]
        self._stats_txt = ax.text(
            1.55, 3.4, "", transform=ax.transData,
            color=TEXT, fontsize=8, family="monospace", va="top",
            bbox=dict(facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                      boxstyle="round,pad=0.4"),
        )

    def update(self, *, webcam_rgb, gesture, confidence, spikes, activity_ema,
               drone_pos, cmd, t, drone_orientation=(0.0, 0.0, 0.0),
               info_lines=()):
        if self._t0 is None:
            self._t0 = t

        if webcam_rgb is not None:
            self._cam_im.set_data(webcam_rgb)
            self._cam_im.set_extent((0, webcam_rgb.shape[1], webcam_rgb.shape[0], 0))
            self.ax_cam.set_xlim(0, webcam_rgb.shape[1])
            self.ax_cam.set_ylim(webcam_rgb.shape[0], 0)
            self._update_encodings(webcam_rgb)
        else:
            self._cam_im.set_data(np.zeros((10, 10, 3), np.uint8))
        self._cam_txt.set_text(f"t = {t:6.2f}s")
        self._cam_gest.set_text(f"{gesture.upper()}  {confidence*100:3.0f}%")

        act = np.clip(activity_ema.astype(np.float32), 0.0, 1.0)
        self._update_retinal(act)

        act_boost = np.clip(act * 4.0, 0.0, 1.0)
        boost = act_boost[:, None] * np.array([1.0, 0.55, 0.0, 0.0])
        rgba = np.clip(self.base_colors + boost, 0.0, 1.0)
        rgba[:, 3] = np.clip(0.30 + 0.70 * act_boost, 0.0, 1.0)
        self._brain_scat.set_facecolors(rgba)
        self._brain_scat.set_sizes(3.0 + 90.0 * act_boost)

        if self._edge_pre.size:
            self._edge_glow *= 0.75
            fired = (spikes[self._edge_pre] > 0)
            self._edge_glow[fired] = 1.0
            g = self._edge_glow[:, None]
            edge_rgba = np.concatenate([
                0.70 + 0.30 * g, 0.70 + 0.20 * g,
                0.85 - 0.50 * g, 0.05 + 0.75 * g,
            ], axis=1)
            self._edge_coll.set_color(edge_rgba)
            self._edge_coll.set_linewidths(0.4 + 1.8 * g.ravel())
        if self.rotate_brain:
            self.ax_brain.view_init(
                elev=18.0,
                azim=(-70.0 + 15.0 * np.sin(self._frame * 0.02)),
            )

        n_active = int((activity_ema > 0.05).sum())
        step_spikes = int((spikes > 0).sum())
        self._total_spikes += step_spikes
        self._time_hist.append(t - self._t0)
        for key, line in self._neural_lines.items():
            idx = self.regions[key]
            rate = float(activity_ema[idx].mean()) * 1000.0 if idx.size else 0.0
            self._activity_hist[key].append(rate)
            xs = np.array(self._time_hist)
            ys = np.array(self._activity_hist[key])[-len(xs):]
            line.set_data(xs - xs[-1], ys)
        y_all = np.concatenate([np.array(v) for v in self._activity_hist.values()])
        y_top = float(np.clip(y_all.max() * 1.25, 20.0, 500.0))
        self.ax_neural.set_ylim(0, y_top)

        for bar, val in zip(self._motor_bars, [cmd[0], cmd[1], cmd[2], cmd[3]]):
            bar.set_width(float(np.clip(val, -1.5, 1.5)))
        for txt, val in zip(self._motor_txt, [cmd[0], cmd[1], cmd[2], cmd[3]]):
            txt.set_text(f"{val:+.2f}")

        elapsed = t - self._t0
        active_regions = ", ".join(
            REGION_LABEL[k].split("\n")[0]
            for k in self.regions
            if float(activity_ema[self.regions[k]].mean() * 1000) > 5.0
        ) or "—"
        self._stats_txt.set_text(
            f"Sim time     : {elapsed:6.2f} s\n"
            f"Neural dt    :   1 ms\n"
            f"Active neurons: {n_active:5d}\n"
            f"Total spikes : {self._total_spikes:7d}\n"
            f"Gesture      : {gesture.upper()}"
        )
        self._brain_bar_txt.set_text(
            f"N={self.N:,} neurons   |   edges={self.conn.W.nnz:,}   |   "
            f"active regions: {active_regions}"
        )

        self._traj.append(tuple(drone_pos))
        if len(self._traj) > 1500:
            self._traj = self._traj[-1500:]
        xs, ys, zs = zip(*self._traj)
        self._traj_line.set_data(xs, ys)
        self._traj_line.set_3d_properties(zs)
        roll, pitch, yaw = drone_orientation
        self._drone_mesh.update(drone_pos[0], drone_pos[1], drone_pos[2],
                                 roll, pitch, yaw, thrust=cmd[0])
        cx, cy, cz = drone_pos
        h = self._world_half
        self.ax_drone.set_xlim(cx - h, cx + h)
        self.ax_drone.set_ylim(cy - h, cy + h)
        z_lo = max(0.0, cz - 3.0)
        self.ax_drone.set_zlim(z_lo, z_lo + 10.0)
        cam_azim = (-60.0 + np.degrees(yaw) * 0.5 + self._frame * 0.15) % 360
        self.ax_drone.view_init(elev=20.0, azim=cam_azim)
        speed = 0.0
        if len(self._traj) >= 2:
            dp = np.array(self._traj[-1]) - np.array(self._traj[-2])
            speed = float(np.linalg.norm(dp) * self.fps)
        self._drone_txt.set_text(
            f"Position : ({drone_pos[0]:+.2f}, {drone_pos[1]:+.2f}, "
            f"{drone_pos[2]:.2f}) m\n"
            f"Speed    : {speed:.2f} m/s\n"
            f"Orient   : R={np.degrees(roll):+.1f}  "
            f"P={np.degrees(pitch):+.1f}  Y={np.degrees(yaw):+.1f}\n"
            f"Gesture  : {gesture.upper()}"
        )

        self.fig.canvas.draw_idle()
        if self.show:
            self.fig.canvas.flush_events()
        if self.writer is not None:
            self.writer.grab_frame()
        self._frame += 1

    def _update_encodings(self, rgb):
        img = rgb.astype(np.float32) / 255.0
        lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] +
               0.114 * img[:, :, 2])
        lum_s = _downsample(lum, 32)
        self._enc_ims[0].set_data(lum_s)
        self._enc_ims[0].set_clim(0, 1)
        gy, gx = np.gradient(lum)
        edges = np.hypot(gx, gy)
        e_s = _downsample(edges, 32)
        if e_s.max() > 0:
            e_s = e_s / e_s.max()
        self._enc_ims[1].set_data(e_s)
        prev = getattr(self, "_prev_lum", lum)
        flow = np.abs(lum - prev)
        f_s = _downsample(flow, 32)
        if f_s.max() > 0:
            f_s = f_s / f_s.max()
        self._enc_ims[2].set_data(f_s)
        self._prev_lum = lum

    def _update_retinal(self, activity):
        idx = self.regions.get("region_optic_lobe", np.array([], np.int32))
        if idx.size < 16:
            idx = np.arange(min(256, self.N))
        vals = activity[idx][:256]
        if vals.size < 256:
            vals = np.pad(vals, (0, 256 - vals.size))
        heat = vals.reshape(16, 16)
        self._retinal_im.set_data(heat)
        self._retinal_im.set_clim(0, max(1e-3, float(heat.max())))

    def close(self):
        if self.writer is not None:
            try:
                self.writer.finish()
            except Exception:
                pass
        plt.close(self.fig)
