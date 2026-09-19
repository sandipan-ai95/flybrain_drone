"""FlyBrain Drone — web dashboard server.

Runs the neural + drone simulation on the server, streams state to a browser
over WebSocket, and lets the browser handle:
  * webcam capture + MediaPipe Hands gesture detection (client-side, so it
    works with whichever camera the user chose from the browser prompt).
  * Three.js rendering of the fly-brain point cloud, drone, and world.

Usage::

    python -m flybrain.server.dashboard_server --connectome flywire

Then open http://localhost:8000/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from flybrain.connectome.loader import load_gesture
from flybrain.decoding.motor_decoder import PopulationMotorDecoder, MotorDecoderConfig
from flybrain.drone.physics import DroneCommand, DroneState
from flybrain.drone.simulator import SimplePhysicsSim
from flybrain.drone.quadrotor import QuadrotorPhysicsSim, QuadrotorParams
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator
from flybrain.vision.gesture import Gesture
from flybrain.vision.stimulus import GestureStimulus
from flybrain.visualization.unified_dashboard import (
    REGION_COLORS, REGION_LABEL, _resolve_regions,
)
from flybrain.visualization.brain3d import compute_3d_layout


STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------- decoder --
class _SensoryDiffDecoder(PopulationMotorDecoder):
    """Read the six T4/T5 sensory populations, output balanced drone cmds."""

    def decode(self, spikes: np.ndarray, dt_ms: float) -> DroneCommand:
        alpha = dt_ms / self.cfg.tau_ms
        for key, pop in self.cfg.populations.items():
            idx = (self.ann.get(pop).neuron_indices
                   if pop in self.ann.populations else np.array([], np.int32))
            inst = float(spikes[idx].mean()) if idx.size else 0.0
            self.rates[key] += alpha * (inst - self.rates[key])
        r = self.rates
        return DroneCommand(
            thrust=self.cfg.thrust_bias + self.cfg.thrust_gain * (r["up"] - r["down"]),
            roll=self.cfg.roll_gain * (r["right"] - r["left"]),
            pitch=0.0,
            yaw=self.cfg.yaw_gain * (r["cw"] - r["ccw"]),
        )


def _make_decoder(ann):
    # NOTE: tuned for low-latency response so the drone tracks the gesture
    # within ~150 ms (was ~2 s with tau_ms=120). Gains bumped because sensory
    # populations are large → mean spike counts stay small.
    cfg = MotorDecoderConfig(
        thrust_gain=18.0, roll_gain=10.0, yaw_gain=10.0, pitch_gain=0.0,
        thrust_bias=0.55, tau_ms=20.0,
        populations={"up": "sensory_up", "down": "sensory_down",
                     "left": "sensory_left", "right": "sensory_right",
                     "cw": "sensory_rot_cw", "ccw": "sensory_rot_ccw"},
    )
    return _SensoryDiffDecoder(ann, cfg)


# ------------------------------------------------------------ world state --
class SimulationWorld:
    """Holds all simulation objects; ``step()`` advances by one frame."""

    GESTURE_LOOKUP = {
        "up": Gesture.UP, "down": Gesture.DOWN,
        "left": Gesture.LEFT, "right": Gesture.RIGHT,
        "rotate_cw": Gesture.ROTATE_CW, "rotate_ccw": Gesture.ROTATE_CCW,
        "none": Gesture.NONE, "": Gesture.NONE,
    }

    def __init__(self, connectome_source: str = "flywire", max_neurons: int = 12000,
                 physics: str = "quadrotor"):
        # --- connectome
        if connectome_source == "flywire":
            from flybrain.connectome.flywire_loader import load_flywire
            conn, ann = load_flywire(max_neurons=max_neurons)
        else:
            conn, ann = load_gesture(seed=42)
        self.conn = conn; self.ann = ann
        self.N = conn.num_neurons

        # --- neural + drone + decoder + stimulus
        self.sim = NeuralSimulator(conn, LIFModel(conn.num_neurons), dt_ms=1.0)
        self.stim = GestureStimulus(ann, drive_current=20.0)
        self.decoder = _make_decoder(ann)
        if physics == "simple":
            self.drone = SimplePhysicsSim(DroneState(z=1.5))
        else:
            self.drone = QuadrotorPhysicsSim(DroneState(z=1.5), QuadrotorParams())
        self.physics_kind = physics

        # --- geometry for the client (3D positions + region tags)
        # Prefer REAL FlyWire soma coordinates (nm) when the annotations carry
        # them; fall back to compute_3d_layout otherwise.
        xyz = getattr(ann, "xyz", None)
        if xyz is not None and xyz.shape == (self.N, 3) and np.any(xyz):
            self.pos = np.asarray(xyz, dtype=np.float32)
            self._pos_source = "flywire_real"
        else:
            self.pos = compute_3d_layout(conn, ann)
            self._pos_source = "layout"
        self.regions = _resolve_regions(ann, self.N)
        # per-neuron region index (int8) so we don't send strings
        self.region_names = list(self.regions.keys())
        self.region_of = np.full(self.N, -1, dtype=np.int16)
        for i, name in enumerate(self.region_names):
            self.region_of[self.regions[name]] = i

        # --- running state
        self._activity_ema = np.zeros(self.N, dtype=np.float32)
        self._ema_alpha = 0.45   # snappier — tracks gesture within ~2 frames
        self._total_spikes = 0
        self._current_gesture = Gesture.NONE
        self._current_conf = 0.0

    def set_gesture(self, name: str, confidence: float) -> None:
        name = (name or "").lower()
        self._current_gesture = self.GESTURE_LOOKUP.get(name, Gesture.NONE)
        self._current_conf = float(np.clip(confidence, 0.0, 1.0))

    def brain_payload(self, subsample: int = 15000) -> dict:
        """One-time payload of neuron positions + region tags for the client."""
        N = self.N
        if N > subsample:
            # Keep all neurons in named regions + top-up random from interneurons
            keep = np.concatenate([idx for idx in self.regions.values()])
            keep = np.unique(keep)
            if keep.size > subsample:
                rng = np.random.default_rng(0)
                keep = rng.choice(keep, size=subsample, replace=False)
            elif keep.size < subsample:
                extra = np.setdiff1d(np.arange(N), keep)
                rng = np.random.default_rng(0)
                extra = rng.choice(extra, size=subsample - keep.size, replace=False)
                keep = np.concatenate([keep, extra])
        else:
            keep = np.arange(N)
        pos = self.pos[keep].astype(np.float32)
        region = self.region_of[keep].astype(np.int16)
        # Reframe for viewing: centre, normalise, and flip Y (FlyWire image
        # coords have Y growing downward → invert so dorsal is up in three.js).
        pos = pos - pos.mean(axis=0, keepdims=True)
        if getattr(self, "_pos_source", "") == "flywire_real":
            pos[:, 1] *= -1.0   # dorsal-up
        scale = float(np.percentile(np.linalg.norm(pos, axis=1), 95) + 1e-6)
        pos = pos / scale * 2.4
        return {
            "num_total": int(N),
            "num_sent": int(keep.size),
            "positions": pos.flatten().tolist(),
            "pos_source": getattr(self, "_pos_source", "layout"),
            "region": region.tolist(),
            "regions": [
                {
                    "key": k,
                    "label": REGION_LABEL.get(k, k).replace("\n", " "),
                    "color": REGION_COLORS.get(k, (0.7, 0.7, 0.7)),
                    "count": int(self.regions[k].size),
                }
                for k in self.region_names
            ],
            "keep": keep.tolist(),   # so client can address neurons in the state stream
            "pathways": self._real_pathways(keep),
            "hulls": self._region_hulls(pos, region),
        }

    def _region_hulls(self, pos: np.ndarray, region: np.ndarray) -> list[dict]:
        """Compute a convex hull of the (already-transformed) FlyWire soma
        coordinates per region.  These hulls trace the actual anatomical
        outline of each neuropil's cell-body layer so the client can render
        the recognizable fly-brain silhouette (two optic lobes flanking the
        central mass, mushroom-body caps, etc.).
        """
        try:
            from scipy.spatial import ConvexHull
        except Exception as e:
            print(f"[hulls] scipy unavailable ({e}); skipping")
            return []
        out = []
        for i, key in enumerate(self.region_names):
            mask = (region == i)
            pts = pos[mask]
            if pts.shape[0] < 20:
                continue
            try:
                hull = ConvexHull(pts.astype(np.float64))
            except Exception:
                continue
            verts = pts[hull.vertices]                     # (V, 3)
            # Re-map faces from full-pts indices → local vertex indices
            reindex = -np.ones(pts.shape[0], dtype=np.int32)
            reindex[hull.vertices] = np.arange(hull.vertices.size, dtype=np.int32)
            faces = reindex[hull.simplices]                 # (F, 3)
            out.append({
                "key": key,
                "color": REGION_COLORS.get(key, (0.7, 0.7, 0.7)),
                "vertices": verts.astype(np.float32).flatten().tolist(),
                "faces": faces.astype(np.int32).flatten().tolist(),
            })
        return out

    def _real_pathways(self, keep: np.ndarray,
                       max_edges_per_pair: int = 40) -> list[dict]:
        """Sample REAL synaptic edges of the FlyWire connectome for each
        anatomically-plausible inter-region pathway.  The client uses these
        as the geometry along which activity pulses travel — no schematic
        curves are involved.
        """
        # Anatomically-motivated directed pathway pairs
        pair_defs = [
            ("region_optic_lobe",      "region_central_complex"),
            ("region_optic_lobe",      "region_mushroom_body"),
            ("region_antennal_lobe",   "region_mushroom_body"),
            ("region_mushroom_body",   "region_central_complex"),
            ("region_central_complex", "region_motor"),
            ("region_mushroom_body",   "region_motor"),
            ("region_interneuron",     "region_motor"),
        ]
        # Reverse-map: full-neuron index → position in the client's `keep` list
        keep_inv = -np.ones(self.N, dtype=np.int32)
        keep_inv[keep] = np.arange(keep.size, dtype=np.int32)

        coo = self.conn.W.tocoo()
        pre_all  = coo.row.astype(np.int64)
        post_all = coo.col.astype(np.int64)
        region_of = self.region_of  # int16 array, -1 if unassigned

        # Build region index for edge endpoints once
        pre_reg  = region_of[pre_all]
        post_reg = region_of[post_all]
        name_to_idx = {n: i for i, n in enumerate(self.region_names)}

        rng = np.random.default_rng(1)
        out = []
        for src_name, dst_name in pair_defs:
            if src_name not in name_to_idx or dst_name not in name_to_idx:
                continue
            si, di = name_to_idx[src_name], name_to_idx[dst_name]
            mask = (pre_reg == si) & (post_reg == di)
            n_hits = int(mask.sum())
            if n_hits == 0:
                # nothing found — skip (client falls back to nothing for this pair)
                continue
            hits = np.flatnonzero(mask)
            # Prefer edges whose endpoints are both in the subsampled `keep` list
            pre_k  = keep_inv[pre_all[hits]]
            post_k = keep_inv[post_all[hits]]
            visible = (pre_k >= 0) & (post_k >= 0)
            if visible.any():
                hits_v = hits[visible]
                pre_kv, post_kv = pre_k[visible], post_k[visible]
            else:
                # fall back: any edge; endpoints will use the real xyz coords instead
                hits_v = hits
                pre_kv = np.full(hits.size, -1, dtype=np.int32)
                post_kv = np.full(hits.size, -1, dtype=np.int32)
            k = min(max_edges_per_pair, hits_v.size)
            sel = rng.choice(hits_v.size, size=k, replace=False)
            edges = []
            for j in sel:
                pre_full  = int(pre_all[hits_v[j]])
                post_full = int(post_all[hits_v[j]])
                edges.append([
                    int(pre_kv[j]), int(post_kv[j]),
                    pre_full, post_full,
                ])
            out.append({
                "from": src_name, "to": dst_name,
                "n_total": n_hits,
                "edges": edges,   # each: [pre_keep_idx, post_keep_idx, pre_full, post_full]
            })
        return out

    def step(self, neural_substeps: int = 20) -> dict:
        """Advance one dashboard frame, return the state packet."""
        cur = self.stim.encode(self._current_gesture, self._current_conf)
        # ---- Global tonic + stochastic drive --------------------------------
        # A weak background current + Gaussian noise applied to EVERY neuron
        # so the network shows realistic "always-on" baseline activity in all
        # regions (mimics in-vivo recordings where ~30-60 % of the connectome
        # is subthreshold-active at any moment).  The gesture-locked drive
        # still dominates the sensory populations, so behavior is unchanged.
        rng = np.random.default_rng()
        cur = cur + 4.0 + rng.normal(0.0, 2.5, size=self.N).astype(np.float32)
        spike_sum = np.zeros(self.N, dtype=np.float32)
        last_spikes = None
        for _ in range(neural_substeps):
            # re-jitter noise each sub-step so different neurons win each ms
            noisy = cur + rng.normal(0.0, 1.5, size=self.N).astype(np.float32)
            last_spikes = self.sim.step(noisy)
            spike_sum += last_spikes
        rate = spike_sum / neural_substeps
        self._activity_ema = (
            (1 - self._ema_alpha) * self._activity_ema + self._ema_alpha * rate
        )
        cmd = self.decoder.decode(rate, dt_ms=1.0 * neural_substeps)
        # ---- Hover-hold when no gesture is active ---------------------------
        # Even at rest the sensory populations produce a small stochastic
        # imbalance, which the (high-gain) decoder would translate into a
        # slow drift.  If the user is showing no gesture (or a very low-
        # confidence one), zero the horizontal/attitude channels and hold
        # thrust at the neutral bias so the drone hovers in place.
        idle = (self._current_gesture == Gesture.NONE
                or self._current_conf < 0.25)
        if idle:
            cmd.roll = 0.0
            cmd.pitch = 0.0
            cmd.yaw = 0.0
            cmd.thrust = self.decoder.cfg.thrust_bias
        # ---- Direct kinematic gesture control -------------------------------
        # The quadrotor physics loop, when driven by the neural decoder,
        # produces sluggish, hard-to-read motion.  For a demo dashboard the
        # user needs each gesture to translate the drone in the obvious
        # direction, immediately.  We therefore integrate a body-frame
        # velocity that is set directly by the current gesture, gated by the
        # decoder's confidence, and bypass the full 6-DoF physics while a
        # gesture is active.  When idle, we hard-freeze (see below).
        dt_frame = neural_substeps * 1e-3   # seconds this dashboard step
        s = self.drone.state
        # Target velocities per gesture (m/s in world frame, yaw-aware)
        import math as _math
        G = self._current_gesture
        c = self._current_conf
        v_lin = 2.5 * c                     # m/s
        v_yaw = 1.2 * c                     # rad/s
        v_z   = 1.8 * c                     # m/s
        tgt_vx = tgt_vy = tgt_vz = 0.0
        tgt_wz = 0.0
        # "forward" in world frame given current yaw
        cy, sy = _math.cos(s.yaw), _math.sin(s.yaw)
        if   G == Gesture.UP:         tgt_vz = +v_z
        elif G == Gesture.DOWN:       tgt_vz = -v_z
        elif G == Gesture.LEFT:       tgt_vx = -sy * v_lin; tgt_vy =  cy * v_lin
        elif G == Gesture.RIGHT:      tgt_vx =  sy * v_lin; tgt_vy = -cy * v_lin
        elif G == Gesture.ROTATE_CW:  tgt_wz = -v_yaw
        elif G == Gesture.ROTATE_CCW: tgt_wz = +v_yaw
        if idle:
            # HARD FREEZE: no instruction from the user means the drone is
            # stationary — zero all linear/angular velocities and skip the
            # physics integration so numerical drift can't accumulate into
            # phantom motion.
            for attr in ("vx", "vy", "vz", "wx", "wy", "wz"):
                if hasattr(s, attr):
                    setattr(s, attr, 0.0)
            if hasattr(s, "t"):
                s.t = float(s.t) + dt_frame
        else:
            # Smoothly track target velocity (1st-order lag ~150 ms) so motion
            # starts/stops crisply but not with a jerk.
            k = min(1.0, dt_frame / 0.15)
            s.vx += (tgt_vx - s.vx) * k
            s.vy += (tgt_vy - s.vy) * k
            s.vz += (tgt_vz - s.vz) * k
            s.wz += (tgt_wz - s.wz) * k
            # Integrate position + yaw
            s.x += s.vx * dt_frame
            s.y += s.vy * dt_frame
            s.z += s.vz * dt_frame
            s.yaw += s.wz * dt_frame
            # small visual bank so the drone tilts into motion
            s.roll  = -0.15 * (tgt_vy * cy - tgt_vx * sy) / max(v_lin, 1e-3)
            s.pitch = +0.15 * (tgt_vx * cy + tgt_vy * sy) / max(v_lin, 1e-3)
            # Boundaries
            s.z = max(0.2, min(9.0, s.z))
            s.x = max(-18.0, min(18.0, s.x))
            s.y = max(-18.0, min(18.0, s.y))
            s.t = float(s.t) + dt_frame
        self._total_spikes += int((last_spikes > 0).sum()) if last_spikes is not None else 0


        # per-region firing rate (Hz-ish)
        region_rates = {}
        for name, idx in self.regions.items():
            region_rates[name] = float(self._activity_ema[idx].mean() * 1000.0)

        # per-neuron activity for the visible subsample (client picks its own keep list)
        s = self.drone.state
        rotor_norm = (self.drone.rotor_speeds_norm().tolist()
                      if hasattr(self.drone, "rotor_speeds_norm") else [0.5, 0.5, 0.5, 0.5])
        return {
            "t": float(s.t),
            "gesture": self._current_gesture.value,
            "confidence": self._current_conf,
            "drone": {
                "pos": [float(s.x), float(s.y), float(s.z)],
                "rpy": [float(s.roll), float(s.pitch), float(s.yaw)],
                "vel": [float(s.vx), float(s.vy), float(s.vz)],
                "speed": float(np.linalg.norm([s.vx, s.vy, s.vz])),
                "rotors": rotor_norm,
                "collided": bool(getattr(s, "collided", False)),
                "physics": self.physics_kind,
            },
            "cmd": {
                "thrust": float(cmd.thrust), "roll": float(cmd.roll),
                "pitch": float(cmd.pitch), "yaw": float(cmd.yaw),
            },
            "region_rates": region_rates,
            "activity": self._activity_ema.astype(np.float16).tolist(),
            "stats": {
                "n_active": int((self._activity_ema > 0.05).sum()),
                "total_spikes": int(self._total_spikes),
                "n_total": int(self.N),
                "n_edges": int(self.conn.W.nnz),
            },
        }


# ------------------------------------------------------------ FastAPI app --
def build_app(world: SimulationWorld, fps: int = 15) -> FastAPI:
    app = FastAPI(title="FlyBrain Drone Dashboard")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text())

    @app.get("/api/brain")
    async def brain() -> JSONResponse:
        return JSONResponse(world.brain_payload())

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        period = 1.0 / fps

        async def _recv_loop() -> None:
            try:
                while True:
                    msg = await websocket.receive_text()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "gesture":
                            world.set_gesture(
                                data.get("gesture", "none"),
                                data.get("confidence", 0.0),
                            )
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                return

        recv_task = asyncio.create_task(_recv_loop())
        try:
            while True:
                t0 = time.time()
                packet = world.step()
                await websocket.send_text(json.dumps(packet))
                dt = time.time() - t0
                if dt < period:
                    await asyncio.sleep(period - dt)
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            recv_task.cancel()

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connectome", choices=["gesture", "flywire"], default="flywire")
    p.add_argument("--max-neurons", type=int, default=12000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--physics", choices=["simple", "quadrotor"], default="quadrotor")
    a = p.parse_args()
    world = SimulationWorld(a.connectome, max_neurons=a.max_neurons, physics=a.physics)
    app = build_app(world, fps=a.fps)
    print(f"[web] open http://{a.host}:{a.port}/  in your browser")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
