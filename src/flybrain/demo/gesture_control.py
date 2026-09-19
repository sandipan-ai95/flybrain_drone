"""Gesture-controlled fly-connectome drone demo.

Pipeline:

    webcam frame ──► hand landmarks ──► gesture label
                                          │
                                          ▼
                              sensory-population current
                                          │
                                          ▼
                                  LIF connectome
                                          │
                                          ▼
                              motor-population spikes
                                          │
                                          ▼
                              PopulationMotorDecoder
                                          │
                                          ▼
                                   drone physics
                                          │
                                          ▼
                            LiveDashboard (records MP4)

Usage
-----
    # webcam + MediaPipe (requires: pip install -e ".[gesture]" and ffmpeg for MP4)
    python -m flybrain.demo.gesture_control

    # headless / no camera — uses a scripted gesture sequence, still records MP4
    python -m flybrain.demo.gesture_control --scripted --no-show --steps 600
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain.connectome.loader import load_gesture
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator
from flybrain.decoding.motor_decoder import PopulationMotorDecoder, MotorDecoderConfig
from flybrain.drone.simulator import SimplePhysicsSim
from flybrain.drone.physics import DroneState, DroneCommand
from flybrain.vision.gesture import (
    Gesture, MediaPipeGestureRecognizer, ScriptedGestureRecognizer,
)
from flybrain.vision.stimulus import GestureStimulus
from flybrain.visualization.unified_dashboard import UnifiedDashboard


def _decoder_for_gesture_conn(ann) -> PopulationMotorDecoder:
    """Motor decoder for both synthetic and FlyWire connectomes.

    Reads the firing rate of each *sensory* T4/T5 population directly and
    maps them to drone commands. Rationale: the FlyWire loader currently
    splits descending neurons into ``motor_*`` sub-populations by ``side``
    only, so the differential ``motor_thrust − motor_descent`` is ~noise
    and produces no usable command. The T4/T5 sensory rates *do* carry the
    gesture, so we drive control from them and let the connectome
    modulate the signal along the way.

    Includes a **hover bias** so the drone doesn't fall out of the sky
    when no gesture is present.
    """
    cfg = MotorDecoderConfig(
        thrust_gain=6.0,        # up-vs-down thrust modulation
        roll_gain=2.5, yaw_gain=2.5, pitch_gain=0.0,
        thrust_bias=0.6,        # baseline hover (net vertical accel offset)
        tau_ms=120.0,
        populations={"up": "sensory_up", "down": "sensory_down",
                     "left": "sensory_left", "right": "sensory_right",
                     "cw": "sensory_rot_cw", "ccw": "sensory_rot_ccw"},
    )
    return _DifferentialDecoder(ann, cfg)


class _DifferentialDecoder(PopulationMotorDecoder):
    """Read sensory-population firing rates directly.

    Hover bias keeps the drone in the air with no input; gesture
    populations then push it up/down or roll/yaw.
    """

    def decode(self, spikes: np.ndarray, dt_ms: float) -> DroneCommand:
        alpha = dt_ms / self.cfg.tau_ms
        for key, pop in self.cfg.populations.items():
            idx = self.ann.get(pop).neuron_indices if pop in self.ann.populations else np.array([], np.int32)
            inst = float(spikes[idx].mean()) if idx.size else 0.0
            self.rates[key] += alpha * (inst - self.rates[key])
        r = self.rates
        # Normalise: subtract baseline so idle firing → 0 command.
        u = r["up"] - r["down"]
        lr = r["right"] - r["left"]
        cwccw = r["cw"] - r["ccw"]
        return DroneCommand(
            thrust=self.cfg.thrust_bias + self.cfg.thrust_gain * u,
            roll=self.cfg.roll_gain * lr,
            pitch=0.0,
            yaw=self.cfg.yaw_gain * cwccw,
        )


# --------------------------------------------------------------------------- run
def run(
    steps: int = 600,
    scripted: bool = False,
    device: int = 0,
    show: bool = True,
    out_dir: Path = Path("experiments/gesture_run"),
    seed: int = 42,
    fps: int = 15,
    connectome_source: str = "gesture",
) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Connectome + neural sim
    if connectome_source == "flywire":
        from flybrain.connectome.flywire_loader import load_flywire
        conn, ann = load_flywire()
    elif connectome_source == "hemibrain":
        from flybrain.connectome.hemibrain_loader import load_hemibrain
        conn, ann = load_hemibrain(max_neurons=5000)
    else:
        conn, ann = load_gesture(seed=seed)
    print(f"[demo] connectome: N={conn.num_neurons}  edges={conn.W.nnz}")
    sim = NeuralSimulator(conn, LIFModel(conn.num_neurons), dt_ms=1.0)
    stim = GestureStimulus(ann, drive_current=20.0)
    decoder = _decoder_for_gesture_conn(ann)
    drone = SimplePhysicsSim(DroneState(z=1.5))

    # 2. Gesture input
    if scripted:
        recog = ScriptedGestureRecognizer()
        webcam = None
        print("[demo] using scripted gesture sequence (no webcam)")
    else:
        webcam = None
        recog = None
        try:
            from flybrain.vision.webcam import WebcamCapture
            # device=None → auto-pick the built-in landscape camera and skip
            # portrait iPhone/Continuity feeds.
            webcam = WebcamCapture(device=(device if device >= 0 else None))
        except Exception as e:
            print(f"[demo] ❌ webcam failed: {e}")
        if webcam is not None:
            try:
                recog = MediaPipeGestureRecognizer()
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[demo] ❌ MediaPipe failed: {e}")
                webcam.release(); webcam = None
        if webcam is None or recog is None:
            print("[demo] falling back to scripted gesture sequence")
            recog = ScriptedGestureRecognizer()
            webcam = None
        else:
            print("[demo] ✅ webcam + MediaPipe active")

    # 3. Unified dashboard (single screen, records one MP4)
    dash = UnifiedDashboard(conn, ann, raster_window=200,
                            record_path=out_dir / "session.mp4",
                            fps=fps, show=show, rotate_brain=True)

    # 4. Rolling activity EMA for pretty brain colouring
    activity_ema = np.zeros(conn.num_neurons, dtype=np.float32)
    ema_alpha = 0.15

    # 5. Log everything for evaluation
    log = {"t": [], "gesture": [], "confidence": [],
           "spikes_per_step": [], "cmd": [], "pos": []}

    neural_steps_per_drone = 10
    drone_dt = 0.01
    frame_period = 1.0 / fps
    t_last = time.time()

    try:
        for k in range(steps):
            # -- 5a. Sense
            rgb = webcam.read() if webcam is not None else None
            gres = recog.detect(rgb) if rgb is not None else recog.detect(None)
            ext_current = stim.encode(gres.gesture, gres.confidence)

            # -- 5b. Brain (accumulate spikes across the sub-steps so the
            # decoder sees a firing *rate*, not just the last binary vector)
            step_spikes = 0
            spike_sum = np.zeros(conn.num_neurons, dtype=np.float32)
            spikes = None
            for _ in range(neural_steps_per_drone):
                spikes = sim.step(ext_current)
                spike_sum += spikes
                step_spikes += int(spikes.sum())
            spike_rate = spike_sum / neural_steps_per_drone     # 0..1 per neuron
            activity_ema = (1 - ema_alpha) * activity_ema + ema_alpha * spike_rate

            # -- 5c. Motor + drone
            cmd = decoder.decode(spike_rate,
                                 dt_ms=sim.dt * neural_steps_per_drone)
            drone.step(cmd, drone_dt)

            # -- 5d. Log
            log["t"].append(drone.state.t)
            log["gesture"].append(gres.gesture.value)
            log["confidence"].append(gres.confidence)
            log["spikes_per_step"].append(step_spikes)
            log["cmd"].append([cmd.thrust, cmd.roll, cmd.pitch, cmd.yaw])
            log["pos"].append([drone.state.x, drone.state.y, drone.state.z])

            # -- 5e. Visualize (single screen: camera + 3-D brain + raster + drone)
            dash.update(
                webcam_rgb=rgb,
                gesture=gres.gesture.value, confidence=gres.confidence,
                spikes=(spikes if spikes is not None else np.zeros(conn.num_neurons)),
                activity_ema=activity_ema,
                drone_pos=(drone.state.x, drone.state.y, drone.state.z),
                drone_orientation=(drone.state.roll, drone.state.pitch, drone.state.yaw),
                cmd=(cmd.thrust, cmd.roll, cmd.pitch, cmd.yaw),
                t=drone.state.t,
                info_lines=[f"active neurons : {int((activity_ema>0.05).sum()):4d}",
                            f"step spikes    : {step_spikes:4d}"],
            )

            # console tick every second
            if k % fps == 0:
                r = decoder.rates
                print(f"t={drone.state.t:5.2f}s  gesture={gres.gesture.value:>7}  "
                      f"spikes={step_spikes:4d}  "
                      f"rates(U={r.get('up',0):.3f} D={r.get('down',0):.3f} "
                      f"L={r.get('left',0):.3f} R={r.get('right',0):.3f})  "
                      f"cmd(T={cmd.thrust:+.2f} R={cmd.roll:+.2f} Y={cmd.yaw:+.2f})  "
                      f"pos=({drone.state.x:+.2f},{drone.state.y:+.2f},{drone.state.z:.2f})")

            # cap loop at target fps when we have a display
            if show:
                dt_wall = time.time() - t_last
                if dt_wall < frame_period:
                    time.sleep(frame_period - dt_wall)
                t_last = time.time()
    finally:
        if webcam is not None:
            webcam.release()
        recog.close()
        dash.close()

    # 6. Persist evaluation logs
    (out_dir / "gesture_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n[demo] wrote {out_dir}/session.mp4 (if ffmpeg installed) "
          f"and gesture_log.json ({len(log['t'])} frames).")
    return {"frames": len(log["t"]), "final_pos": log["pos"][-1] if log["pos"] else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--scripted", action="store_true", help="no webcam; use scripted gestures")
    p.add_argument("--device", type=int, default=-1,
                   help="webcam device index; -1 = auto-pick built-in "
                        "landscape camera (skips iPhone/Continuity).")
    p.add_argument("--list-cameras", action="store_true",
                   help="probe attached cameras and exit.")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--out", type=Path, default=Path("experiments/gesture_run"))
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--connectome", choices=["gesture", "flywire", "hemibrain"], default="gesture",
                   help="'gesture' = built-in synthetic; "
                        "'flywire' = FlyWire FAFB CSVs in data/raw/flywire/; "
                        "'hemibrain' = live Janelia neuprint (no login required)")
    a = p.parse_args()
    if a.list_cameras:
        from flybrain.vision.webcam import probe_devices
        for i, w, h in probe_devices():
            tag = " (portrait — likely iPhone)" if h > w else ""
            print(f"  device {i}: {w}x{h}{tag}")
        return
    run(steps=a.steps, scripted=a.scripted, device=a.device, show=a.show,
        out_dir=a.out, fps=a.fps, seed=a.seed, connectome_source=a.connectome)


if __name__ == "__main__":
    main()
