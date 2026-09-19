"""Milestone-1 closed-loop demo.

Camera → visual encoder → connectome (LIF) → motor decoder → drone → new camera.

Run:
    python -m flybrain.demo.hover_loop --steps 500

This is a *proof-of-concept*, not a demonstration of biological control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.connectome.loader import load_synthetic
from flybrain.neuroscience.neuron_models import LIFModel
from flybrain.neuroscience.simulator import NeuralSimulator
from flybrain.vision.camera import SyntheticCamera
from flybrain.vision.encoder import VisualEncoder, EncoderConfig
from flybrain.drone.simulator import SimplePhysicsSim
from flybrain.drone.physics import DroneState
from flybrain.decoding.motor_decoder import PopulationMotorDecoder, MotorDecoderConfig
from flybrain.visualization.activity import population_activity
from flybrain.visualization.trajectory import TrajectoryLogger


def run(steps: int = 500, out_dir: Path = Path("experiments/demo_run"), seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    conn, ann = load_synthetic(seed=seed)
    model = LIFModel(conn.num_neurons)
    sim = NeuralSimulator(conn, model, dt_ms=1.0)

    encoder = VisualEncoder(
        EncoderConfig(resolution=(16, 16), channels=("brightness", "motion")),
        num_neurons=conn.num_neurons,
        sensory_indices=ann.get("visual").neuron_indices,
        rng=np.random.default_rng(seed),
    )
    camera = SyntheticCamera(resolution=(32, 32), target_xy=(3.0, 0.5))
    drone = SimplePhysicsSim(DroneState(z=1.0))
    decoder = PopulationMotorDecoder(ann, MotorDecoderConfig(thrust_bias=0.0))

    drone_dt = 0.01   # 100 Hz drone; 10× neural steps per physics step.
    neural_steps_per_drone = 10

    log_path = out_dir / "trajectory.csv"
    activity_log = []
    total_spikes = 0

    with TrajectoryLogger(log_path) as tlog:
        for k in range(steps):
            frame = camera.capture(drone.state)
            ext_current = encoder.encode(frame.rgb)
            for _ in range(neural_steps_per_drone):
                spikes = sim.step(ext_current)
                total_spikes += int(spikes.sum())
            cmd = decoder.decode(sim.state.spikes, dt_ms=sim.dt * neural_steps_per_drone)
            drone.step(cmd, drone_dt)
            tlog.log(drone.state)

            if k % 50 == 0:
                pa = population_activity(sim.state.spikes, ann)
                print(f"step={k:4d}  t={drone.state.t:6.2f}s  "
                      f"pos=({drone.state.x:+.2f},{drone.state.y:+.2f},{drone.state.z:.2f})  "
                      f"cmd=(T={cmd.thrust:+.2f} R={cmd.roll:+.2f} P={cmd.pitch:+.2f} Y={cmd.yaw:+.2f})  "
                      f"spikes/step≈{total_spikes / max(1, (k+1)*neural_steps_per_drone):.1f}  "
                      f"pop={ {k2:round(v,3) for k2,v in pa.items()} }")
                activity_log.append({"step": k, **pa, "cmd_thrust": cmd.thrust})

    metrics = {
        "steps": steps,
        "final_position": [drone.state.x, drone.state.y, drone.state.z],
        "total_spikes": total_spikes,
        "collided_final": drone.state.collided,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "activity_log.json").write_text(json.dumps(activity_log, indent=2))
    print(f"\nDone. Wrote {log_path} and metrics.json under {out_dir}/")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("experiments/demo_run"))
    a = p.parse_args()
    run(steps=a.steps, out_dir=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
