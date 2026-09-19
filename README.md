---
title: FlyBrain Drone
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FlyBrain Drone

A research simulation investigating whether the neural connectivity architecture of
the *Drosophila melanogaster* connectome can serve as the computational substrate
for controlling a simulated flying drone.

## 🚀 Deploy

- **Local:** `pip install -r requirements.txt && pip install -e . && python -m flybrain.server.dashboard_server`
- **Docker:** `docker build -t flybrain-drone . && docker run --rm -p 7860:7860 flybrain-drone`
- **Hugging Face Spaces:** push this repo to a Space with **SDK: Docker** — the YAML header above and the included `Dockerfile` are all you need.

Full instructions and tuning knobs are in [`DEPLOYMENT.md`](./DEPLOYMENT.md).

> **Closed loop:** camera → visual encoder → connectome → neural propagation →
> motor decoder → drone physics → new camera frame → …

## ⚠️ Scientific honesty (read first)

This project is **not** a biologically faithful simulation of a living fly brain.
It combines:

1. **Experimentally measured structural connectivity** (from open connectome datasets
   such as FlyWire / hemibrain, ingested in later milestones).
2. **Computational modelling assumptions** we add on top:
   * A **Leaky Integrate-and-Fire (LIF)** neuron model — replaceable via the
     `NeuralModel` interface (see `src/neuroscience/neuron_models.py`).
   * A **hand-crafted visual encoder** mapping camera pixels to sensory-neuron
     currents. Retinal / optic-lobe cell-type identity is *approximated* until
     real annotations are wired in (Milestone 3).
   * A **motor decoder** that reads population activity and emits drone commands.
     Biological motor-neuron identities are *not* invented — the decoder is
     labelled as experimental infrastructure, separate from the connectome.

Every assumption is documented in the module that makes it.

## Architecture

```
3D env → camera → visual encoder → connectome (LIF sim) → motor decoder → drone → …
```

See §1 of the project spec and `docs/architecture.md` (added in Milestone 3).

## Project layout

```
flybrain-drone/
├── README.md
├── pyproject.toml
├── configs/               # YAML experiment configs
├── data/                  # raw / processed / metadata / annotations
├── src/
│   ├── connectome/        # loader.py, graph.py, annotations.py
│   ├── neuroscience/      # neuron_models.py, simulator.py, synapses.py
│   ├── vision/            # camera.py, encoder.py, optical_flow.py
│   ├── drone/             # simulator.py, physics.py, controller.py
│   ├── decoding/          # motor_decoder.py, learned_decoder.py
│   ├── visualization/     # brain.py, activity.py, trajectory.py
│   └── experiments/       # evaluation.py, ablation.py
├── tests/
├── notebooks/
└── experiments/
```

## Milestones

| # | Status | Description |
|---|--------|-------------|
| 1 | ✅ this commit | Tiny synthetic connectome, LIF sim, stub camera + drone, closed loop |
| 2 | ⏳ | Small real Drosophila visual subgraph (FlyWire / hemibrain) |
| 3 | ⏳ | Real connectome annotations (neuron types, populations) |
| 4 | ⏳ | Realistic visual input (PyBullet camera / procedural stimuli) |
| 5 | ⏳ | Live neural activity 3D visualization |
| 6 | ⏳ | Obstacle / gate navigation tasks |
| 7 | ⏳ | Learned motor decoder (PPO / behaviour cloning) |
| 8 | ⏳ | Ablation experiments |
| 9 | ⏳ | Full research dashboard |
| 10| ⏳ | Benchmark vs. conventional NN baselines |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Heavy dependencies (PyBullet, PyTorch, PyTorch-Geometric, dash/plotly) are optional
extras and pulled in only when the corresponding milestone is enabled.

## Run the Milestone-1 demo

```bash
python -m flybrain.demo.hover_loop --steps 500
```

You should see the drone react to a moving bright target by shifting its motor
output, with per-step spike counts and drone state printed to stdout. Trajectory
and neural activity are saved to `experiments/demo_run/`.

## Run tests

```bash
pytest -q
```

## Dataset sources (used from Milestone 2)

* **FlyWire** — full female adult fly brain connectome (CC BY 4.0 for the public release).
  <https://flywire.ai/>
* **hemibrain v1.2.1** — Janelia (CC BY 4.0). <https://www.janelia.org/project-team/flyem/hemibrain>
* **MANC** — male adult nerve cord, for motor-neuron populations.

We do **not** redistribute raw data; the loader downloads from the official sources
under their licenses. See `src/connectome/loader.py`.

## License

Code: MIT. Data: subject to each dataset's own license.
