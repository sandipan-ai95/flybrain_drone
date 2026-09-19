# Contributing to FlyBrain Drone

Thanks for your interest in improving FlyBrain Drone! 🧠🚁 This project is meant to be
easy to hack on. Whether it's a bug fix, a new gesture, a better decoder, or docs — PRs
are welcome.

## Getting set up

```bash
git clone https://github.com/sandipan-ai95/flybrain_drone.git
cd flybrain_drone
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # installs dev extras (pytest, ruff)
```

Run the app:

```bash
python -m flybrain.server.dashboard_server
```

Run the checks before you push:

```bash
pytest -q          # tests
ruff check .       # lint
```

## Where things live

| You want to change… | Look in… |
|---------------------|----------|
| Hand gestures / detection | `src/flybrain/server/static/index.html` (`classifyHandGesture`) |
| Gesture → neural stimulus | `src/flybrain/vision/stimulus.py`, `src/flybrain/vision/gesture.py` |
| Neuron model / spiking sim | `src/flybrain/neuroscience/` |
| Motor decoding | `src/flybrain/decoding/` |
| Drone physics | `src/flybrain/drone/` |
| Connectome loading | `src/flybrain/connectome/` |
| Dashboard UI / 3D | `src/flybrain/server/static/index.html` |
| Server / API / WebSocket | `src/flybrain/server/dashboard_server.py` |

## Good first issues

- Add a new flight gesture (define it in the classifier + `Gesture` enum + server map).
- Improve the visual encoder (edges/flow → richer sensory currents).
- Add a scored navigation task (fly through the gates).
- Swap in a learned motor decoder (behaviour cloning / RL).
- Improve mobile / small-screen layout of the dashboard.

## Pull request guidelines

1. **Keep it focused.** One logical change per PR.
2. **Explain the "why."** A short description of the problem and approach helps a lot.
3. **Stay scientifically honest.** If you add a modelling assumption, document it in the
   module that makes it (this is a core project value).
4. **Don't commit large data.** The FlyWire CSVs already ship; don't add new big binaries.
5. **Tests / lint pass.** Run `pytest -q` and `ruff check .` locally.

## Code style

- Python: follow `ruff` (line length 100, target py310). Prefer clear names and small
  functions.
- JavaScript (dashboard): keep the single-file dashboard readable; comment non-obvious
  math (coordinate frames, gesture geometry).

## Reporting bugs

Open an issue with:
- what you did, what you expected, what happened,
- your OS + browser + Python version,
- any console / terminal errors.

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](./LICENSE).
