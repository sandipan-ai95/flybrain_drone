# Deployment Guide — FlyBrain Drone Dashboard

The dashboard is a **FastAPI + WebSocket** server that streams neural /
drone simulation state to a browser. All heavy client work — webcam
capture, MediaPipe hand tracking, Three.js rendering — runs **in the
browser**, so the server stays small and stateless-per-connection.

Because the app needs a persistent WebSocket, it **cannot** be hosted on
pure static hosts (GitHub Pages, Netlify static, etc.). Use a container
host: Hugging Face Spaces (Docker SDK), Fly.io, Railway, Render, or any
Docker-capable VM.

---

## 1. Local (native Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m flybrain.server.dashboard_server --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/> and **allow camera access** when prompted.

## 2. Local (Docker)

```bash
docker build -t flybrain-drone .
docker run --rm -p 7860:7860 flybrain-drone
```

Open <http://localhost:7860/>.

> ⚠️ **Camera + `getUserMedia`** requires a secure context. `localhost`
> and `127.0.0.1` are treated as secure by browsers, so local dev works.
> In production you **must** serve over HTTPS.

## 3. Hugging Face Spaces (recommended, free tier works)

Spaces with the **Docker SDK** support FastAPI + WebSockets out of the
box, terminate TLS for you (so `getUserMedia` works), and expose the
container on `$PORT = 7860`.

### Steps

1. Create a new Space → **SDK: Docker** → link to a GitHub repo or push
   directly.
2. Add this header to a `README.md` **at the root of the Space**
   (Spaces reads YAML frontmatter):

   ```yaml
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
   ```

3. Commit the existing `Dockerfile`, `requirements.txt`, `src/`,
   `data/`, `configs/`, `pyproject.toml`, `README.md`.
4. The Space will build and run automatically. WebSockets are proxied
   transparently — no code changes needed.

### Runtime tuning via Space **Variables**

| Variable                 | Default     | Notes                                      |
| ------------------------ | ----------- | ------------------------------------------ |
| `FLYBRAIN_CONNECTOME`    | `gesture`   | `gesture` (fast) or `flywire` (heavy)      |
| `FLYBRAIN_MAX_NEURONS`   | `8000`      | Lower on free CPU tier                     |
| `FLYBRAIN_FPS`           | `15`        | Server tick / stream rate                  |
| `FLYBRAIN_PHYSICS`       | `quadrotor` | `simple` or `quadrotor`                    |

On the free CPU basic tier, keep `max-neurons ≤ 8000` and `fps ≤ 15`.

## 4. GitHub (source hosting)

Push the repo to GitHub. GitHub Pages **cannot** run the backend, but:

* Hugging Face Spaces can be linked directly to a GitHub repo and will
  redeploy on push.
* Add a **"Deploy to Spaces"** badge to the README:

  ```markdown
  [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/spaces/new?sdk=docker)
  ```

## 5. Any other Docker host

The same `Dockerfile` works on Fly.io, Railway, Render, or a plain VM.
Point the platform at port `7860` (or set `PORT` and let the CMD pick it
up automatically) and make sure **WebSocket upgrades** are allowed by
the platform's proxy (they are on Fly.io, Railway, Render, HF Spaces).

## Troubleshooting

* **Black webcam / no hand landmarks** — browser blocked camera, or you
  loaded the page over plain HTTP from a non-`localhost` origin.
* **`/api/brain` 500** — connectome data missing. Ensure `data/` is
  copied into the image (the `Dockerfile` does this) or switch to
  `--connectome gesture`.
* **Drone doesn't respond to gestures** — check the browser console;
  the gesture classifier logs the smoothed gesture each tick.
* **HF Space stuck "Building"** — free tier images must be < 50 GB;
  ours is well under that. Check the build logs for missing wheels.
