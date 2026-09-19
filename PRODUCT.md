# 📦 Product Overview — FlyBrain Drone

> **Turning a real fly's brain into a live, hand-controlled flight system —
> a window into brain-based (neuromorphic) computing.**

---

## 1. The one-liner

**FlyBrain Drone lets anyone pilot a simulated drone using hand gestures, where the
"pilot" is a real map of a fruit fly's brain running in real time.** It is a playable,
visual demonstration that biological neural wiring — not hand-coded rules — can be used
to translate perception into action.

---

## 2. Why it matters

Nature already solved efficient, real-time flight control inside an insect brain smaller
than a grain of sand. **Neuromorphic computing** — building machines that compute the
way brains do — is an active research frontier (Intel Loihi, SpiNNaker, DARPA programs).
FlyBrain Drone makes that idea tangible:

- It uses a **real connectome** (the FlyWire map of ~100k neurons and their synapses),
  not a made-up network.
- It runs a **spiking neural simulation** in the loop, so you literally watch signals
  propagate from "eye" to "muscle".
- It closes the loop to a **physical control task** (flying a quad-rotor), which is what
  makes it more than a pretty visualization.

The result is a portfolio-grade, science-communication-ready artifact that is equally at
home in a classroom, a research demo, or a LinkedIn post.

---

## 3. Who it's for

| Audience | Value |
|----------|-------|
| **Students & educators** | A hands-on way to teach connectomics, spiking neurons, and sensorimotor loops. |
| **Neuroscience / ML researchers** | A reproducible sandbox to swap neuron models, decoders, and tasks over real connectivity. |
| **Neuromorphic / robotics engineers** | A "connectome-in-the-loop" reference for brain-based control. |
| **Science communicators & recruiters** | An eye-catching, self-explanatory demo of applied neuroscience + engineering. |
| **Curious tinkerers** | Clone, run, wave your hand, fly a drone with a fly's brain. |

---

## 4. Core features

1. **Real-time gesture piloting** — browser-side MediaPipe hand tracking → intuitive
   finger-count poses for ascend/descend, strafe, forward/back, rotate, arm/disarm.
2. **Connectome-in-the-loop** — a LIF spiking network over the real FlyWire matrix maps
   sensory input to motor output every frame.
3. **Living 3D brain** — real soma coordinates, neuropil hulls, dendritic/axonal
   morphology, and animated synaptic signal pulses.
4. **Quad-rotor world** — realistic drone model, follow-cam, trajectory trail, gates,
   collision flashes, live HUD.
5. **Instrumentation** — per-region firing-rate charts, motor-output bars, spike stats.
6. **Neural sonification** — hear the brain's activity as live, spatialized audio.
7. **Zero-friction deploy** — one Python command, one Docker container, or Hugging Face
   Spaces.

---

## 5. How it works (at a glance)

```
📷 Camera → 👁 Visual encoding → 🧠 Fly connectome (LIF) → ⚡ Motor decoder → 🚁 Drone
    ▲                                                                             │
    └──────────────────────────── closed loop ◄──────────────────────────────────┘
```

- **Perception:** hand → retinal features → gesture class.
- **Cognition:** gesture drives sensory neurons; spikes propagate through real synapses.
- **Action:** population activity is decoded into thrust/roll/pitch/yaw; the drone flies;
  the new frame feeds back in.

See the [`README`](./README.md#-how-it-works) for the detailed pipeline.

---

## 6. What's real vs. modelled (honesty statement)

- ✅ **Real:** the connectome structure — neurons, synapses, 3D positions, cell classes
  (FlyWire FAFB v783).
- 🧪 **Modelled conventions:** the LIF neuron dynamics, the pixel→sensory-current visual
  encoder, and the population→drone motor decoder. The gesture-to-cell-type mapping is a
  computational convention, not a biological claim.

This distinction is deliberate and documented in-code so the project stays scientifically
honest.

---

## 7. Design principles

- **Runs on a laptop.** No GPU required; neuron count is capped and configurable.
- **Clone-and-go.** Real data ships with the repo; no accounts, keys, or downloads.
- **Legible over clever.** Every module states its assumptions; the UI narrates the
  5-stage pipeline.
- **Swappable parts.** Neuron model, decoder, physics, and connectome are all behind
  clean interfaces.

---

## 8. Roadmap

| Stage | Status | Description |
|------:|:------:|-------------|
| Real connectome in the loop | ✅ | FlyWire FAFB v783 driving a LIF sim |
| Live 3D brain + drone dashboard | ✅ | Three.js visualization, WebSocket streaming |
| Robust hand-gesture control | ✅ | Finger-count + direction scheme, smoothing |
| Forward/back + richer control set | ✅ | 8 flight gestures + arm/disarm |
| Neural sonification | ✅ | Region activity → live audio |
| Obstacle / gate navigation tasks | ⏳ | Scored courses, autonomy metrics |
| Learned motor decoder (RL / BC) | ⏳ | Replace hand-tuned decoder with trained policy |
| Ablation & benchmarking | ⏳ | Compare vs. conventional NN baselines |
| Alternative connectomes | ⏳ | hemibrain / MANC, larva, other species |

---

## 9. Success criteria

- A newcomer can **clone and fly within ~2 minutes**.
- The pipeline is **visible and explainable** end-to-end on one screen.
- Researchers can **swap a component** (neuron model / decoder / task) without touching
  the rest.
- The demo is compelling enough to **stand alone in a talk or post**.

---

## 10. FAQ

**Is this a real fly brain thinking?** It runs a spiking simulation over a real map of a
fly brain's wiring. The *structure* is real; the *dynamics* are a standard model. It is
a demonstration of connectome-based control, not a living organism.

**Do I need a GPU or special hardware?** No — just Python 3.10+ and a browser. A webcam is
optional (there's a simulated mode).

**Can I use my own connectome or decoder?** Yes. The loader, neuron model, decoder, and
physics are modular. See the `src/flybrain/` layout in the README.

**What license?** Code is MIT; the FlyWire data is CC BY 4.0. Attribute the FlyWire
consortium if you reuse the data.
