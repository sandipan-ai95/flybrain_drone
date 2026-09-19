"""Visual encoder: RGB frame → sensory-neuron input current vector.

Spec §4/§5: do NOT pass raw pixels directly into the connectome. Extract
retinal-style features (brightness, contrast, motion), then map them onto a
declared sensory population.

Scientific assumption (Milestone 1): sensory-population neuron indices are
provided by :mod:`flybrain.connectome.annotations`. In the synthetic connectome
those are just neurons 0..n_visual-1. In real datasets they will be the
identified photoreceptor / lamina / medulla neurons from FlyWire annotations
(wired in Milestone 3). The retinotopic mapping used here (row-major raster
into the visual population) is a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .optical_flow import frame_difference


@dataclass
class EncoderConfig:
    resolution: tuple[int, int] = (16, 16)
    channels: tuple[str, ...] = ("brightness", "motion")
    gain: float = 15.0                  # scales current into LIF units (mV via r_m)
    noise_std: float = 0.0


class VisualEncoder:
    def __init__(
        self,
        config: EncoderConfig,
        num_neurons: int,
        sensory_indices: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = config
        self.num_neurons = num_neurons
        self.sensory_indices = np.asarray(sensory_indices, dtype=np.int64)
        self.rng = rng or np.random.default_rng(0)
        self._prev_rgb_small: np.ndarray | None = None

        self.h, self.w = config.resolution
        self.n_pixels = self.h * self.w
        self.n_ch = len(config.channels)
        # Random retinotopic projection: (n_sensory,) ← (n_pixels * n_ch,)
        # Sparse random binary receptive fields.
        self.rf = self._build_receptive_fields()

    def _build_receptive_fields(self) -> np.ndarray:
        n_s = self.sensory_indices.size
        feat_dim = self.n_pixels * self.n_ch
        # Each sensory neuron listens to ~4 random features.
        rf = np.zeros((n_s, feat_dim), dtype=np.float32)
        for i in range(n_s):
            idx = self.rng.integers(0, feat_dim, size=4)
            rf[i, idx] = 1.0 / 4.0
        return rf

    # ---------------------------------------------------------------- encode
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """Return a full-length current vector (shape (num_neurons,))."""
        # Resize by simple slicing/pooling.
        img = self._resize(rgb)
        feats = []
        for ch in self.cfg.channels:
            if ch == "brightness":
                feats.append(img.astype(np.float32).mean(axis=-1).ravel() / 255.0)
            elif ch == "motion":
                feats.append(frame_difference(self._prev_rgb_small, img).ravel())
            elif ch == "contrast":
                gray = img.astype(np.float32).mean(axis=-1) / 255.0
                gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
                gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
                feats.append((gx + gy).ravel())
            else:
                raise ValueError(f"unknown channel {ch!r}")
        feat_vec = np.concatenate(feats).astype(np.float32)
        sensory_current = self.rf @ feat_vec        # (n_sensory,)
        sensory_current *= self.cfg.gain
        if self.cfg.noise_std > 0:
            sensory_current += self.rng.normal(0, self.cfg.noise_std, sensory_current.shape).astype(np.float32)
        # Scatter into full vector.
        out = np.zeros(self.num_neurons, dtype=np.float32)
        out[self.sensory_indices] = sensory_current
        self._prev_rgb_small = img
        return out

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        ys = (np.linspace(0, H - 1, self.h)).astype(np.int32)
        xs = (np.linspace(0, W - 1, self.w)).astype(np.int32)
        return rgb[np.ix_(ys, xs)]
