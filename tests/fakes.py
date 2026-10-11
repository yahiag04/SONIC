"""Test doubles. These are NOT models and are never used outside tests.

FakeEmbedder maps audio windows to a 2-D "embedding" whose angle depends on
window energy, so louder windows look more like the target text. This lets the
localization logic be tested deterministically without CLAP.
"""

from __future__ import annotations

import numpy as np


class FakeEmbedder:
    def __init__(self, target_text: str, energy_scale: float = 10.0):
        self.target_text = target_text
        self.energy_scale = energy_scale
        self.audio_calls = 0

    def embed_text(self, texts):
        out = np.array([[1.0, 0.0] if t == self.target_text else [0.0, 1.0] for t in texts])
        return out

    def embed_audio(self, windows):
        self.audio_calls += 1
        e = np.array([np.sqrt(np.mean(np.square(w))) * self.energy_scale for w in windows])
        v = np.stack([e, np.ones_like(e)], axis=1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeSeparator:
    """Returns a fixed fraction of a known target plus a fixed leakage of a known background."""

    def __init__(self, target: np.ndarray, background: np.ndarray, gain: float = 0.9, leak: float = 0.05):
        self.target = target
        self.background = background
        self.gain = gain
        self.leak = leak

    def separate(self, wav, text):
        assert len(wav) == len(self.target)
        return (self.gain * self.target + self.leak * self.background).astype(np.float32)
