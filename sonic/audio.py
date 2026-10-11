"""Audio I/O and signal utilities shared by all SONIC modules.

All internal processing uses mono float32 at ``SR`` (32 kHz, AudioSep's rate).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 32000
EPS = 1e-12


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average channels of a (samples, channels) array; 1-D input is returned as is."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    """Polyphase resampling; output length is round(len(x) * new_sr / orig_sr)."""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == new_sr:
        return x
    frac = Fraction(new_sr, orig_sr)
    y = resample_poly(x, frac.numerator, frac.denominator)
    target = int(round(len(x) * new_sr / orig_sr))
    if len(y) > target:
        y = y[:target]
    elif len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y.astype(np.float32)


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load an audio file as mono float32 at ``sr``."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    return resample(to_mono(x), file_sr, sr)


def save_audio(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Write mono float audio as 32-bit float WAV (no quantization of results)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + EPS))


def scale_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    return x * (target_rms / rms(x))


def scale_to_snr(signal: np.ndarray, reference: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``signal`` so that 20*log10(rms(signal)/rms(reference)) == snr_db."""
    target = rms(reference) * 10.0 ** (snr_db / 20.0)
    return scale_to_rms(signal, target)


def fit_peak(stems: dict[str, np.ndarray], peak: float = 0.9) -> tuple[dict[str, np.ndarray], float]:
    """Apply one common gain to every stem so the mixture peak is <= ``peak``.

    Using a shared gain keeps ``mixture == sum(stems)`` exact and preserves SNRs.
    Returns the (possibly rescaled) stems and the gain applied.
    """
    mix = sum(stems.values())
    mix_peak = float(np.max(np.abs(mix)))
    if mix_peak <= peak:
        return dict(stems), 1.0
    gain = peak / mix_peak
    return {k: v * gain for k, v in stems.items()}, gain


def frame_starts(n: int, win: int, hop: int) -> np.ndarray:
    """Start indices of windows of length ``win`` (step ``hop``) covering ``n`` samples.

    The final window is aligned to the end of the signal so no samples are left out.
    """
    if n <= win:
        return np.array([0], dtype=np.int64)
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return np.array(starts, dtype=np.int64)


def raised_cosine(n: int) -> np.ndarray:
    """Monotone ramp from 0 to 1 over ``n`` samples (Hann half-window)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
