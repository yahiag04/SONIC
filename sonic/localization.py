"""Temporal localization of a text-described sound event (SONIC's contribution).

Pipeline:
  1. Slide overlapping windows over the clip and embed each with CLAP.
  2. Score each window as  cos(a, t_target) - max_k cos(a, t_background_k).
  3. Normalize scores per clip, spread them onto a fixed-rate frame grid by
     averaging all windows covering a frame.
  4. Threshold, median-smooth, fill short gaps, drop short events, pad.
  5. Convert intervals to a sample-level mask with raised-cosine crossfades.

The embedder is any object with ``embed_text(list[str])`` and
``embed_audio(list[np.ndarray])`` returning L2-normalized rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from scipy.ndimage import median_filter

from sonic.audio import frame_starts, raised_cosine

Interval = tuple[float, float]

DEFAULT_BACKGROUND_PROMPTS = (
    "ambient background noise",
    "a quiet room",
    "people talking",
    "rain and wind",
)


class Embedder(Protocol):
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_audio(self, windows: Sequence[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class LocalizationParams:
    win_s: float = 1.0
    hop_s: float = 0.25
    normalization: str = "zscore"  # zscore | minmax | none
    threshold: float = 0.5
    median_kernel_s: float = 0.25
    min_event_s: float = 0.1
    min_gap_s: float = 0.25
    pad_s: float = 0.1
    frame_rate: int = 100
    background_prompts: tuple[str, ...] = DEFAULT_BACKGROUND_PROMPTS


@dataclass
class WindowScores:
    starts_s: np.ndarray
    win_s: float
    target_sim: np.ndarray
    background_sim: np.ndarray  # max over background prompts
    scores: np.ndarray  # target_sim - background_sim


@dataclass
class LocalizationResult:
    intervals: list[Interval]
    frame_times: np.ndarray
    frame_scores: np.ndarray  # normalized, frame-level
    threshold: float
    window_scores: WindowScores | None = field(default=None, repr=False)


def window_starts(n: int, sr: int, win_s: float, hop_s: float) -> np.ndarray:
    return frame_starts(n, int(round(win_s * sr)), max(1, int(round(hop_s * sr))))


def window_scores(
    x: np.ndarray,
    sr: int,
    target: str,
    embedder: Embedder,
    win_s: float,
    hop_s: float,
    background_prompts: Sequence[str] = DEFAULT_BACKGROUND_PROMPTS,
    batch_size: int = 32,
) -> WindowScores:
    win = int(round(win_s * sr))
    starts = window_starts(len(x), sr, win_s, hop_s)
    text = np.asarray(embedder.embed_text([target, *background_prompts]), dtype=np.float64)
    t_target, t_bg = text[0], text[1:]

    sims_t, sims_b = [], []
    for b in range(0, len(starts), batch_size):
        batch = [x[s: s + win] for s in starts[b: b + batch_size]]
        a = np.asarray(embedder.embed_audio(batch), dtype=np.float64)
        sims_t.append(a @ t_target)
        sims_b.append((a @ t_bg.T).max(axis=1) if len(t_bg) else np.zeros(len(a)))
    target_sim = np.concatenate(sims_t)
    bg_sim = np.concatenate(sims_b)
    return WindowScores(
        starts_s=starts / sr,
        win_s=win_s,
        target_sim=target_sim,
        background_sim=bg_sim,
        scores=target_sim - bg_sim,
    )


def normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if method == "none":
        return scores
    if method == "zscore":
        std = scores.std()
        return np.zeros_like(scores) if std < 1e-12 else (scores - scores.mean()) / std
    if method == "minmax":
        span = scores.max() - scores.min()
        return np.zeros_like(scores) if span < 1e-12 else (scores - scores.min()) / span
    raise ValueError(f"unknown normalization {method!r}")


def frame_curve(
    starts_s: np.ndarray, scores: np.ndarray, win_s: float, duration_s: float, frame_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average the scores of every window covering each frame centre."""
    n_frames = max(1, int(round(duration_s * frame_rate)))
    times = (np.arange(n_frames) + 0.5) / frame_rate
    acc = np.zeros(n_frames)
    cnt = np.zeros(n_frames)
    for s, v in zip(starts_s, scores):
        covered = (times >= s) & (times < s + win_s)
        acc[covered] += v
        cnt[covered] += 1
    if np.any(cnt == 0):  # frames past the last window (very short clips)
        centers = starts_s + win_s / 2
        for i in np.flatnonzero(cnt == 0):
            acc[i] = scores[np.argmin(np.abs(centers - times[i]))]
            cnt[i] = 1
    return times, acc / cnt


def median_smooth(active: np.ndarray, kernel: int) -> np.ndarray:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel == 1:
        return active.astype(bool)
    return median_filter(active.astype(np.uint8), size=kernel, mode="nearest").astype(bool)


def active_to_intervals(active: np.ndarray, frame_rate: int) -> list[Interval]:
    padded = np.concatenate([[False], active.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(edges[i] / frame_rate, edges[i + 1] / frame_rate) for i in range(0, len(edges), 2)]


def _merge(intervals: list[Interval], max_gap: float = 0.0) -> list[Interval]:
    out: list[list[float]] = []
    for t0, t1 in sorted(intervals):
        if out and t0 - out[-1][1] <= max_gap + 1e-9:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def postprocess(
    intervals: list[Interval], min_event_s: float, min_gap_s: float, pad_s: float, duration_s: float
) -> list[Interval]:
    """Fill gaps < min_gap_s, drop events < min_event_s, pad, clip and merge."""
    merged = _merge(intervals, max_gap=min_gap_s)
    kept = [(a, b) for a, b in merged if b - a >= min_event_s - 1e-9]
    padded = [(max(0.0, a - pad_s), min(duration_s, b + pad_s)) for a, b in kept]
    return _merge(padded)


def intervals_to_mask(intervals: list[Interval], n: int, sr: int, crossfade_s: float) -> np.ndarray:
    """Mask equal to 1 inside intervals, with raised-cosine ramps placed outside them."""
    mask = np.zeros(n, dtype=np.float64)
    L = int(round(crossfade_s * sr))
    up = raised_cosine(L + 1)[:-1]  # 0 .. <1, length L
    down = up[::-1]
    for t0, t1 in intervals:
        i0, i1 = max(0, int(round(t0 * sr))), min(n, int(round(t1 * sr)))
        if i1 <= i0:
            continue
        mask[i0:i1] = 1.0
        if L:
            a = max(0, i0 - L)
            mask[a:i0] = np.maximum(mask[a:i0], up[L - (i0 - a):])
            b = min(n, i1 + L)
            mask[i1:b] = np.maximum(mask[i1:b], down[: b - i1])
    return mask


def soft_mask(
    times: np.ndarray, curve: np.ndarray, n: int, sr: int, threshold: float, temperature: float
) -> np.ndarray:
    """Sigmoid of the frame-level score, linearly interpolated to samples (ablation)."""
    frame_mask = 1.0 / (1.0 + np.exp(-(curve - threshold) / max(temperature, 1e-6)))
    t = np.arange(n) / sr
    return np.clip(np.interp(t, times, frame_mask), 0.0, 1.0)


def localize_from_scores(ws: WindowScores, duration_s: float, params: LocalizationParams) -> LocalizationResult:
    norm = normalize_scores(ws.scores, params.normalization)
    times, curve = frame_curve(ws.starts_s, norm, ws.win_s, duration_s, params.frame_rate)
    active = median_smooth(curve > params.threshold, round(params.median_kernel_s * params.frame_rate))
    intervals = postprocess(
        active_to_intervals(active, params.frame_rate),
        params.min_event_s, params.min_gap_s, params.pad_s, duration_s,
    )
    return LocalizationResult(intervals, times, curve, params.threshold, ws)


def localize(x: np.ndarray, sr: int, target: str, embedder: Embedder, params: LocalizationParams) -> LocalizationResult:
    ws = window_scores(x, sr, target, embedder, params.win_s, params.hop_s, params.background_prompts)
    return localize_from_scores(ws, len(x) / sr, params)
