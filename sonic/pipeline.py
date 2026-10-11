"""End-to-end editing: instruction -> separation -> (localization) -> masked removal.

The three methods share the same separated estimate and differ only in the
temporal mask applied to it:

* ``baseline``: mask == 1 everywhere (global removal)
* ``sonic``: mask from CLAP temporal localization
* ``oracle``: mask from ground-truth intervals (upper bound for localization)
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from sonic.config import SonicConfig
from sonic.editing import masked_remove
from sonic.instruction import parse_instruction
from sonic.localization import (
    Interval,
    LocalizationResult,
    intervals_to_mask,
    localize_from_scores,
    postprocess,
    soft_mask,
    window_scores,
)

METHODS = ("baseline", "sonic", "oracle")


@dataclass
class EditOutput:
    y: np.ndarray
    mask: np.ndarray
    intervals: list[Interval] | None  # None for the baseline
    localization: LocalizationResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    peak_gpu_mb: float = float("nan")


def oracle_intervals(gt: list[Interval], cfg: SonicConfig, duration_s: float) -> list[Interval]:
    """Ground-truth intervals with the same padding SONIC applies to its detections."""
    return postprocess(gt, min_event_s=0.0, min_gap_s=0.0, pad_s=cfg.localization.pad_s, duration_s=duration_s)


def build_mask(
    method: str,
    n: int,
    sr: int,
    cfg: SonicConfig,
    loc: LocalizationResult | None = None,
    gt_intervals: list[Interval] | None = None,
) -> tuple[np.ndarray, list[Interval] | None]:
    if method == "baseline":
        return np.ones(n), None
    if method == "sonic":
        if loc is None:
            raise ValueError("sonic needs a LocalizationResult")
        if cfg.editing.mask == "soft":
            m = soft_mask(loc.frame_times, loc.frame_scores, n, sr, loc.threshold, cfg.editing.soft_temperature)
            return m, loc.intervals
        if cfg.editing.mask != "hard":
            raise ValueError(f"unknown mask type {cfg.editing.mask!r}")
        return intervals_to_mask(loc.intervals, n, sr, cfg.editing.crossfade_s), loc.intervals
    if method == "oracle":
        if gt_intervals is None:
            raise ValueError("oracle needs ground-truth intervals")
        ints = oracle_intervals(gt_intervals, cfg, n / sr)
        return intervals_to_mask(ints, n, sr, cfg.editing.crossfade_s), ints
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def edit_with_estimate(
    x: np.ndarray,
    s_hat: np.ndarray,
    method: str,
    cfg: SonicConfig,
    sr: int,
    loc: LocalizationResult | None = None,
    gt_intervals: list[Interval] | None = None,
) -> EditOutput:
    """Apply one method's mask to an already-computed separation estimate."""
    mask, intervals = build_mask(method, len(x), sr, cfg, loc, gt_intervals)
    return EditOutput(y=masked_remove(x, s_hat, mask), mask=mask, intervals=intervals, localization=loc)


class _Timer:
    """Wall-clock stage timer; ``sync`` flushes queued GPU work before reading the clock."""

    def __init__(self, sync):
        self.sync = sync
        self.timings: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        self.timings[name] = time.perf_counter() - t0


def edit(
    x: np.ndarray,
    instruction: str,
    model,
    cfg: SonicConfig,
    method: str = "sonic",
    gt_intervals: list[Interval] | None = None,
) -> EditOutput:
    """Run the full pipeline on one clip with a real (or test) model.

    ``model`` must provide ``separate`` and, for ``sonic``, ``embed_text`` /
    ``embed_audio``. Optional ``synchronize``/``reset_peak_memory``/``peak_memory_mb``
    hooks are used for accurate GPU timing and memory accounting.
    """
    sr = getattr(model, "sr", 32000)
    sync = getattr(model, "synchronize", lambda: None)
    timer = _Timer(sync)
    if hasattr(model, "reset_peak_memory"):
        model.reset_peak_memory()

    with timer("parse"):
        target = parse_instruction(instruction).target
    with timer("separation"):
        s_hat = model.separate(x, target)
    loc = None
    if method == "sonic":
        p = cfg.localization
        with timer("localization"):
            ws = window_scores(x, sr, target, model, p.win_s, p.hop_s, p.background_prompts)
            loc = localize_from_scores(ws, len(x) / sr, p)
    with timer("editing"):
        out = edit_with_estimate(x, s_hat, method, cfg, sr, loc, gt_intervals)
    out.timings = dict(timer.timings, total=sum(timer.timings.values()))
    if hasattr(model, "peak_memory_mb"):
        out.peak_gpu_mb = model.peak_memory_mb()
    return out
