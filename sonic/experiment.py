"""Experiment orchestration shared by the CLI scripts.

Stage 1 (GPU, ``cache_split``): for every sample, run AudioSep once and compute
CLAP window scores for each window size; store estimate, scores, timings.
Stage 2 (CPU, ``run_methods`` / ``evaluate_config``): apply baseline / SONIC /
oracle masks to the cached estimate and score them against the references.
All methods therefore use the identical separation output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sonic.audio import SR, load_audio, save_audio
from sonic.config import SonicConfig
from sonic.evaluation import evaluate_sample
from sonic.localization import WindowScores, localize_from_scores, window_scores
from sonic.pipeline import METHODS, edit_with_estimate


@dataclass
class Sample:
    sample_id: str
    meta: dict
    x: np.ndarray
    s: np.ndarray
    b: np.ndarray

    @property
    def intervals(self) -> list[tuple[float, float]]:
        return [tuple(i) for i in self.meta["intervals"]]


def list_samples(data_root: Path, split: str) -> list[str]:
    manifest = Path(data_root) / split / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing; run scripts/prepare_dataset.py first")
    return list(pd.read_csv(manifest).sample_id)


def load_sample(data_root: Path, split: str, sample_id: str, sr: int = SR) -> Sample:
    d = Path(data_root) / split / sample_id
    meta = json.loads((d / "meta.json").read_text())
    return Sample(sample_id, meta, load_audio(d / "mixture.wav", sr),
                  load_audio(d / "target.wav", sr), load_audio(d / "background.wav", sr))


# ---------------------------------------------------------------- caching --

def _win_key(win_s: float) -> str:
    return f"w{win_s:g}"


def cache_dir_for(cache_root: Path, split: str, sample_id: str) -> Path:
    return Path(cache_root) / split / sample_id


def save_window_scores(path: Path, ws: WindowScores) -> None:
    np.savez(path, starts_s=ws.starts_s, win_s=ws.win_s, target_sim=ws.target_sim,
             background_sim=ws.background_sim, scores=ws.scores)


def load_window_scores(path: Path) -> WindowScores:
    z = np.load(path)
    return WindowScores(z["starts_s"], float(z["win_s"]), z["target_sim"], z["background_sim"], z["scores"])


def clap_similarity(model, audio: np.ndarray, text: str) -> float:
    """Cosine similarity between CLAP embeddings of a clip and a text (auxiliary metric only)."""
    return float(model.embed_audio([audio])[0] @ model.embed_text([text])[0])


def cache_split(
    model,
    cfg: SonicConfig,
    split: str,
    window_sizes: Iterable[float],
    overlap: float = 0.75,
    sample_ids: list[str] | None = None,
    overwrite: bool = False,
    log=print,
) -> None:
    """Run separation + CLAP window scoring once per sample and cache the results."""
    sr = getattr(model, "sr", SR)
    window_sizes = sorted(set(float(w) for w in window_sizes) | {cfg.localization.win_s})
    ids = sample_ids or list_samples(cfg.paths.data_root, split)
    sync = getattr(model, "synchronize", lambda: None)
    warmed = False
    for k, sid in enumerate(ids):
        out = cache_dir_for(cfg.paths.cache_dir, split, sid)
        done = out / "timings.json"
        if done.exists() and not overwrite:
            continue
        out.mkdir(parents=True, exist_ok=True)
        smp = load_sample(cfg.paths.data_root, split, sid, sr)
        prompt = smp.meta["target_prompt"]
        if not warmed:  # exclude one-off CUDA/cuDNN initialisation from timings
            model.separate(smp.x, prompt)
            warmed = True
        timings: dict[str, float] = {}
        if hasattr(model, "reset_peak_memory"):
            model.reset_peak_memory()
        sync()
        t0 = time.perf_counter()
        s_hat = model.separate(smp.x, prompt)
        sync()
        timings["separation_s"] = time.perf_counter() - t0
        if hasattr(model, "peak_memory_mb"):
            timings["peak_gpu_mb_separation"] = model.peak_memory_mb()
        save_audio(out / "s_hat.wav", s_hat, sr)

        for w in window_sizes:
            hop = w * (1 - overlap) if w != cfg.localization.win_s else cfg.localization.hop_s
            if hasattr(model, "reset_peak_memory"):
                model.reset_peak_memory()
            sync()
            t0 = time.perf_counter()
            ws = window_scores(smp.x, sr, prompt, model, w, hop, cfg.localization.background_prompts)
            sync()
            timings[f"localization_{_win_key(w)}_s"] = time.perf_counter() - t0
            if hasattr(model, "peak_memory_mb"):
                timings[f"peak_gpu_mb_localization_{_win_key(w)}"] = model.peak_memory_mb()
            save_window_scores(out / f"scores_{_win_key(w)}.npz", ws)

        # Auxiliary CLAP similarity of the unedited mixture to the target text.
        if hasattr(model, "embed_audio"):
            timings["clap_sim_mixture"] = clap_similarity(model, smp.x, prompt)
        done.write_text(json.dumps(timings, indent=2))
        log(f"[{k + 1}/{len(ids)}] cached {sid} (separation {timings['separation_s']:.2f}s)")


@dataclass
class CachedSample:
    sample: Sample
    s_hat: np.ndarray
    timings: dict
    scores: dict[str, WindowScores]


def load_cached(cfg: SonicConfig, split: str, sample_id: str, sr: int = SR) -> CachedSample:
    smp = load_sample(cfg.paths.data_root, split, sample_id, sr)
    d = cache_dir_for(cfg.paths.cache_dir, split, sample_id)
    if not (d / "timings.json").exists():
        raise FileNotFoundError(f"no cache for {split}/{sample_id}; run scripts/run_separation.py")
    scores = {p.stem.removeprefix("scores_"): load_window_scores(p) for p in sorted(d.glob("scores_*.npz"))}
    return CachedSample(smp, load_audio(d / "s_hat.wav", sr), json.loads((d / "timings.json").read_text()), scores)


# ------------------------------------------------------------- evaluation --

def apply_method(cs: CachedSample, method: str, cfg: SonicConfig, sr: int = SR):
    loc = None
    if method == "sonic":
        key = _win_key(cfg.localization.win_s)
        if key not in cs.scores:
            raise KeyError(f"window size {cfg.localization.win_s}s not cached for {cs.sample.sample_id}")
        loc = localize_from_scores(cs.scores[key], len(cs.sample.x) / sr, cfg.localization)
    return edit_with_estimate(cs.sample.x, cs.s_hat, method, cfg, sr, loc, cs.sample.intervals)


def method_cost(timings: dict, method: str, win_s: float) -> dict[str, float]:
    sep = timings.get("separation_s", float("nan"))
    peak_sep = timings.get("peak_gpu_mb_separation", float("nan"))
    if method == "sonic":
        loc_t = timings.get(f"localization_{_win_key(win_s)}_s", float("nan"))
        peak_loc = timings.get(f"peak_gpu_mb_localization_{_win_key(win_s)}", float("nan"))
        return {"latency_s": sep + loc_t, "latency_localization_s": loc_t,
                "peak_gpu_mb": float(np.nanmax([peak_sep, peak_loc])) if np.isfinite([peak_sep, peak_loc]).any() else float("nan")}
    return {"latency_s": sep, "latency_localization_s": 0.0, "peak_gpu_mb": peak_sep}


def evaluate_config(
    cfg: SonicConfig,
    split: str,
    methods: Iterable[str] = METHODS,
    sample_ids: list[str] | None = None,
    output_dir: Path | None = None,
    sr: int = SR,
    cached: dict[str, CachedSample] | None = None,
) -> pd.DataFrame:
    """Score each method on each cached sample; optionally write edited audio."""
    ids = sample_ids or list_samples(cfg.paths.data_root, split)
    rows = []
    for sid in ids:
        cs = cached[sid] if cached is not None and sid in cached else load_cached(cfg, split, sid, sr)
        smp = cs.sample
        for method in methods:
            out = apply_method(cs, method, cfg, sr)
            r = {"sample_id": sid, "scenario": smp.meta["scenario"], "method": method,
                 "target_prompt": smp.meta["target_prompt"], "snr_db": smp.meta["snr_db"],
                 "n_events": smp.meta["n_events"]}
            r.update(evaluate_sample(smp.x, out.y, smp.s, smp.b, smp.intervals, sr,
                                     pred_intervals=out.intervals,
                                     mask=out.mask if method != "baseline" else None))
            r.update(method_cost(cs.timings, method, cfg.localization.win_s))
            r["n_pred_intervals"] = len(out.intervals) if out.intervals is not None else float("nan")
            r["masked_fraction"] = float(np.mean(out.mask > 0))
            rows.append(r)
            if output_dir is not None:
                d = Path(output_dir) / sid
                save_audio(d / f"{method}.wav", out.y, sr)
                if out.intervals is not None:
                    (d / f"{method}_intervals.json").write_text(json.dumps(out.intervals))
    return pd.DataFrame(rows)
