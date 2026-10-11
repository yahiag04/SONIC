"""Experiment C: generative (diffusion) removal, global vs. temporally localized.

For each eval clip and each strength, AudioLDM2 removes the target by
regeneration under three masks, which reuse exactly the masks of the
subtractive experiment:

* ``gen_global``: no mask (whole clip regenerated)
* ``gen_sonic``: SONIC's CLAP-localized mask
* ``gen_oracle``: the ground-truth-interval mask

The subtractive conditions (baseline / sonic / oracle) are scored with the
same metrics, so generative and subtractive editing can be compared directly.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sonic.audio import SR, save_audio
from sonic.config import SonicConfig
from sonic.evaluation import evaluate_sample, region_mask
from sonic.experiment import apply_method, list_samples, load_cached

GEN_MASKS = {"gen_global": None, "gen_sonic": "sonic", "gen_oracle": "oracle"}
SUBTRACTIVE = ("baseline", "sonic", "oracle")


def sample_seed(sample_id: str, base: int) -> int:
    """Deterministic per-clip seed, identical across conditions (same initial noise)."""
    return (zlib.crc32(sample_id.encode()) + base) % (2**31)


def event_clap_similarity(embedder, audio: np.ndarray, intervals, sr: int, text: str) -> float:
    """CLAP similarity between the target text and the audio inside the event regions only."""
    seg = audio[region_mask(intervals, len(audio), sr)]
    if len(seg) < int(0.25 * sr):
        return float("nan")
    return float(embedder.embed_audio([seg])[0] @ embedder.embed_text([text])[0])


def run_generative(
    cfg: SonicConfig,
    split: str,
    editor,
    strengths=None,
    sample_ids: list[str] | None = None,
    output_dir: Path | None = None,
    log=print,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Run all generative conditions; returns per-sample metrics and the outputs (for CLAP scoring)."""
    strengths = tuple(strengths or cfg.generative.strengths)
    ids = sample_ids or list_samples(cfg.paths.data_root, split)
    rows, outputs = [], {}
    for k, sid in enumerate(ids):
        cs = load_cached(cfg, split, sid)
        smp = cs.sample
        masks = {name: (None if m is None else apply_method(cs, m, cfg).mask) for name, m in GEN_MASKS.items()}
        ints = {name: (None if m is None else apply_method(cs, m, cfg).intervals) for name, m in GEN_MASKS.items()}
        outputs[sid] = {}
        for s in strengths:
            for name, mask in masks.items():
                t0 = time.perf_counter()
                y = editor.edit(smp.x, SR, smp.meta["target_prompt"], s, sample_mask=mask,
                                seed=sample_seed(sid, cfg.generative.seed))
                dt = time.perf_counter() - t0
                method = f"{name}@{s:g}"
                r = {"sample_id": sid, "scenario": smp.meta["scenario"], "method": method, "family": "generative",
                     "strength": s, "target_prompt": smp.meta["target_prompt"], "latency_generation_s": dt}
                r.update(evaluate_sample(smp.x, y, smp.s, smp.b, smp.intervals, SR, pred_intervals=ints[name],
                                         mask=mask))
                r["masked_fraction"] = 1.0 if mask is None else float(np.mean(mask > 0))
                rows.append(r)
                outputs[sid][method] = y
                if output_dir is not None:
                    save_audio(Path(output_dir) / sid / f"{method}.wav", y, SR)
        for method in SUBTRACTIVE:
            out = apply_method(cs, method, cfg)
            r = {"sample_id": sid, "scenario": smp.meta["scenario"], "method": method, "family": "subtractive",
                 "strength": float("nan"), "target_prompt": smp.meta["target_prompt"]}
            r.update(evaluate_sample(smp.x, out.y, smp.s, smp.b, smp.intervals, SR, pred_intervals=out.intervals,
                                     mask=out.mask if method != "baseline" else None))
            r["masked_fraction"] = float(np.mean(out.mask > 0))
            rows.append(r)
            outputs[sid][method] = out.y
        outputs[sid]["mixture"] = smp.x
        log(f"[{k + 1}/{len(ids)}] {sid}: generative conditions done")
    return pd.DataFrame(rows), outputs


def add_event_clap(df: pd.DataFrame, outputs, embedder, cfg: SonicConfig, split: str) -> pd.DataFrame:
    """Add CLAP target similarity inside the true event regions, and its drop relative to the mixture."""
    sims, drops = [], []
    mix_cache: dict[str, float] = {}
    for r in df.itertuples():
        cs_meta = load_cached(cfg, split, r.sample_id).sample
        ints = cs_meta.intervals
        if r.sample_id not in mix_cache:
            mix_cache[r.sample_id] = event_clap_similarity(embedder, outputs[r.sample_id]["mixture"], ints, SR,
                                                           r.target_prompt)
        s = event_clap_similarity(embedder, outputs[r.sample_id][r.method], ints, SR, r.target_prompt)
        sims.append(s)
        drops.append(mix_cache[r.sample_id] - s)
    return df.assign(clap_event_target_sim=sims, clap_event_target_drop=drops)
