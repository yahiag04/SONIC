"""Build dataset splits on disk from ESC-50 and LibriSpeech.

Layout written:
    <data_root>/<split>/<sample_id>/{mixture,target,background}.wav
    <data_root>/<split>/<sample_id>/stems/{bed,speech,target}.wav
    <data_root>/<split>/<sample_id>/meta.json
    <data_root>/<split>/manifest.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sonic.audio import SR, load_audio, save_audio
from sonic.config import DatasetConfig
from sonic.dataset import (
    BED_CLASSES,
    SCENARIOS,
    concat_crossfade,
    make_instruction,
    synthesize_sample,
    trim_active,
)

SPLIT_FOLDS = {"dev": (1, 2, 3, 4), "eval": (5,)}
SPLIT_INDEX = {"dev": 0, "eval": 1}


@dataclass
class SourcePool:
    """Audio file pools for one split."""

    esc: pd.DataFrame  # columns: path, category, fold
    speech: dict[str, list[Path]]  # speaker id -> utterance files

    def files(self, category: str) -> list[Path]:
        return list(self.esc.loc[self.esc.category == category, "path"])


def load_esc50(esc_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(esc_root / "meta" / "esc50.csv")
    meta["path"] = [esc_root / "audio" / f for f in meta.filename]
    return meta[["path", "category", "fold"]]


def list_librispeech(libri_root: Path) -> dict[str, list[Path]]:
    """speaker -> sorted .flac files (LibriSpeech/dev-clean/<spk>/<chapter>/*.flac)."""
    speakers: dict[str, list[Path]] = {}
    for f in sorted(Path(libri_root).rglob("*.flac")):
        speakers.setdefault(f.parent.parent.name, []).append(f)
    return speakers


def split_pools(esc: pd.DataFrame, speakers: dict[str, list[Path]]) -> dict[str, SourcePool]:
    """ESC-50 folds 1-4 -> dev, 5 -> eval; speakers split 80/20 (disjoint, sorted)."""
    spk = sorted(speakers)
    cut = int(round(0.8 * len(spk)))
    spk_split = {"dev": spk[:cut], "eval": spk[cut:]}
    return {
        split: SourcePool(
            esc=esc[esc.fold.isin(folds)].reset_index(drop=True),
            speech={s: speakers[s] for s in spk_split[split]},
        )
        for split, folds in SPLIT_FOLDS.items()
    }


def _bed(pool: SourcePool, rng, n: int, sr: int, cache: dict) -> tuple[np.ndarray, str, list[str]]:
    cls = BED_CLASSES[int(rng.integers(len(BED_CLASSES)))]
    files = [f for f in pool.files(cls) if _active_len(f, sr, cache) >= 4.0]
    if not files:
        raise RuntimeError(f"no usable bed clips for class {cls}")
    picks = [files[int(i)] for i in rng.choice(len(files), size=min(3, len(files)), replace=False)]
    clips = [trim_active(_load(p, sr, cache), sr, top_db=40) for p in picks]
    return concat_crossfade(clips, n, sr), cls, [Path(p).name for p in picks]


def _load(path, sr, cache):
    key = str(path)
    if key not in cache:
        cache[key] = load_audio(path, sr)
    return cache[key]


def _active_len(path, sr, cache) -> float:
    return len(trim_active(_load(path, sr, cache), sr, top_db=40)) / sr


def _speech(pool: SourcePool, rng, n: int, sr: int) -> tuple[np.ndarray, str, list[str]]:
    spk = sorted(pool.speech)[int(rng.integers(len(pool.speech)))]
    utts = pool.speech[spk]
    start = int(rng.integers(len(utts)))
    out, used, k = [], [], 0
    while sum(len(u) for u in out) < n and k < len(utts):
        f = utts[(start + k) % len(utts)]
        out.append(load_audio(f, sr))
        used.append(f.name)
        k += 1
    sp = np.concatenate(out)
    if len(sp) < n:
        sp = np.pad(sp, (0, n - len(sp)))
    return sp[:n], spk, used


def build_split(
    pool: SourcePool,
    split: str,
    out_dir: Path,
    n_per_scenario: int,
    cfg: DatasetConfig,
    sr: int = SR,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    n = int(round(cfg.length_s * sr))
    cache: dict = {}
    rows = []
    for sc_idx, (sc_name, sc) in enumerate(SCENARIOS.items()):
        for i in range(n_per_scenario):
            rng = np.random.default_rng([cfg.seed, SPLIT_INDEX[split], sc_idx, i])
            target_cls = sc.targets[int(rng.integers(len(sc.targets)))]
            n_events = int(rng.integers(sc.n_events[0], sc.n_events[1] + 1))
            cand = [f for f in pool.files(target_cls) if _active_len(f, sr, cache) >= 0.2]
            if not cand:
                raise RuntimeError(f"no usable clips for target class {target_cls} in split {split}")
            idx = rng.choice(len(cand), size=n_events, replace=len(cand) < n_events)
            ev_files = [cand[int(j)] for j in idx]
            events = [trim_active(_load(f, sr, cache), sr)[: int(sc.max_event_s * sr)] for f in ev_files]
            bed, bed_cls, bed_files = _bed(pool, rng, n, sr, cache)
            speech, spk, sp_files = (None, None, [])
            sp_db = None
            if sc.with_speech:
                speech, spk, sp_files = _speech(pool, rng, n, sr)
                sp_db = float(rng.uniform(cfg.speech_to_bed_db_min, cfg.speech_to_bed_db_max))
            snr = float(rng.uniform(cfg.snr_db_min, cfg.snr_db_max))
            instruction, prompt = make_instruction(target_cls, rng)
            sample = synthesize_sample(
                bed, events, snr, rng, sr, speech=speech,
                speech_to_bed_db=sp_db if sp_db is not None else 0.0, length_s=cfg.length_s,
            )
            sid = f"{sc_name}_{i:03d}"
            d = out_dir / split / sid
            save_audio(d / "mixture.wav", sample["mixture"], sr)
            save_audio(d / "target.wav", sample["target"], sr)
            save_audio(d / "background.wav", sample["background"], sr)
            for k, v in sample["stems"].items():
                save_audio(d / "stems" / f"{k}.wav", v, sr)
            meta = {
                "sample_id": sid, "split": split, "scenario": sc_name,
                "instruction": instruction, "target_prompt": prompt, "target_class": target_cls,
                "bed_class": bed_cls, "intervals": sample["intervals"], "n_events": n_events,
                "snr_db": snr, "speech_to_bed_db": sp_db, "speaker": spk,
                "sources": {"events": [Path(f).name for f in ev_files], "bed": bed_files, "speech": sp_files},
                "peak_gain": sample["gain"], "sr": sr, "length_s": cfg.length_s, "seed": cfg.seed,
            }
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            rows.append({k: meta[k] for k in ("sample_id", "scenario", "instruction", "target_prompt",
                                              "target_class", "bed_class", "n_events", "snr_db")})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / split / "manifest.csv", index=False)
    return manifest
