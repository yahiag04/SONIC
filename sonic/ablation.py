"""One-factor-at-a-time ablations of SONIC's localization/editing parameters.

Run on the dev split only. The best value per factor (by mean ``sisdr_bg`` of
the SONIC output, tie-break ``loc_f1``) is combined into a tuned config that is
then used, unchanged, on the eval split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sonic.config import SonicConfig, replace_section
from sonic.experiment import evaluate_config, list_samples, load_cached

ABLATION_WINDOWS = (0.5, 1.0, 2.0)
ABLATION_THRESHOLDS = (-0.5, 0.0, 0.5, 1.0)
ABLATION_CROSSFADES = (0.0, 0.01, 0.05, 0.2)
ABLATION_MASKS = ("hard", "soft")
WINDOW_OVERLAP = 0.75


@dataclass(frozen=True)
class Variant:
    factor: str
    value: object
    cfg: SonicConfig


def variants(cfg: SonicConfig) -> list[Variant]:
    out = []
    for w in ABLATION_WINDOWS:
        hop = cfg.localization.hop_s if w == cfg.localization.win_s else w * (1 - WINDOW_OVERLAP)
        out.append(Variant("win_s", w, replace_section(cfg, "localization", win_s=w, hop_s=hop)))
    for t in ABLATION_THRESHOLDS:
        out.append(Variant("threshold", t, replace_section(cfg, "localization", threshold=t)))
    for c in ABLATION_CROSSFADES:
        out.append(Variant("crossfade_s", c, replace_section(cfg, "editing", crossfade_s=c)))
    for m in ABLATION_MASKS:
        out.append(Variant("mask", m, replace_section(cfg, "editing", mask=m)))
    return out


def run_ablations(cfg: SonicConfig, split: str = "dev", log=print) -> pd.DataFrame:
    ids = list_samples(cfg.paths.data_root, split)
    cached = {sid: load_cached(cfg, split, sid) for sid in ids}
    frames = []
    for v in variants(cfg):
        df = evaluate_config(v.cfg, split, methods=["sonic"], sample_ids=ids, cached=cached)
        df.insert(0, "factor", v.factor)
        df.insert(1, "value", str(v.value))
        frames.append(df)
        log(f"{v.factor}={v.value}: sisdr_bg={df.sisdr_bg.mean():.2f} dB, loc_f1={df.loc_f1.mean():.3f}")
    # Reference rows: baseline and oracle with the default config.
    ref = evaluate_config(cfg, split, methods=["baseline", "oracle"], sample_ids=ids, cached=cached)
    ref.insert(0, "factor", "reference")
    ref.insert(1, "value", ref.method)
    frames.append(ref)
    return pd.concat(frames, ignore_index=True)


_SECTION_OF = {"win_s": "localization", "threshold": "localization",
               "crossfade_s": "editing", "mask": "editing"}


def select_best(df: pd.DataFrame, cfg: SonicConfig) -> tuple[SonicConfig, list[str]]:
    """Pick the dev-best value per factor and return the tuned config + override strings."""
    tuned = cfg
    overrides = []
    for factor, section in _SECTION_OF.items():
        g = df[df.factor == factor].groupby("value")[["sisdr_bg", "loc_f1"]].mean()
        if g.empty:
            continue
        best = g.sort_values(["sisdr_bg", "loc_f1"], ascending=False).index[0]
        value = best if factor == "mask" else float(best)
        changes = {factor: value}
        if factor == "win_s":
            changes["hop_s"] = cfg.localization.hop_s if np.isclose(value, cfg.localization.win_s) \
                else value * (1 - WINDOW_OVERLAP)
            overrides.append(f"localization.hop_s={changes['hop_s']}")
        tuned = replace_section(tuned, section, **changes)
        overrides.append(f"{section}.{factor}={value}")
    return tuned, overrides
