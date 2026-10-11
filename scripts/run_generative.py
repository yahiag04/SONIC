"""Experiment C: diffusion-based removal (AudioLDM2), global vs. SONIC-localized vs. oracle.

    python scripts/run_generative.py --split eval --config results/ablations/tuned.yaml [--limit 10]
    python scripts/run_generative.py --check            # front-end sanity check only

Requires cached eval estimates (scripts/run_separation.py --split eval), which
provide the SONIC masks. Writes <results_dir>/generative/<split>/.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, config_from_args, load_model
from evaluate import compact, environment, md_table

from sonic.audio import SR, save_audio
from sonic.evaluation import aggregate, log_spectral_distance, paired_test, sdr
from sonic.experiment import list_samples, load_sample
from sonic.generative import AudioLDM2Editor
from sonic.generative_experiment import add_event_clap, run_generative

METRICS = ["sdr_bg", "sdr_bg_out", "sdr_bg_in", "target_suppression_db", "lsd_out", "lsd_in",
           "clap_event_target_sim", "clap_event_target_drop", "latency_generation_s", "masked_fraction"]


def check_front_end(editor, cfg, split, out: Path, n: int = 3) -> dict:
    """Vocoder and VAE round trips on real clips: a wrong mel front end shows up as a large LSD."""
    from sonic.audio import resample

    rows = []
    for sid in list_samples(cfg.paths.data_root, split)[:n]:
        x16 = resample(load_sample(cfg.paths.data_root, split, sid).x, SR, editor.sr)
        mel = editor.log_mel(x16)
        voc = editor.vocode(mel)[: len(x16)]
        full16 = editor.roundtrip(x16)
        save_audio(out / "check" / f"{sid}_vocoder.wav", voc, editor.sr)
        save_audio(out / "check" / f"{sid}_vae_vocoder.wav", full16, editor.sr)
        rows.append({"sample_id": sid,
                     "lsd_vocoder_db": log_spectral_distance(voc, x16[: len(voc)], editor.sr),
                     "lsd_vae_vocoder_db": log_spectral_distance(full16, x16[: len(full16)], editor.sr),
                     "sdr_vocoder_db": sdr(voc, x16[: len(voc)])})
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    return df.mean(numeric_only=True).to_dict()


def main():
    p = base_parser(__doc__)
    p.add_argument("--split", default="eval", choices=["dev", "eval"])
    p.add_argument("--limit", type=int, default=None, help="only the first N clips per scenario")
    p.add_argument("--check", action="store_true", help="run the front-end sanity check and exit")
    p.add_argument("--save-audio", action="store_true", help="write every generated clip (large)")
    args = p.parse_args()
    cfg = config_from_args(args)
    out = Path(cfg.paths.results_dir) / "generative" / args.split
    out.mkdir(parents=True, exist_ok=True)

    editor = AudioLDM2Editor(cfg.generative, device=cfg.separation.device)
    print(f"AudioLDM2 loaded on {editor.device} ({editor.dtype})")
    check = check_front_end(editor, cfg, args.split, out)
    (out / "front_end_check.json").write_text(json.dumps(check, indent=2))
    if args.check:
        return

    ids = list_samples(cfg.paths.data_root, args.split)
    if args.limit:
        ids = [i for i in ids if int(i.rsplit("_", 1)[1]) < args.limit]
    df, outputs = run_generative(cfg, args.split, editor, sample_ids=ids,
                                 output_dir=out / "audio" if args.save_audio else None)
    for sid, outs in outputs.items():  # keep the demo clips (first clip of each scenario) for listening
        if sid.endswith("_000"):
            for method, y in outs.items():
                save_audio(out / "examples" / sid / f"{method}.wav", y, SR)
    del editor
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    model = load_model(cfg)  # AudioSep's CLAP, independent of AudioLDM2's own CLAP
    df = add_event_clap(df, outputs, model, cfg, args.split)
    df.to_csv(out / "per_sample.csv", index=False)
    (out / "environment.json").write_text(json.dumps(environment(), indent=2))

    metrics = [m for m in METRICS if m in df]
    overall = aggregate(df, metrics, by=["method"])
    by_sc = aggregate(df, metrics, by=["scenario", "method"])
    overall.to_csv(out / "summary_overall.csv", index=False)
    by_sc.to_csv(out / "summary_by_scenario.csv", index=False)

    pairs = []
    for s in cfg.generative.strengths:
        g = lambda n: f"{n}@{s:g}"
        pairs += [(g("gen_global"), g("gen_sonic")), (g("gen_global"), g("gen_oracle")),
                  (g("gen_sonic"), g("gen_oracle")), ("sonic", g("gen_sonic")), ("oracle", g("gen_oracle"))]
    stats = pd.DataFrame([paired_test(df, m, a, b) for a, b in pairs
                          for m in ["sdr_bg", "sdr_bg_out", "lsd_out", "lsd_in", "target_suppression_db",
                                    "clap_event_target_drop"]])
    stats.to_csv(out / "stats.csv", index=False)

    order = ["baseline", "sonic", "oracle"] + [f"{n}@{s:g}" for s in cfg.generative.strengths
                                               for n in ("gen_global", "gen_sonic", "gen_oracle")]
    overall = overall.set_index("method").reindex([o for o in order if o in set(overall.method)]).reset_index()
    md = [f"# Experiment C: generative removal (split `{args.split}`)", "",
          f"Clips: {df.sample_id.nunique()}. Model: `{cfg.generative.model_id}` (pretrained, unchanged), "
          f"{cfg.generative.steps} DDIM steps, guidance {cfg.generative.guidance_scale}, "
          f"negative prompt = target, prompt = \"{cfg.generative.prompt}\".",
          "`gen_*@s`: regenerate a fraction s of the diffusion schedule; masks are the same as in the subtractive experiment.",
          "LSD = log-spectral distance to the reference background (dB, lower is better, phase-insensitive).",
          "`clap_event_target_drop`: decrease of CLAP similarity to the target text inside true event regions (higher = more removed).",
          "", "## Front-end check (vocoder / VAE round trip, LSD dB)", "",
          md_table(pd.DataFrame([check])), "",
          "## Overall", "", md_table(compact(overall, ["method"], metrics)), "",
          "## By scenario", "",
          md_table(compact(by_sc, ["scenario", "method"], ["sdr_bg_out", "lsd_out", "lsd_in", "clap_event_target_drop"])),
          "", "## Paired Wilcoxon tests (difference = b - a)", "", md_table(stats)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:20]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
