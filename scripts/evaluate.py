"""Main experiment: baseline vs SONIC vs oracle on one split, with statistics and figures.

    python scripts/evaluate.py --split eval --config results/ablations/tuned.yaml [--clap]

Outputs in <results_dir>/<run-name>/<split>/:
    per_sample.csv, summary_overall.csv, summary_by_scenario.csv, stats.csv,
    summary.md, config.yaml, figures/*.png, audio/<sample>/<method>.wav
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from _common import base_parser, config_from_args, load_model

from sonic.audio import load_audio
from sonic.evaluation import aggregate, paired_test
from sonic.experiment import clap_similarity, evaluate_config, load_cached
from sonic.pipeline import METHODS
from sonic.plotting import plot_method_bars

QUALITY = ["sdr_bg", "sisdr_bg", "sisdri", "sdr_bg_out", "sdr_bg_in", "mse_out", "target_suppression_db"]
LOCALIZATION = ["loc_precision", "loc_recall", "loc_f1", "loc_iou", "onset_err_s", "offset_err_s", "event_recall"]
COST = ["latency_s", "latency_localization_s", "peak_gpu_mb", "masked_fraction", "edge_click_db"]
AUX = ["clap_sim_output", "clap_target_sim_drop"]
PAIRS = [("baseline", "sonic"), ("baseline", "oracle"), ("sonic", "oracle")]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def compact(summary: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """mean [ci_low, ci_high] strings for readable Markdown tables."""
    out = summary[keys].copy()
    for m in metrics:
        if f"{m}_mean" in summary:
            out[m] = [f"{a:.2f} [{b:.2f}, {c:.2f}]" if np.isfinite(a) else "n/a"
                      for a, b, c in zip(summary[f"{m}_mean"], summary[f"{m}_ci_low"], summary[f"{m}_ci_high"])]
    return out


def add_clap(df: pd.DataFrame, cfg, split: str, audio_dir: Path) -> pd.DataFrame:
    model = load_model(cfg)
    sims, drops = [], []
    for r in df.itertuples():
        y = load_audio(audio_dir / r.sample_id / f"{r.method}.wav", model.sr)
        s = clap_similarity(model, y, r.target_prompt)
        mix = load_cached(cfg, split, r.sample_id).timings.get("clap_sim_mixture", np.nan)
        sims.append(s)
        drops.append(mix - s)
    return df.assign(clap_sim_output=sims, clap_target_sim_drop=drops)


def main():
    p = base_parser(__doc__)
    p.add_argument("--split", default="eval", choices=["dev", "eval"])
    p.add_argument("--run-name", default="main")
    p.add_argument("--clap", action="store_true", help="also compute the auxiliary CLAP metric (loads the model)")
    args = p.parse_args()
    cfg = config_from_args(args)
    out = Path(cfg.paths.results_dir) / args.run_name / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    df = evaluate_config(cfg, args.split, methods=METHODS, output_dir=out / "audio")
    if args.clap:
        df = add_clap(df, cfg, args.split, out / "audio")
    df.to_csv(out / "per_sample.csv", index=False)

    metrics = [m for m in QUALITY + LOCALIZATION + COST + AUX if m in df]
    overall = aggregate(df, metrics, by=["method"])
    by_sc = aggregate(df, metrics, by=["scenario", "method"])
    overall.to_csv(out / "summary_overall.csv", index=False)
    by_sc.to_csv(out / "summary_by_scenario.csv", index=False)

    stats = pd.DataFrame([paired_test(df, m, a, b) for m in QUALITY + ["edge_click_db"] if m in df
                          for a, b in PAIRS])
    for scenario, g in df.groupby("scenario"):
        st = pd.DataFrame([paired_test(g, m, "baseline", "sonic") for m in ["sdr_bg", "sdr_bg_out", "target_suppression_db"]])
        st.insert(0, "scenario", scenario)
        stats = pd.concat([stats, st], ignore_index=True)
    stats.to_csv(out / "stats.csv", index=False)

    figs = out / "figures"
    plot_method_bars(by_sc, ["sdr_bg", "sdr_bg_out", "sdr_bg_in", "target_suppression_db"], figs / "quality_by_scenario.png")
    loc_sc = by_sc[by_sc.method.isin(["sonic"])]
    plot_method_bars(loc_sc, ["loc_precision", "loc_recall", "loc_iou"], figs / "localization_by_scenario.png")

    order = {m: i for i, m in enumerate(METHODS)}
    overall = overall.sort_values("method", key=lambda s: s.map(order))
    lines = [
        f"# SONIC results: split `{args.split}`, run `{args.run_name}`",
        "",
        f"Samples: {df.sample_id.nunique()} ({', '.join(f'{k}: {v}' for k, v in df.drop_duplicates('sample_id').scenario.value_counts().sort_index().items())}).",
        "Values: mean [95% bootstrap CI]. SI-SDR/suppression in dB; latency in s; memory in MB.",
        "",
        "## Reconstruction quality (overall)",
        md_table(compact(overall, ["method"], QUALITY)),
        "",
        "## Reconstruction quality by scenario",
        md_table(compact(by_sc, ["scenario", "method"], ["sdr_bg", "sdr_bg_out", "sdr_bg_in", "target_suppression_db"])),
        "",
        "## Localization (SONIC vs oracle)",
        md_table(compact(by_sc[by_sc.method != "baseline"], ["scenario", "method"], LOCALIZATION)),
        "",
        "## Cost",
        md_table(compact(overall, ["method"], [m for m in COST if m in df])),
        "",
        "## Paired Wilcoxon tests (difference = b - a)",
        md_table(stats.fillna({"scenario": "all"}) if "scenario" in stats else stats),
    ]
    if args.clap:
        lines += ["", "## Auxiliary CLAP similarity (not a fidelity measure)",
                  md_table(compact(overall, ["method"], AUX))]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
