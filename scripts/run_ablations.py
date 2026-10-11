"""Ablations on the dev split (window size, threshold, crossfade, hard/soft mask).

    python scripts/run_ablations.py

Requires cached dev estimates (scripts/run_separation.py --split dev). Writes
<results_dir>/ablations/{ablations.csv, summary.csv, summary.md, tuned.yaml,
tuned_overrides.txt, figures/*.png}. ``tuned.yaml`` is then passed unchanged
to the eval-split run.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from _common import base_parser, config_from_args
from evaluate import compact, md_table

from sonic.ablation import run_ablations, select_best
from sonic.evaluation import aggregate
from sonic.plotting import plot_ablation

METRICS = ["sdr_bg", "sisdr_bg", "sdr_bg_out", "sdr_bg_in", "target_suppression_db", "loc_f1", "loc_iou", "edge_click_db"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--split", default="dev", choices=["dev"], help="ablations never use the eval split")
    args = p.parse_args()
    cfg = config_from_args(args)
    out = Path(cfg.paths.results_dir) / "ablations"
    out.mkdir(parents=True, exist_ok=True)

    df = run_ablations(cfg, args.split)
    df.to_csv(out / "ablations.csv", index=False)
    summary = aggregate(df, METRICS, by=["factor", "value"])
    summary.to_csv(out / "summary.csv", index=False)

    tuned, overrides = select_best(df, cfg)
    (out / "tuned.yaml").write_text(yaml.safe_dump(tuned.to_dict(), sort_keys=False))
    (out / "tuned_overrides.txt").write_text("\n".join(overrides) + "\n")

    for factor in ["win_s", "threshold", "crossfade_s", "mask"]:
        plot_ablation(df, factor, ["sdr_bg", "sdr_bg_out", "target_suppression_db", "loc_f1"],
                      out / "figures" / f"ablation_{factor}.png")

    md = ["# SONIC ablations (dev split)", "",
          "One factor varied at a time around the default config; SONIC method only.",
          "Reference rows = baseline / oracle with the default config.", "",
          md_table(compact(summary, ["factor", "value"], METRICS)), "",
          "## Selected (dev-best by sdr_bg, tie-break loc_f1)", "",
          *[f"- `{o}`" for o in overrides]]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
