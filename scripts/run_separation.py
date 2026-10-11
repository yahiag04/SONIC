"""Stage 1 (GPU): run AudioSep once per sample and cache CLAP window scores.

    python scripts/run_separation.py --split dev
    python scripts/run_separation.py --split eval

The cached estimate is shared by baseline, SONIC and oracle, which makes the
comparison fair by construction and lets ablations run without the GPU.
"""

from __future__ import annotations

from _common import base_parser, config_from_args, load_model

from sonic.ablation import ABLATION_WINDOWS, WINDOW_OVERLAP
from sonic.experiment import cache_split


def main():
    p = base_parser(__doc__)
    p.add_argument("--split", required=True, choices=["dev", "eval"])
    p.add_argument("--windows", nargs="+", type=float, default=list(ABLATION_WINDOWS),
                   help="window sizes (s) to cache scores for")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    model = load_model(cfg)
    print(f"AudioSep loaded on {model.device}")
    cache_split(model, cfg, args.split, args.windows, overlap=WINDOW_OVERLAP, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
