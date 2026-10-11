"""Experiment B: SONIC temporally localized removal (optionally with the oracle mask).

    python scripts/run_sonic.py --split eval [--oracle] [--config results/ablations/tuned.yaml]
    python scripts/run_sonic.py --audio in.wav --instruction "Remove the dog barking"
"""

from __future__ import annotations

from _common import add_single_file_args, base_parser, config_from_args, run_single_file, run_split


def main():
    p = base_parser(__doc__)
    add_single_file_args(p)
    p.add_argument("--oracle", action="store_true", help="also run the ground-truth-mask upper bound")
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.audio:
        run_single_file(args, cfg, "sonic")
    elif args.split:
        run_split(args, cfg, ["sonic", "oracle"] if args.oracle else ["sonic"])
    else:
        p.error("give --split or --audio")


if __name__ == "__main__":
    main()
