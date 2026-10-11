"""Experiment A: global removal baseline (y = x - AudioSep estimate).

    python scripts/run_baseline.py --split eval            # uses cached estimates
    python scripts/run_baseline.py --audio in.wav --instruction "Remove the dog barking"
"""

from __future__ import annotations

from _common import add_single_file_args, base_parser, config_from_args, run_single_file, run_split


def main():
    p = base_parser(__doc__)
    add_single_file_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.audio:
        run_single_file(args, cfg, "baseline")
    elif args.split:
        run_split(args, cfg, ["baseline"])
    else:
        p.error("give --split or --audio")


if __name__ == "__main__":
    main()
