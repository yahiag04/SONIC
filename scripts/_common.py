"""Shared CLI helpers for the scripts (config loading, overrides, model loading)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow running scripts without `pip install -e .`
    sys.path.insert(0, str(ROOT))

from sonic.config import SonicConfig, load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. localization.threshold=0.3")
    return p


def config_from_args(args) -> SonicConfig:
    return load_config(args.config, args.overrides)


def load_model(cfg: SonicConfig):
    from sonic.separation import AudioSepModel

    return AudioSepModel(cfg.paths.models_dir, device=cfg.separation.device, use_chunk=cfg.separation.use_chunk)


def add_single_file_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=["dev", "eval"], help="run on a cached dataset split")
    p.add_argument("--audio", help="single input file (end-to-end, loads the model)")
    p.add_argument("--instruction", help="editing instruction for --audio")
    p.add_argument("--out", default=None, help="output directory (default: <results_dir>/<run-name>)")
    p.add_argument("--run-name", default="main")


def run_single_file(args, cfg: SonicConfig, method: str) -> None:
    """End-to-end edit of one file with the real model; writes audio, intervals, timings, plot."""
    import json

    from sonic.audio import load_audio, save_audio
    from sonic.pipeline import edit
    from sonic.plotting import plot_localization

    if not args.instruction:
        raise SystemExit("--instruction is required with --audio")
    model = load_model(cfg)
    x = load_audio(args.audio, model.sr)
    out = edit(x, args.instruction, model, cfg, method=method)
    out_dir = Path(args.out or Path(cfg.paths.results_dir) / "single") / Path(args.audio).stem
    save_audio(out_dir / f"{method}.wav", out.y, model.sr)
    info = {"instruction": args.instruction, "method": method, "intervals": out.intervals,
            "timings_s": out.timings, "peak_gpu_mb": out.peak_gpu_mb}
    (out_dir / f"{method}.json").write_text(json.dumps(info, indent=2))
    if out.localization is not None:
        loc = out.localization
        plot_localization(x, model.sr, loc.frame_times, loc.frame_scores, loc.threshold,
                          out.intervals, [], out.mask, args.instruction, out_dir / "localization.png")
    print(json.dumps(info, indent=2))
    print(f"wrote {out_dir}")


def run_split(args, cfg: SonicConfig, methods: list[str]) -> None:
    """Apply methods to the cached estimates of a split; write audio and per-sample metrics."""
    from sonic.experiment import evaluate_config

    out_dir = Path(args.out or Path(cfg.paths.results_dir) / args.run_name) / args.split
    df = evaluate_config(cfg, args.split, methods=methods, output_dir=out_dir / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in methods:
        df[df.method == m].to_csv(out_dir / f"metrics_{m}.csv", index=False)
    cols = ["sdr_bg", "sdr_bg_out", "target_suppression_db"]
    print(df.groupby("method")[cols].mean().round(2).to_string())
    print(f"wrote {out_dir}")
