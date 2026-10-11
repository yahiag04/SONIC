"""Smoke-test the CPU-stage CLI scripts on a cache produced with a test double.

Only stage 2 (masking, evaluation, ablations, examples) is exercised; stage 1
(real AudioSep inference) can only run on a GPU runtime with the weights.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sonic.ablation import ABLATION_WINDOWS
from sonic.config import load_config
from sonic.dataset_builder import build_split, list_librispeech, load_esc50, split_pools
from sonic.experiment import cache_split
from tests.conftest import EnergyModel

ROOT = Path(__file__).resolve().parent.parent


def run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def workspace(fake_sources, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ws")
    overrides = [f"paths.data_root={tmp / 'data'}", f"paths.cache_dir={tmp / 'cache'}",
                 f"paths.results_dir={tmp / 'results'}"]
    cfg = load_config(overrides=overrides)
    pools = split_pools(load_esc50(fake_sources[0]), list_librispeech(fake_sources[1]))
    for split in ("dev", "eval"):
        build_split(pools[split], split, tmp / "data", n_per_scenario=1, cfg=cfg.dataset)
        cache_split(EnergyModel(), cfg, split, ABLATION_WINDOWS, log=lambda *_: None)
    return tmp, overrides


def test_ablations_then_eval_then_examples(workspace):
    tmp, ov = workspace
    run("run_ablations.py", *ov)
    tuned = tmp / "results" / "ablations" / "tuned.yaml"
    assert tuned.exists() and (tmp / "results" / "ablations" / "figures" / "ablation_threshold.png").exists()

    run("evaluate.py", "--split", "eval", "--config", tuned, *ov)
    out = tmp / "results" / "main" / "eval"
    df = pd.read_csv(out / "per_sample.csv")
    assert set(df.method) == {"baseline", "sonic", "oracle"} and len(df) == 9
    assert (out / "summary.md").read_text().startswith("# SONIC results")
    assert (out / "figures" / "quality_by_scenario.png").exists()

    run("run_baseline.py", "--split", "eval", *ov)
    run("run_sonic.py", "--split", "eval", "--oracle", *ov)
    assert (out / "metrics_oracle.csv").exists()

    run("make_examples.py", "--config", tuned, "--out", tmp / "examples", *ov)
    ex = sorted((tmp / "examples").glob("example*"))
    assert len(ex) == 3
    for d in ex:
        for f in ("mixture.wav", "reference.wav", "baseline.wav", "sonic.wav", "oracle.wav",
                  "localization.png", "metrics.json", "instruction.txt"):
            assert (d / f).exists(), d / f
