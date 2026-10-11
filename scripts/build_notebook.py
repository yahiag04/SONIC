"""Generate notebooks/sonic_colab.ipynb (kept as code so the notebook stays reviewable).

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# SONIC: Semantic and Temporal Control for Audio Editing

Instruction-guided **sound removal** with explicit **temporal localization**.

This notebook reproduces every result in the repository from a clean Colab runtime:

1. clone the repo and install dependencies
2. check the GPU
3. download the pretrained AudioSep + CLAP weights (third-party, MIT / CC0)
4. build the synthetic dataset from ESC-50 and LibriSpeech
5. **stage 1** (GPU): one AudioSep pass per clip plus CLAP window scores, cached
6. ablations on the **dev** split → tuned config
7. main comparison on the held-out **eval** split: baseline vs SONIC vs oracle
8. demo examples with audio players and localization plots
9. save everything

**Runtime:** select *Runtime → Change runtime type → T4 GPU*. A full run takes roughly
an hour (dominated by dataset download and stage 1); set `QUICK = True` for a ~10 min smoke test.
Numbers from a `QUICK` run are **not** the reported results."""),
    md("## 0. Configuration\nEdit only this cell. All paths are configurable, and nothing personal is hardcoded."),
    code("""REPO_URL = "https://github.com/yahiag04/SONIC.git"
BRANCH = "main"
WORK_DIR = "/content/SONIC"          # repository checkout
DATA_DIR = "/content/sonic_data"     # raw corpora + generated mixtures
MODELS_DIR = "/content/sonic_models" # AudioSep code + checkpoints (never committed)
RESULTS_DIR = "/content/sonic_results" # kept outside the checkout
USE_DRIVE = False                    # True: keep data/models/results on Google Drive
DRIVE_DIR = "/content/drive/MyDrive/sonic"
QUICK = False                        # True: tiny dataset for a smoke test only"""),
    code("""import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR, MODELS_DIR, RESULTS_DIR = (f"{DRIVE_DIR}/data", f"{DRIVE_DIR}/models", f"{DRIVE_DIR}/results")
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

OV = [f"paths.data_root={DATA_DIR}", f"paths.models_dir={MODELS_DIR}",
      f"paths.results_dir={RESULTS_DIR}", f"paths.cache_dir={RESULTS_DIR}/cache"]
if QUICK:
    OV += ["dataset.n_eval_per_scenario=3", "dataset.n_dev_per_scenario=2"]
OVS = " ".join(OV)
print(OVS)

import subprocess
def sh(cmd):
    # run a shell command, stream its output, and stop "Run all" if it fails
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"command failed (exit {p.returncode}): {cmd}")"""),
    md("## 1. Clone and install"),
    code("""%cd /content
if os.path.isdir(f"{WORK_DIR}/.git"):
    !git -C {WORK_DIR} pull --ff-only
else:
    !rm -rf {WORK_DIR}
    !git clone -b {BRANCH} {REPO_URL} {WORK_DIR}
assert os.path.isfile(f"{WORK_DIR}/sonic/__init__.py"), "clone failed: see the git output above"
%cd {WORK_DIR}
!git log -1 --oneline"""),
    code("""!pip install -q -r requirements-colab.txt
!pip install -q -e .
import sys
sys.path.insert(0, WORK_DIR)  # make `sonic` importable in this kernel without a restart
sh("python -m pytest -q tests")  # unit tests (test doubles only, no model)"""),
    md("## 2. GPU check"),
    code("""import torch, subprocess
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"{p.total_memory / 2**30:.1f} GB")
else:
    print("WARNING: no GPU. Stage 1 will be very slow on CPU.")
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true"""),
    md("## 3. Download pretrained models\nAudioSep (Liu et al., 2023) is cloned at a pinned commit; checkpoints come from the official HF repo `audo/AudioSep`."),
    code('sh(f"python scripts/fetch_models.py {OVS}")'),
    md("### Model sanity check\nLoads AudioSep once and separates an example file from the official AudioSep demo Space. If this cell fails, nothing downstream is valid, and there is no silent fallback."),
    code("""import numpy as np
from sonic.config import load_config
from sonic.audio import load_audio
from sonic.separation import AudioSepModel
cfg = load_config(overrides=OV)
model = AudioSepModel(cfg.paths.models_dir, device=cfg.separation.device)
print("loaded on", model.device)
from huggingface_hub import hf_hub_download
ex_path = hf_hub_download("Audio-AGI/AudioSep", "examples/acoustic_guitar.wav", repo_type="space",
                          local_dir=f"{MODELS_DIR}/space_examples")
ex = load_audio(ex_path)[: 10 * 32000]
est = model.separate(ex, "acoustic guitar")
print("input rms %.4f | estimate rms %.4f | peak GPU %.0f MB" % (
    np.sqrt(np.mean(ex**2)), np.sqrt(np.mean(est**2)), model.peak_memory_mb()))
sims = model.embed_audio([ex]) @ model.embed_text(["acoustic guitar", "dog barking"]).T
print("CLAP similarity  guitar: %.3f   dog: %.3f" % tuple(sims[0]))
del model; torch.cuda.empty_cache()"""),
    md("## 4. Build the dataset\nESC-50 (CC BY-NC 3.0) folds 1-4 → **dev**, fold 5 → **eval**; LibriSpeech dev-clean (CC BY 4.0) speakers split 80/20. Mixtures are deterministic given the seed."),
    code('sh(f"python scripts/prepare_dataset.py {OVS}")'),
    md("## 5. Stage 1: separation + CLAP window scores (GPU)\nOne AudioSep pass per clip, shared by all methods, plus CLAP scores for window sizes 0.5/1/2 s. Re-running skips cached samples."),
    code('sh(f"python scripts/run_separation.py --split dev {OVS}")'),
    code('sh(f"python scripts/run_separation.py --split eval {OVS}")'),
    md("## 6. Ablations on the dev split → tuned config"),
    code("""sh(f"python scripts/run_ablations.py {OVS}")
from IPython.display import Markdown, Image, display, Audio
display(Markdown(open(f"{RESULTS_DIR}/ablations/summary.md").read()))
for f in ["win_s", "threshold", "crossfade_s", "mask"]:
    display(Image(f"{RESULTS_DIR}/ablations/figures/ablation_{f}.png"))"""),
    md("## 7. Main experiment on the held-out eval split\nThe tuned config from step 6 is used unchanged. Baseline, SONIC and oracle share the same AudioSep estimate."),
    code("""TUNED = f"{RESULTS_DIR}/ablations/tuned.yaml"
sh(f"python scripts/evaluate.py --split eval --config {TUNED} --clap {OVS}")
display(Markdown(open(f"{RESULTS_DIR}/main/eval/summary.md").read()))
display(Image(f"{RESULTS_DIR}/main/eval/figures/quality_by_scenario.png"))
display(Image(f"{RESULTS_DIR}/main/eval/figures/localization_by_scenario.png"))"""),
    md("## 8. Demo examples\nThe first eval sample of each scenario (not hand-picked): original, reference (ground-truth target-free audio), baseline, SONIC, oracle."),
    code("""EX = f"{RESULTS_DIR}/examples"
sh(f"python scripts/make_examples.py --config {TUNED} --out {EX} {OVS}")
import glob, json
for d in sorted(glob.glob(f"{EX}/example*")):
    meta = json.load(open(f"{d}/metrics.json"))
    display(Markdown(f"### {os.path.basename(d)}\\n**Instruction:** {meta['instruction']}"))
    for name in ["mixture", "reference", "baseline", "sonic", "oracle"]:
        print(name); display(Audio(f"{d}/{name}.wav"))
    display(Image(f"{d}/localization.png"))
    rows = {m: {k: round(v, 2) for k, v in meta["metrics"][m].items() if k in
                ("sdr_bg", "sdr_bg_out", "sdr_bg_in", "target_suppression_db", "loc_iou")} for m in meta["metrics"]}
    display(Markdown("```\\n" + json.dumps(rows, indent=1) + "\\n```"))"""),
    md("### Try your own recording (optional)\nUpload a file and write an instruction such as *Remove the dog barking*. Clips around 10 s work best (AudioSep/CLAP were trained on 10 s segments)."),
    code("""MY_AUDIO = ""   # e.g. "/content/my_clip.wav" (upload via the Files pane)
MY_INSTRUCTION = "Remove the dog barking while preserving everything else."
if MY_AUDIO:
    sh(f'python scripts/run_sonic.py --audio "{MY_AUDIO}" --instruction "{MY_INSTRUCTION}" --config {TUNED} {OVS}')
    out = f"{RESULTS_DIR}/single/{os.path.splitext(os.path.basename(MY_AUDIO))[0]}"
    display(Audio(MY_AUDIO)); display(Audio(f"{out}/sonic.wav")); display(Image(f"{out}/localization.png"))"""),
    md("## 9. Save results\nZips results (without the large audio cache) and examples. Copy `sonic_results.zip` back to the repo's `results/` to update the report."),
    code("""%cd {WORK_DIR}
!cd {RESULTS_DIR} && zip -qr /content/sonic_results.zip . -x "cache/*" "main/eval/audio/*"
!ls -lh /content/sonic_results.zip
if USE_DRIVE:
    !cp /content/sonic_results.zip {DRIVE_DIR}/
try:
    from google.colab import files
    files.download("/content/sonic_results.zip")
except Exception as e:
    print("download manually from the Files pane:", e)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = ROOT / "notebooks" / "sonic_colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
