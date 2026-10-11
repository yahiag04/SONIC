"""Clone AudioSep at a pinned commit and download its pretrained checkpoints.

    python scripts/fetch_models.py [paths.models_dir=/content/models]

Nothing here is committed to the repository; weights come from the official
AudioSep Hugging Face repo (audo/AudioSep).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _common import base_parser, config_from_args

from sonic.separation import AUDIOSEP_COMMIT, AUDIOSEP_REPO, CLAP_CKPT, HF_REPO, SEP_CKPT


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    models = Path(cfg.paths.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    repo = models / "AudioSep"
    if not repo.exists():
        subprocess.run(["git", "clone", AUDIOSEP_REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", AUDIOSEP_COMMIT], check=True)
    print(f"AudioSep code at {repo} (commit {AUDIOSEP_COMMIT[:7]})")

    from huggingface_hub import hf_hub_download

    ckpt_dir = models / "checkpoints"
    link_dir = repo / "checkpoint"  # AudioSep builds a default CLAP_Encoder at import time from here
    link_dir.mkdir(exist_ok=True)
    for name in (SEP_CKPT, CLAP_CKPT):
        path = Path(hf_hub_download(HF_REPO, name, local_dir=str(ckpt_dir))).resolve()
        link = link_dir / name
        if not link.exists():
            link.symlink_to(path)
        print(f"checkpoint: {path} ({path.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
