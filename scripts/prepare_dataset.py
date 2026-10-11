"""Download ESC-50 + LibriSpeech dev-clean and build the dev/eval mixtures.

    python scripts/prepare_dataset.py [--raw-dir data/raw] [overrides...]

ESC-50 (CC BY-NC 3.0, K. J. Piczak) and LibriSpeech (CC BY 4.0, Panayotov et al.)
are downloaded at runtime and never committed. ESC-50 folds 1-4 feed the dev
split (tuning/ablations) and fold 5 feeds the eval split (reported results).
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from _common import base_parser, config_from_args

from sonic.dataset_builder import build_split, list_librispeech, load_esc50, split_pools

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
LIBRI_URLS = (
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://us.openslr.org/resources/12/dev-clean.tar.gz",
)


def download(urls, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in ([urls] if isinstance(urls, str) else urls):
        try:
            print(f"downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.rename(dest)
            return dest
        except Exception as e:  # try the next mirror
            last = e
            print(f"  failed: {e}")
    raise RuntimeError(f"could not download {dest.name}") from last


def main():
    p = base_parser(__doc__)
    p.add_argument("--raw-dir", default=None, help="where raw corpora are stored (default: <data_root>/raw)")
    p.add_argument("--splits", nargs="+", default=["dev", "eval"])
    args = p.parse_args()
    cfg = config_from_args(args)
    data_root = Path(cfg.paths.data_root)
    raw = Path(args.raw_dir) if args.raw_dir else data_root / "raw"

    esc_root = raw / "ESC-50-master"
    if not esc_root.exists():
        z = download(ESC50_URL, raw / "esc50.zip")
        with zipfile.ZipFile(z) as fh:
            fh.extractall(raw)
    libri_root = raw / "LibriSpeech" / "dev-clean"
    if not libri_root.exists():
        t = download(LIBRI_URLS, raw / "dev-clean.tar.gz")
        with tarfile.open(t) as fh:
            fh.extractall(raw)

    pools = split_pools(load_esc50(esc_root), list_librispeech(libri_root))
    sizes = {"dev": cfg.dataset.n_dev_per_scenario, "eval": cfg.dataset.n_eval_per_scenario}
    for split in args.splits:
        m = build_split(pools[split], split, data_root, sizes[split], cfg.dataset)
        print(f"{split}: {len(m)} samples -> {data_root / split}")
        print(m.groupby("scenario").size().to_string())


if __name__ == "__main__":
    main()
