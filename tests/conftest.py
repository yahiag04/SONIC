"""Shared fixtures: synthetic stand-ins for ESC-50 / LibriSpeech and a test-double model."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sonic import dataset as ds
from tests.fakes import FakeEmbedder

SR = 32000


def _write(path, x, sr=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr)


@pytest.fixture(scope="module")
def fake_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    esc = root / "ESC-50"
    rows = []
    classes = set(ds.BED_CLASSES) | {c for sc in ds.SCENARIOS.values() for c in sc.targets}
    for cls in sorted(classes):
        for fold in (1, 5):
            for j in range(2):
                fname = f"{fold}-{cls}-{j}.wav"
                if cls in ds.BED_CLASSES:
                    x = 0.1 * rng.standard_normal(5 * 44100)
                else:
                    x = np.zeros(5 * 44100)
                    x[44100:2 * 44100] = 0.5 * np.sin(2 * np.pi * 800 * np.arange(44100) / 44100)
                _write(esc / "audio" / fname, x)
                rows.append({"filename": fname, "fold": fold, "target": 0, "category": cls,
                             "esc10": False, "src_file": 0, "take": "A"})
    (esc / "meta").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(esc / "meta" / "esc50.csv", index=False)
    libri = root / "LibriSpeech" / "dev-clean"
    for spk in range(5):
        for u in range(3):
            _write(libri / str(spk) / "1" / f"{spk}-1-{u}.flac",
                   0.1 * np.sin(2 * np.pi * 200 * np.arange(4 * 16000) / 16000), sr=16000)
    return esc, libri


class EnergyModel:
    """Test double: separates by oracle access to the true target (with leakage)."""

    sr = SR

    def __init__(self):
        self.emb = FakeEmbedder("__target__")

    def separate(self, wav, text):
        return (0.05 * wav).astype(np.float32)  # leaky, deliberately imperfect

    def embed_text(self, texts):
        # treat the first text as the target prompt
        return np.array([[1.0, 0.0]] + [[0.0, 1.0]] * (len(texts) - 1))

    def embed_audio(self, windows):
        return self.emb.embed_audio(windows)


