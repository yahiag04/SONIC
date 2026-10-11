"""AudioSep wrapper: text-queried separation plus CLAP text/audio embeddings.

AudioSep (Liu et al., 2023; https://github.com/Audio-AGI/AudioSep, MIT) and the
LAION CLAP checkpoint it uses are pretrained third-party components. This
module only loads them and exposes two small interfaces:

* ``Separator.separate(wav, text) -> target estimate``
* ``Embedder.embed_text / embed_audio`` (L2-normalized CLAP embeddings)

torch and the AudioSep code are imported lazily so the rest of the package
(and the unit tests) run without them. There is deliberately no fallback:
if the model cannot be loaded, construction raises.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from sonic.audio import SR

AUDIOSEP_REPO = "https://github.com/Audio-AGI/AudioSep.git"
AUDIOSEP_COMMIT = "944583f18b84589dc965de3ad77525c945334252"
HF_REPO = "audo/AudioSep"
SEP_CKPT = "audiosep_base_4M_steps.ckpt"
CLAP_CKPT = "music_speech_audioset_epoch_15_esc_89.98.pt"


class Separator(Protocol):
    def separate(self, wav: np.ndarray, text: str) -> np.ndarray: ...


class Embedder(Protocol):
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_audio(self, windows: Sequence[np.ndarray]) -> np.ndarray: ...


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@contextlib.contextmanager
def _audiosep_compat(audiosep_dir: Path):
    """Run inside the AudioSep repo with two compatibility shims for current Colab.

    1. AudioSep resolves config/checkpoint paths relative to its repo root, so we
       chdir there and put it on sys.path.
    2. torch>=2.6 defaults ``torch.load(weights_only=True)``, which rejects these
       checkpoints. We force ``weights_only=False`` only while loading, and only for
       the two official checkpoints downloaded from the AudioSep HF repo.
    3. transformers>=4.31 no longer stores RoBERTa's ``position_ids`` buffer, so the
       CLAP checkpoint's copy of it is an unexpected key under strict loading;
       it is dropped (it is a constant arange, not a learned weight).
    """
    import torch

    old_cwd, old_path = os.getcwd(), list(sys.path)
    orig_load = torch.load

    def load_trusted(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    os.chdir(audiosep_dir)
    sys.path.insert(0, str(audiosep_dir))
    torch.load = load_trusted
    try:
        from models.CLAP.open_clip import factory

        orig_sd = factory.load_state_dict

        def load_state_dict_compat(*args, **kwargs):
            sd = orig_sd(*args, **kwargs)
            sd.pop("text_branch.embeddings.position_ids", None)
            return sd

        factory.load_state_dict = load_state_dict_compat
        try:
            yield
        finally:
            factory.load_state_dict = orig_sd
    finally:
        torch.load = orig_load
        os.chdir(old_cwd)
        sys.path[:] = old_path + [str(audiosep_dir)]  # keep importable for lazy imports


class AudioSepModel:
    """Loads AudioSep once and serves both separation and CLAP embeddings."""

    sr = SR

    def __init__(self, models_dir: str | Path, device: str = "auto", use_chunk: bool = False):
        import torch

        self.torch = torch
        models_dir = Path(models_dir).resolve()
        audiosep_dir = models_dir / "AudioSep"
        sep_ckpt = models_dir / "checkpoints" / SEP_CKPT
        clap_ckpt = models_dir / "checkpoints" / CLAP_CKPT
        for p in (audiosep_dir, sep_ckpt, clap_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found; run scripts/fetch_models.py --models-dir {models_dir}")

        self.device = resolve_device(device)
        self.use_chunk = use_chunk
        with _audiosep_compat(audiosep_dir):
            from models.clap_encoder import CLAP_Encoder
            from utils import load_ss_model, parse_yaml

            configs = parse_yaml(str(audiosep_dir / "config" / "audiosep_base.yaml"))
            query_encoder = CLAP_Encoder(pretrained_path=str(clap_ckpt)).eval()
            self.model = load_ss_model(
                configs=configs, checkpoint_path=str(sep_ckpt), query_encoder=query_encoder
            ).eval().to(self.device)
        self.query_encoder = self.model.query_encoder

    # ---- Separator -------------------------------------------------------
    def separate(self, wav: np.ndarray, text: str) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            cond = self.query_encoder.get_query_embed(modality="text", text=[text], device=self.device)
            inp = {"mixture": torch.as_tensor(np.asarray(wav, np.float32))[None, None, :].to(self.device),
                   "condition": cond}
            if self.use_chunk:
                out = np.squeeze(self.model.ss_model.chunk_inference(inp))
            else:
                out = self.model.ss_model(inp)["waveform"].squeeze().detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        if len(out) < len(wav):
            out = np.pad(out, (0, len(wav) - len(out)))
        return out[: len(wav)]

    # ---- Embedder --------------------------------------------------------
    @staticmethod
    def _normalize(e: np.ndarray) -> np.ndarray:
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            e = self.query_encoder.get_query_embed(modality="text", text=list(texts), device=self.device)
        return self._normalize(e.detach().cpu().numpy().astype(np.float64))

    def embed_audio(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        n = max(len(w) for w in windows)
        batch = np.zeros((len(windows), n), dtype=np.float32)
        for i, w in enumerate(windows):
            batch[i, : len(w)] = w
        with torch.no_grad():
            e = self.query_encoder.get_query_embed(
                modality="audio", audio=torch.as_tensor(batch).to(self.device), device=self.device
            )
        return self._normalize(e.detach().cpu().numpy().astype(np.float64))

    # ---- Resource accounting ---------------------------------------------
    def reset_peak_memory(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float:
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 2**20
        return float("nan")

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
