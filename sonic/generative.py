"""Generative (diffusion) removal with optional temporal localization.

The pretrained model is AudioLDM2 (Liu et al., 2024; ``cvssp/audioldm2``,
CC BY-NC-SA 4.0), used unchanged through ``diffusers``. This module adds an
SDEdit-style editor:

1. the input is converted to the model's log-mel representation and encoded
   with the VAE;
2. the latent is noised to a fraction ``strength`` of the diffusion schedule;
3. it is denoised with classifier-free guidance, using a generic positive
   prompt and the **target as the negative prompt** (steering away from it);
4. with a temporal mask, latents outside the mask are replaced at every step
   by the correspondingly noised original latents (blended / RePaint-style
   inpainting), so only the masked region is regenerated;
5. the decoded audio is composited into the original waveform with the same
   crossfaded sample mask SONIC uses, so unmasked samples stay bit-identical.

Without a mask (``mask=None``) the whole clip is regenerated, which is the
usual global generative edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonic.audio import resample


@dataclass(frozen=True)
class GenerativeParams:
    model_id: str = "cvssp/audioldm2"
    prompt: str = "natural background ambience, high quality"
    steps: int = 100
    guidance_scale: float = 3.5
    strengths: tuple[float, ...] = (0.6, 1.0)
    seed: int = 0
    fp16: bool = True


# ------------------------------------------------------------- pure helpers --

def sample_mask_to_frames(mask: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    """Average a sample-level mask over each analysis hop -> one value per frame."""
    m = np.zeros(n_frames)
    for i in range(n_frames):
        seg = mask[i * hop:(i + 1) * hop]
        m[i] = seg.max() if len(seg) else 0.0
    return m


def frames_to_latent(frame_mask: np.ndarray, factor: int, n_latent: int, dilate: int = 1) -> np.ndarray:
    """Max-pool a frame mask to latent resolution and dilate, so edits fully cover the region."""
    lat = np.zeros(n_latent)
    for j in range(n_latent):
        seg = frame_mask[j * factor:(j + 1) * factor]
        lat[j] = 1.0 if len(seg) and seg.max() > 0 else 0.0
    if dilate:
        idx = np.flatnonzero(lat)
        for k in range(1, dilate + 1):
            lat[np.clip(idx - k, 0, n_latent - 1)] = 1.0
            lat[np.clip(idx + k, 0, n_latent - 1)] = 1.0
    return lat


def composite(x: np.ndarray, generated: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Waveform composite: generated audio where mask > 0, original elsewhere (bit-identical)."""
    if mask is None:
        return generated.astype(np.float32)
    y = np.array(x, dtype=np.float32, copy=True)
    on = mask > 0
    y[on] = ((1 - mask[on]) * x[on] + mask[on] * generated[on]).astype(np.float32)
    return y


# ------------------------------------------------------------ model wrapper --

class AudioLDM2Editor:
    """Loads AudioLDM2 once; ``edit`` performs (masked) SDEdit removal."""

    def __init__(self, params: GenerativeParams = GenerativeParams(), device: str = "auto", pipe=None):
        import torch

        self.torch = torch
        self.p = params
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if pipe is None:
            from diffusers import AudioLDM2Pipeline, DDIMScheduler

            dtype = torch.float16 if (params.fp16 and device.startswith("cuda")) else torch.float32
            pipe = AudioLDM2Pipeline.from_pretrained(params.model_id, torch_dtype=dtype)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.dtype = next(self.pipe.unet.parameters()).dtype
        voc = self.pipe.vocoder.config
        self.sr = int(voc.sampling_rate)
        self.hop = int(np.prod(voc.upsample_rates))
        self.n_mels = int(voc.model_in_dim)
        self.n_fft = 1024
        import librosa

        self.mel_basis = torch.tensor(
            librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=0, fmax=self.sr // 2),
            dtype=torch.float32,
        )
        self._prompt_cache: dict[str, tuple] = {}

    # AudioLDM's TacotronSTFT front end: reflect padding, magnitude STFT, Slaney mel, log(clamp 1e-5).
    def log_mel(self, wav: np.ndarray) -> "torch.Tensor":
        torch = self.torch
        y = torch.as_tensor(np.asarray(wav, np.float32))[None, None]
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(y, (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                          window=torch.hann_window(self.n_fft), center=False, return_complex=True).abs()
        mel = torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))  # (1, n_mels, T)
        return mel.transpose(1, 2)  # (1, T, n_mels)

    def _embeds(self, negative: str):
        if negative not in self._prompt_cache:
            with self.torch.no_grad():
                self._prompt_cache[negative] = self.pipe.encode_prompt(
                    [self.p.prompt], self.device, 1, True, negative_prompt=[negative])
        return self._prompt_cache[negative]

    def vocode(self, mel: "torch.Tensor") -> np.ndarray:
        """Log-mel (1, T, F) or (1, 1, T, F) -> waveform at ``self.sr``."""
        with self.torch.no_grad():
            return self.pipe.mel_spectrogram_to_waveform(mel.to(self.device, self.dtype))[0].numpy()

    def roundtrip(self, wav16: np.ndarray) -> np.ndarray:
        """log-mel -> VAE encode -> decode -> vocoder (no diffusion): the model's reconstruction ceiling."""
        torch = self.torch
        mel = self.log_mel(wav16)
        T = mel.shape[1]
        f = self.pipe.vae_scale_factor
        Tp = int(np.ceil(T / f) * f)
        if Tp > T:
            mel = torch.cat([mel, mel[:, -1:].expand(1, Tp - T, self.n_mels)], dim=1)
        with torch.no_grad():
            z = self.pipe.vae.encode(mel[:, None].to(self.device, self.dtype)).latent_dist.mode()
            mel_out = self.pipe.vae.decode(z).sample
        return self.vocode(mel_out)[: len(wav16)]

    def edit(self, x: np.ndarray, x_sr: int, target: str, strength: float,
             sample_mask: np.ndarray | None = None, seed: int | None = None) -> np.ndarray:
        """Remove ``target`` by regeneration; returns audio at ``x_sr`` with len(x) samples."""
        torch = self.torch
        pipe = self.pipe
        x16 = resample(x, x_sr, self.sr)
        n16 = len(x16)
        mel = self.log_mel(x16)  # (1, T, F)
        T = mel.shape[1]
        f = pipe.vae_scale_factor
        Tp = int(np.ceil(T / f) * f)
        if Tp > T:
            mel = torch.cat([mel, mel[:, -1:].expand(1, Tp - T, self.n_mels)], dim=1)
        with torch.no_grad():
            z0 = pipe.vae.encode(mel[:, None].to(self.device, self.dtype)).latent_dist.mode()
            z0 = z0 * pipe.vae.config.scaling_factor

            lat_mask = None
            if sample_mask is not None:
                m16 = resample(sample_mask.astype(np.float32), x_sr, self.sr)[:n16]
                fm = sample_mask_to_frames(np.clip(m16, 0, 1), self.hop, Tp)
                lm = frames_to_latent(fm, f, z0.shape[2])
                lat_mask = torch.as_tensor(lm, device=self.device, dtype=z0.dtype)[None, None, :, None]

            sched = pipe.scheduler
            sched.set_timesteps(self.p.steps, device=self.device)
            n_run = max(1, min(self.p.steps, int(round(self.p.steps * strength))))
            timesteps = sched.timesteps[self.p.steps - n_run:]
            g = torch.Generator(device="cpu").manual_seed(self.p.seed if seed is None else seed)
            noise = torch.randn(z0.shape, generator=g, dtype=torch.float32).to(self.device, z0.dtype)
            z = sched.add_noise(z0, noise, timesteps[:1])

            prompt_embeds, attn, gen_embeds = self._embeds(target)
            for i, t in enumerate(timesteps):
                inp = sched.scale_model_input(torch.cat([z] * 2), t)
                eps = pipe.unet(inp, t, encoder_hidden_states=gen_embeds, encoder_hidden_states_1=prompt_embeds,
                                encoder_attention_mask_1=attn, return_dict=False)[0]
                e_u, e_c = eps.chunk(2)
                eps = e_u + self.p.guidance_scale * (e_c - e_u)
                z = sched.step(eps, t, z).prev_sample
                if lat_mask is not None:
                    known = z0 if i == len(timesteps) - 1 else sched.add_noise(z0, noise, timesteps[i + 1:i + 2])
                    z = lat_mask * z + (1 - lat_mask) * known
            mel_out = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
        wav16 = self.vocode(mel_out)[:n16]
        if len(wav16) < n16:
            wav16 = np.pad(wav16, (0, n16 - len(wav16)))
        gen = resample(wav16, self.sr, x_sr)
        gen = np.pad(gen, (0, max(0, len(x) - len(gen))))[: len(x)]
        return composite(x, gen, sample_mask)
