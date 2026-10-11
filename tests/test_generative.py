"""Generative editor tests.

Pure helpers are always tested. The diffusion loop is exercised with a *tiny,
randomly initialised* AudioLDM2 (the configuration diffusers uses in its own
unit tests). That checks shapes, masking and compositing, not audio quality.
"""

import numpy as np
import pytest

from sonic.generative import composite, frames_to_latent, sample_mask_to_frames


def test_sample_mask_to_frames():
    m = np.zeros(100)
    m[25:35] = 1
    f = sample_mask_to_frames(m, hop=10, n_frames=10)
    assert list(f) == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]


def test_frames_to_latent_maxpool_and_dilate():
    f = np.zeros(16)
    f[5] = 0.3
    assert list(frames_to_latent(f, factor=4, n_latent=4, dilate=0)) == [0, 1, 0, 0]
    assert list(frames_to_latent(f, factor=4, n_latent=4, dilate=1)) == [1, 1, 1, 0]


def test_composite_keeps_unmasked_samples():
    x = np.random.default_rng(0).standard_normal(50).astype(np.float32)
    gen = np.zeros(50, dtype=np.float32)
    mask = np.zeros(50)
    mask[10:20] = 1
    y = composite(x, gen, mask)
    assert np.array_equal(y[:10], x[:10]) and np.array_equal(y[20:], x[20:])
    assert np.all(y[10:20] == 0)
    assert np.array_equal(composite(x, gen, None), gen)


def test_load_pipeline_uses_lm_head_model(monkeypatch):
    """Regression: the checkpoint's GPT2Model lacks generation helpers in recent transformers."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    from sonic.generative import load_pipeline

    seen = {}

    class FakeLM:
        def to(self, dtype):
            seen["lm_dtype"] = dtype
            return self

    class FakePipe:
        scheduler = diffusers.DDIMScheduler()

    def fake_lm(model_id, subfolder):
        seen["subfolder"] = subfolder
        return FakeLM()

    def fake_pipe(model_id, **kw):
        seen.update(kw)
        return FakePipe()

    monkeypatch.setattr(transformers.GPT2LMHeadModel, "from_pretrained", staticmethod(fake_lm))
    monkeypatch.setattr(diffusers.AudioLDM2Pipeline, "from_pretrained", staticmethod(fake_pipe))
    pipe = load_pipeline("x", torch.float16)
    assert seen["subfolder"] == "language_model" and isinstance(seen["language_model"], FakeLM)
    assert seen["lm_dtype"] == torch.float16 and seen["torch_dtype"] == torch.float16
    assert isinstance(pipe.scheduler, diffusers.DDIMScheduler)


@pytest.fixture(scope="module")
def tiny_editor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("librosa")
    from diffusers import AudioLDM2Pipeline, AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    from transformers import (ClapConfig, ClapFeatureExtractor, ClapModel, GPT2Config, GPT2LMHeadModel,
                              RobertaTokenizer, SpeechT5HifiGan, SpeechT5HifiGanConfig, T5Config,
                              T5EncoderModel, T5Tokenizer)

    from sonic.generative import AudioLDM2Editor, GenerativeParams

    torch.manual_seed(0)
    unet = AudioLDM2UNet2DConditionModel(
        block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=8, sample_size=32, in_channels=4,
        out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=(8, 16))
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False)
    vae = AutoencoderKL(block_out_channels=[8, 16], in_channels=1, out_channels=1, norm_num_groups=8,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
                        latent_channels=4)
    text_cfg = {"bos_token_id": 0, "eos_token_id": 2, "hidden_size": 8, "intermediate_size": 37,
                "layer_norm_eps": 1e-05, "num_attention_heads": 1, "num_hidden_layers": 1, "pad_token_id": 1,
                "vocab_size": 1000, "projection_dim": 8}
    audio_cfg = {"spec_size": 8, "window_size": 4, "num_mel_bins": 8, "intermediate_size": 37,
                 "layer_norm_eps": 1e-05, "depths": [1, 1], "num_attention_heads": [1, 1],
                 "num_hidden_layers": 1, "hidden_size": 192, "projection_dim": 8, "patch_size": 2,
                 "patch_stride": 2, "patch_embed_input_channels": 4}
    text_encoder = ClapModel(ClapConfig(text_config=text_cfg, audio_config=audio_cfg, projection_dim=16))
    tokenizer = RobertaTokenizer.from_pretrained("hf-internal-testing/tiny-random-roberta", model_max_length=77)
    fe = ClapFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-ClapModel", hop_length=7900)
    t5 = T5EncoderModel(T5Config(vocab_size=32100, d_model=32, d_ff=37, d_kv=8, num_heads=1, num_layers=1))
    t5_tok = T5Tokenizer.from_pretrained("hf-internal-testing/tiny-random-T5Model", model_max_length=77)
    lm = GPT2LMHeadModel(GPT2Config(n_embd=16, n_head=1, n_layer=1, vocab_size=1000, n_ctx=99, n_positions=99))
    lm.config.max_new_tokens = 8
    proj = AudioLDM2ProjectionModel(text_encoder_dim=16, text_encoder_1_dim=32, langauge_model_dim=16)
    vocoder = SpeechT5HifiGan(SpeechT5HifiGanConfig(
        model_in_dim=8, sampling_rate=16000, upsample_initial_channel=16, upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], normalize_before=False))
    pipe = AudioLDM2Pipeline(vae=vae, text_encoder=text_encoder, text_encoder_2=t5, projection_model=proj,
                             language_model=lm, tokenizer=tokenizer, tokenizer_2=t5_tok, feature_extractor=fe,
                             unet=unet, scheduler=scheduler, vocoder=vocoder)
    return AudioLDM2Editor(GenerativeParams(steps=4, guidance_scale=2.0), device="cpu", pipe=pipe)


def test_tiny_editor_global_and_masked(tiny_editor):
    sr = 32000
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal(sr // 4)).astype(np.float32)  # 0.25 s keeps the tiny model fast
    g = tiny_editor.edit(x, sr, "dog barking", strength=0.5)
    assert g.shape == x.shape and np.isfinite(g).all()

    mask = np.zeros_like(x)
    mask[2000:5000] = 1.0
    y = tiny_editor.edit(x, sr, "dog barking", strength=1.0, sample_mask=mask)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.array_equal(y[mask == 0], x[mask == 0])
    assert not np.array_equal(y[2000:5000], x[2000:5000])


def test_tiny_editor_is_deterministic(tiny_editor):
    x = (0.1 * np.random.default_rng(2).standard_normal(8000)).astype(np.float32)
    a = tiny_editor.edit(x, 32000, "cough", strength=0.6, seed=3)
    b = tiny_editor.edit(x, 32000, "cough", strength=0.6, seed=3)
    assert np.allclose(a, b, atol=1e-5)


def test_tiny_roundtrip_shape(tiny_editor):
    x16 = (0.1 * np.random.default_rng(4).standard_normal(4000)).astype(np.float32)
    y = tiny_editor.roundtrip(x16)
    assert y.shape == x16.shape and np.isfinite(y).all()


def test_run_generative_end_to_end(tiny_editor, fake_sources, tmp_path, monkeypatch):
    from sonic.config import load_config, replace_section
    from sonic.dataset_builder import build_split, list_librispeech, load_esc50, split_pools
    from sonic.experiment import cache_split
    from sonic.generative_experiment import add_event_clap, run_generative
    from tests.conftest import EnergyModel

    cfg = load_config(overrides=[f"paths.data_root={tmp_path / 'data'}", f"paths.cache_dir={tmp_path / 'cache'}"])
    cfg = replace_section(cfg, "generative", strengths=(0.5,))
    pools = split_pools(load_esc50(fake_sources[0]), list_librispeech(fake_sources[1]))
    build_split(pools["eval"], "eval", tmp_path / "data", n_per_scenario=1,
                cfg=replace_section(cfg, "dataset", length_s=6.0).dataset)
    cache_split(EnergyModel(), cfg, "eval", [1.0], log=lambda *_: None)
    df, outputs = run_generative(cfg, "eval", tiny_editor, log=lambda *_: None)
    assert set(df.method) == {"gen_global@0.5", "gen_sonic@0.5", "gen_oracle@0.5", "baseline", "sonic", "oracle"}
    assert len(df) == 3 * 6
    df = add_event_clap(df, outputs, EnergyModel(), cfg, "eval")
    assert "clap_event_target_drop" in df
    g = df[df.method == "gen_global@0.5"]
    assert (g.masked_fraction == 1.0).all()
