import numpy as np
import pytest

from sonic.config import load_config, replace_section
from sonic.pipeline import edit, edit_with_estimate
from tests.fakes import FakeEmbedder, FakeSeparator

SR = 8000


class FakeModel:
    """Separator + embedder test double (not a real model)."""

    def __init__(self, target, background, prompt):
        self.sr = SR
        self._sep = FakeSeparator(target, background)
        self._emb = FakeEmbedder(prompt)

    def separate(self, wav, text):
        return self._sep.separate(wav, text)

    def embed_text(self, texts):
        return self._emb.embed_text(texts)

    def embed_audio(self, windows):
        return self._emb.embed_audio(windows)


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    n = 6 * SR
    b = (0.02 * rng.standard_normal(n)).astype(np.float32)
    s = np.zeros(n, dtype=np.float32)
    s[2 * SR: 3 * SR] = 0.5 * rng.standard_normal(SR)
    return s + b, s, b, [(2.0, 3.0)]


@pytest.fixture
def cfg():
    c = load_config()
    return replace_section(c, "localization", win_s=0.5, hop_s=0.125, background_prompts=("noise",))


def test_config_loads_and_overrides():
    c = load_config(overrides=["localization.threshold=0.3", "editing.mask=soft"])
    assert c.localization.threshold == 0.3 and c.editing.mask == "soft"
    assert isinstance(c.localization.background_prompts, tuple)
    with pytest.raises(KeyError):
        load_config(overrides=["localization.bogus=1"])
    with pytest.raises(ValueError):
        load_config(overrides=["threshold=1"])
    assert c.with_overrides(["editing.crossfade_s=0"]).editing.crossfade_s == 0


def test_baseline_subtracts_everywhere(scene, cfg):
    x, s, b, _ = scene
    model = FakeModel(s, b, "dog barking")
    out = edit(x, "Remove the dog barking.", model, cfg, method="baseline")
    assert np.allclose(out.y, x - model.separate(x, ""), atol=1e-6)
    assert out.intervals is None
    assert set(out.timings) >= {"parse", "separation", "editing", "total"}


def test_sonic_preserves_samples_outside_mask(scene, cfg):
    x, s, b, gt = scene
    model = FakeModel(s, b, "dog barking")
    out = edit(x, "Remove the dog barking.", model, cfg, method="sonic")
    assert len(out.intervals) == 1
    untouched = out.mask == 0
    assert untouched.sum() > 0.5 * len(x)
    assert np.array_equal(out.y[untouched], x[untouched])
    assert "localization" in out.timings


def test_oracle_uses_padded_ground_truth(scene, cfg):
    x, s, b, gt = scene
    out = edit(x, "Remove the dog barking.", FakeModel(s, b, "dog barking"), cfg, method="oracle", gt_intervals=gt)
    pad = cfg.localization.pad_s
    assert out.intervals == [pytest.approx((2.0 - pad, 3.0 + pad))]


def test_localized_editing_beats_global_outside_events(scene, cfg):
    """With a leaky estimate, gating must reduce error where the target is silent."""
    x, s, b, gt = scene
    model = FakeModel(s, b, "dog barking")
    s_hat = model.separate(x, "")
    base = edit_with_estimate(x, s_hat, "baseline", cfg, SR)
    orac = edit_with_estimate(x, s_hat, "oracle", cfg, SR, gt_intervals=gt)
    outside = np.ones(len(x), bool)
    outside[int(1.8 * SR): int(3.2 * SR)] = False
    err = lambda y: np.mean((y[outside] - b[outside]) ** 2)
    assert err(orac.y) < err(base.y)


def test_soft_mask_mode(scene, cfg):
    x, s, b, _ = scene
    c = replace_section(cfg, "editing", mask="soft")
    out = edit(x, "Remove the dog barking.", FakeModel(s, b, "dog barking"), c, method="sonic")
    assert 0 <= out.mask.min() and out.mask.max() <= 1


def test_unknown_method_raises(scene, cfg):
    x, s, b, _ = scene
    with pytest.raises(ValueError):
        edit_with_estimate(x, s, "magic", cfg, SR)
    with pytest.raises(ValueError):
        edit_with_estimate(x, s, "oracle", cfg, SR)
