import numpy as np
import pytest

from sonic import audio


def test_resample_changes_length_by_ratio():
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    y = audio.resample(x, 44100, 32000)
    assert y.dtype == np.float32
    assert len(y) == 32000


def test_resample_same_rate_is_identity():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(audio.resample(x, 32000, 32000), x)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(5), -np.ones(5) * 3], axis=1)
    assert np.allclose(audio.to_mono(stereo), -1.0)


def test_save_load_roundtrip(tmp_path):
    x = (0.5 * np.sin(np.linspace(0, 100, 32000))).astype(np.float32)
    p = tmp_path / "a.wav"
    audio.save_audio(p, x)
    y = audio.load_audio(p)
    assert len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-4


def test_scale_to_snr_achieves_requested_snr():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(16000)
    noise = 3 * rng.standard_normal(16000)
    scaled = audio.scale_to_snr(sig, noise, snr_db=-4.0)
    snr = 20 * np.log10(audio.rms(scaled) / audio.rms(noise))
    assert snr == pytest.approx(-4.0, abs=1e-6)


def test_scale_to_rms():
    x = np.random.default_rng(2).standard_normal(1000)
    assert audio.rms(audio.scale_to_rms(x, 0.1)) == pytest.approx(0.1)


def test_fit_peak_scales_all_stems_consistently():
    stems = {"a": np.full(10, 0.8), "b": np.full(10, 0.7)}
    out, gain = audio.fit_peak(stems, peak=0.9)
    mix = out["a"] + out["b"]
    assert np.max(np.abs(mix)) == pytest.approx(0.9)
    assert np.allclose(out["a"] / out["b"], 0.8 / 0.7)
    assert gain == pytest.approx(0.9 / 1.5)


def test_fit_peak_leaves_quiet_mix_unchanged():
    stems = {"a": np.full(10, 0.1)}
    out, gain = audio.fit_peak(stems, peak=0.9)
    assert gain == 1.0
    assert np.array_equal(out["a"], stems["a"])


def test_frame_starts_cover_signal():
    starts = audio.frame_starts(n=1000, win=300, hop=100)
    assert starts[0] == 0
    assert starts[-1] + 300 >= 1000
    assert np.all(np.diff(starts) <= 100)


def test_frame_starts_short_signal_single_window():
    assert list(audio.frame_starts(n=50, win=300, hop=100)) == [0]


def test_raised_cosine_shape():
    r = audio.raised_cosine(64)
    assert len(r) == 64
    assert r[0] == pytest.approx(0.0, abs=1e-3)
    assert r[-1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(r) >= 0)


def test_raised_cosine_zero_length():
    assert len(audio.raised_cosine(0)) == 0
