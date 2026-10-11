import numpy as np
import pytest

from sonic import dataset as ds
from sonic.instruction import parse_instruction

SR = 8000


def tone(freq, dur, sr=SR, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def make(seed=0, n_events=2, speech=True):
    rng = np.random.default_rng(seed)
    bed = 0.3 * rng.standard_normal(SR * 11)
    sp = tone(220, 12) if speech else None
    events = [tone(1000, 0.8, amp=2.0) for _ in range(n_events)]
    return ds.synthesize_sample(bed, events, snr_db=0.0, rng=rng, sr=SR, speech=sp)


def test_mixture_is_exact_sum_and_not_clipped():
    out = make()
    assert np.array_equal(out["mixture"], out["target"] + out["background"])
    assert np.max(np.abs(out["mixture"])) <= 0.9 + 1e-6
    stem_sum = sum(out["stems"].values())
    assert np.allclose(stem_sum, out["mixture"], atol=1e-6)


def test_intervals_match_target_support():
    out = make(n_events=3, speech=False)
    nz = np.flatnonzero(out["target"])
    support = np.zeros(len(out["target"]), dtype=bool)
    for t0, t1 in out["intervals"]:
        support[int(round(t0 * SR)): int(round(t1 * SR))] = True
    assert np.all(support[nz])
    assert support.sum() == pytest.approx(len(nz), rel=0.01)
    ints = out["intervals"]
    assert all(ints[i + 1][0] - ints[i][1] >= 0.5 - 1e-9 for i in range(len(ints) - 1))


def test_deterministic_for_seed():
    a, b = make(seed=7), make(seed=7)
    assert np.array_equal(a["mixture"], b["mixture"]) and a["intervals"] == b["intervals"]


def test_snr_is_respected_before_peak_gain():
    rng = np.random.default_rng(1)
    bed = 0.01 * rng.standard_normal(SR * 10)
    ev = tone(1000, 1.0, amp=0.01)
    out = ds.synthesize_sample(bed, [ev], snr_db=-3.0, rng=rng, sr=SR)
    t0, t1 = out["intervals"][0]
    active = out["target"][int(t0 * SR): int(t1 * SR)]
    snr = 20 * np.log10(np.sqrt(np.mean(active ** 2)) / np.sqrt(np.mean(out["background"] ** 2)))
    assert snr == pytest.approx(-3.0, abs=0.05)


def test_trim_active_removes_silence():
    x = np.concatenate([np.zeros(SR), tone(500, 0.5), np.zeros(SR)])
    y = ds.trim_active(x, SR)
    assert 0.45 * SR <= len(y) <= 0.6 * SR


def test_concat_crossfade_length_and_continuity():
    clips = [np.ones(SR), np.ones(SR)]
    y = ds.concat_crossfade(clips, length=int(2.5 * SR), sr=SR)
    assert len(y) == int(2.5 * SR)
    assert np.allclose(y, 1.0, atol=1e-6)  # equal-gain sum of identical DC is flat


def test_place_events_raises_when_impossible():
    with pytest.raises(ValueError):
        ds.place_events([900, 900], length=1000, rng=np.random.default_rng(0), min_gap=10, margin=0)


@pytest.mark.parametrize("cls", list(ds.TARGET_PROMPTS))
def test_every_template_parses_back_to_prompt(cls):
    for tmpl in ds.INSTRUCTION_TEMPLATES:
        text = tmpl.format(p=ds.TARGET_PROMPTS[cls])
        assert parse_instruction(text).target == ds.TARGET_PROMPTS[cls]


def test_scenarios_use_known_targets():
    for sc in ds.SCENARIOS.values():
        assert set(sc.targets) <= set(ds.TARGET_PROMPTS)


def test_events_of_different_lengths_fit_and_match_intervals():
    """Regression: sorted onsets were paired with unsorted events of different lengths."""
    for seed in range(200):
        rng = np.random.default_rng(seed)
        bed = 0.1 * rng.standard_normal(SR * 10)
        lengths = [0.3, 1.5, 0.8]
        events = [tone(500 + 100 * i, L, amp=1.0) for i, L in enumerate(lengths)]
        out = ds.synthesize_sample(bed, events, snr_db=0.0, rng=rng, sr=SR)
        ints = out["intervals"]
        assert all(t1 <= 10.0 for _, t1 in ints)
        assert sorted(round(t1 - t0, 3) for t0, t1 in ints) == sorted(lengths)
        assert ints == sorted(ints)
