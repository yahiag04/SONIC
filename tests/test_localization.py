import numpy as np
import pytest

from sonic import localization as loc
from tests.fakes import FakeEmbedder

SR = 8000


def burst_signal(duration=6.0, burst=(2.0, 3.0), seed=0):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(duration * SR))
    i0, i1 = int(burst[0] * SR), int(burst[1] * SR)
    x[i0:i1] += 0.5 * rng.standard_normal(i1 - i0)
    return x.astype(np.float32)


def test_window_scores_peak_on_burst():
    x = burst_signal()
    ws = loc.window_scores(x, SR, "dog", FakeEmbedder("dog"), win_s=0.5, hop_s=0.125,
                           background_prompts=["noise"])
    centers = ws.starts_s + 0.25
    best = centers[np.argmax(ws.scores)]
    assert 2.0 <= best <= 3.0
    assert len(ws.scores) == len(ws.starts_s) == len(ws.target_sim)


def test_window_scores_batches_embedding_calls():
    x = burst_signal()
    emb = FakeEmbedder("dog")
    loc.window_scores(x, SR, "dog", emb, win_s=0.5, hop_s=0.125, background_prompts=["n"], batch_size=8)
    n_windows = len(loc.window_starts(len(x), SR, 0.5, 0.125))
    assert emb.audio_calls == int(np.ceil(n_windows / 8))


def test_normalize_scores():
    s = np.array([1.0, 2.0, 3.0])
    z = loc.normalize_scores(s, "zscore")
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
    m = loc.normalize_scores(s, "minmax")
    assert m.min() == 0.0 and m.max() == 1.0
    assert np.array_equal(loc.normalize_scores(s, "none"), s)
    assert np.all(loc.normalize_scores(np.ones(3), "zscore") == 0.0)
    with pytest.raises(ValueError):
        loc.normalize_scores(s, "bogus")


def test_frame_curve_averages_overlapping_windows():
    starts = np.array([0.0, 0.5])
    times, curve = loc.frame_curve(starts, np.array([0.0, 1.0]), win_s=1.0, duration_s=1.5, frame_rate=10)
    assert len(times) == 15
    assert curve[1] == pytest.approx(0.0)    # t=0.15: only window 0
    assert curve[7] == pytest.approx(0.5)    # t=0.75: both windows
    assert curve[13] == pytest.approx(1.0)   # t=1.35: only window 1


def test_median_smooth_removes_isolated_frames():
    active = np.zeros(20, dtype=bool)
    active[5] = True
    active[10:15] = True
    active[12] = False
    out = loc.median_smooth(active, kernel=3)
    assert not out[5]
    assert out[10:15].all() or out[11:14].all()
    assert out[12]


def test_active_to_intervals():
    active = np.array([0, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert loc.active_to_intervals(active, frame_rate=10) == [(0.1, 0.3), (0.5, 0.8)]


def test_postprocess_gap_fill_min_event_and_pad():
    intervals = [(1.0, 1.5), (1.6, 2.0), (4.0, 4.05), (5.8, 6.0)]
    out = loc.postprocess(intervals, min_event_s=0.1, min_gap_s=0.2, pad_s=0.1, duration_s=6.0)
    assert out == [pytest.approx((0.9, 2.1)), pytest.approx((5.7, 6.0))]


def test_postprocess_merges_after_padding():
    out = loc.postprocess([(1.0, 1.4), (1.7, 2.0)], min_event_s=0.0, min_gap_s=0.0, pad_s=0.2, duration_s=3)
    assert out == [pytest.approx((0.8, 2.2))]


def test_intervals_to_mask_properties():
    n = SR * 4
    mask = loc.intervals_to_mask([(1.0, 2.0)], n, SR, crossfade_s=0.1)
    i0, i1, L = SR, 2 * SR, int(0.1 * SR)
    assert mask.shape == (n,)
    assert np.all(mask[i0:i1] == 1.0)
    assert np.all(mask[: i0 - L] == 0.0)
    assert np.all(mask[i1 + L:] == 0.0)
    assert np.all(np.diff(mask[i0 - L: i0 + 1]) >= 0)
    assert np.all(np.diff(mask[i1 - 1: i1 + L]) <= 0)
    assert mask.min() >= 0 and mask.max() <= 1


def test_intervals_to_mask_no_crossfade_is_step():
    mask = loc.intervals_to_mask([(1.0, 2.0)], SR * 3, SR, crossfade_s=0.0)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[SR - 1] == 0 and mask[SR] == 1 and mask[2 * SR - 1] == 1 and mask[2 * SR] == 0


def test_intervals_to_mask_ramps_clipped_at_edges():
    mask = loc.intervals_to_mask([(0.0, 0.5), (3.9, 4.0)], SR * 4, SR, crossfade_s=0.2)
    assert mask[0] == 1.0 and mask[-1] == 1.0


def test_soft_mask_range_and_monotonic_in_score():
    times = np.linspace(0, 1, 11)
    curve = np.linspace(-2, 2, 11)
    m = loc.soft_mask(times, curve, n=SR, sr=SR, threshold=0.0, temperature=0.5)
    assert m.shape == (SR,)
    assert 0 <= m.min() and m.max() <= 1
    assert np.all(np.diff(m) >= -1e-12)


def test_localize_end_to_end_finds_burst():
    x = burst_signal(burst=(2.0, 3.0))
    params = loc.LocalizationParams(win_s=0.5, hop_s=0.125, threshold=0.5, median_kernel_s=0.11,
                                    min_event_s=0.1, min_gap_s=0.2, pad_s=0.0,
                                    background_prompts=("noise",))
    res = loc.localize(x, SR, "dog", FakeEmbedder("dog"), params)
    assert len(res.intervals) == 1
    t0, t1 = res.intervals[0]
    assert abs(t0 - 2.0) < 0.3 and abs(t1 - 3.0) < 0.3
    assert len(res.frame_times) == len(res.frame_scores)


def test_localize_reuses_precomputed_window_scores():
    x = burst_signal()
    params = loc.LocalizationParams(win_s=0.5, hop_s=0.125, background_prompts=("noise",))
    emb = FakeEmbedder("dog")
    ws = loc.window_scores(x, SR, "dog", emb, params.win_s, params.hop_s, params.background_prompts)
    calls = emb.audio_calls
    res = loc.localize_from_scores(ws, duration_s=len(x) / SR, params=params)
    assert emb.audio_calls == calls
    assert res.intervals
