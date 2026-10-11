import numpy as np
import pandas as pd
import pytest

from sonic import evaluation as ev

SR = 8000


def test_si_sdr_identical_is_large_and_scale_invariant():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(SR)
    assert ev.si_sdr(ref, ref) > 100
    noisy = ref + 0.1 * rng.standard_normal(SR)
    assert ev.si_sdr(3.0 * noisy, ref) == pytest.approx(ev.si_sdr(noisy, ref), abs=1e-9)


def test_si_sdr_known_snr():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(200000)
    noise = rng.standard_normal(200000)
    noise -= ref * (noise @ ref) / (ref @ ref)  # orthogonal
    noise *= np.linalg.norm(ref) / np.linalg.norm(noise) * 10 ** (-10 / 20)
    assert ev.si_sdr(ref + noise, ref) == pytest.approx(10.0, abs=0.01)


def test_sdr_penalises_gain_unlike_si_sdr():
    ref = np.random.default_rng(6).standard_normal(SR)
    assert ev.si_sdr(0.95 * ref, ref) > 100
    assert ev.sdr(0.95 * ref, ref) == pytest.approx(20 * np.log10(1 / 0.05))
    assert ev.sdr(ref, ref) == ev.SDR_CAP_DB


def test_region_mask():
    m = ev.region_mask([(0.5, 1.0)], n=2 * SR, sr=SR)
    assert m.sum() == SR // 2 and m[SR // 2] and not m[SR]


def test_target_suppression_db():
    s = np.zeros(SR)
    s[100:500] = 1.0
    b = np.zeros(SR)
    y = b + 0.1 * s
    region = s != 0
    assert ev.target_suppression_db(y, s, b, region) == pytest.approx(20.0)


def test_frame_metrics():
    m = ev.frame_metrics(pred=[(1.0, 2.0)], gt=[(1.5, 2.5)], duration_s=4.0, frame_rate=100)
    assert m["loc_precision"] == pytest.approx(0.5)
    assert m["loc_recall"] == pytest.approx(0.5)
    assert m["loc_iou"] == pytest.approx(1 / 3)
    assert m["loc_f1"] == pytest.approx(0.5)


def test_frame_metrics_empty_prediction():
    m = ev.frame_metrics(pred=[], gt=[(1.0, 2.0)], duration_s=4.0, frame_rate=100)
    assert m["loc_recall"] == 0.0 and m["loc_iou"] == 0.0 and np.isnan(m["loc_precision"])


def test_boundary_errors_match_by_overlap():
    e = ev.boundary_errors(pred=[(0.9, 2.2), (5.0, 5.5)], gt=[(1.0, 2.0), (3.0, 3.5)])
    assert e["onset_err_s"] == pytest.approx(0.1)
    assert e["offset_err_s"] == pytest.approx(0.2)
    assert e["event_recall"] == pytest.approx(0.5)


def test_evaluate_sample_perfect_and_untouched_regions():
    rng = np.random.default_rng(2)
    n = 4 * SR
    b = 0.1 * rng.standard_normal(n)
    s = np.zeros(n)
    s[SR:2 * SR] = 0.5 * rng.standard_normal(SR)
    x = s + b
    gt = [(1.0, 2.0)]
    mask = np.zeros(n)
    mask[SR:2 * SR] = 1
    r = ev.evaluate_sample(x=x, y=b.copy(), s=s, b=b, gt_intervals=gt, sr=SR, pred_intervals=gt, mask=mask)
    assert r["sisdr_bg"] == ev.SDR_CAP_DB and r["sdr_bg_out"] == ev.SDR_CAP_DB
    assert r["sisdri"] > 0
    assert r["loc_iou"] == pytest.approx(1.0)
    assert r["mse_out"] == pytest.approx(0.0)
    # doing nothing: untouched background outside, zero suppression
    r0 = ev.evaluate_sample(x=x, y=x, s=s, b=b, gt_intervals=gt, sr=SR)
    assert r0["target_suppression_db"] == pytest.approx(0.0)
    assert r0["sisdri"] == pytest.approx(0.0)
    assert "loc_iou" not in r0


def test_short_regions_give_nan():
    n = 2 * SR
    s = np.zeros(n)
    s[100:200] = 1
    b = np.random.default_rng(3).standard_normal(n)
    r = ev.evaluate_sample(x=s + b, y=b, s=s, b=b, gt_intervals=[(100 / SR, 200 / SR)], sr=SR)
    assert np.isnan(r["sdr_bg_in"])


def test_edge_click_metric_detects_hard_switch():
    rng = np.random.default_rng(4)
    n = 2 * SR
    b = 0.05 * rng.standard_normal(n)
    s_hat = np.full(n, 0.3)  # DC estimate: a hard mask switch creates a step
    hard = np.zeros(n)
    hard[SR // 2: SR] = 1
    from sonic.localization import intervals_to_mask
    soft = intervals_to_mask([(0.5, 1.0)], n, SR, crossfade_s=0.05)
    e_hard = ev.edge_click_db(b - hard * s_hat - b + 0, hard, SR)
    e_soft = ev.edge_click_db(b - soft * s_hat - b + 0, soft, SR)
    assert e_hard > e_soft


def test_aggregate_and_paired_test():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(30):
        base = rng.standard_normal()
        rows.append({"sample_id": i, "method": "baseline", "m": base})
        rows.append({"sample_id": i, "method": "sonic", "m": base + 1.0 + 0.1 * rng.standard_normal()})
    df = pd.DataFrame(rows)
    agg = ev.aggregate(df, ["m"], by=["method"])
    row = agg[agg.method == "sonic"].iloc[0]
    assert row.m_ci_low <= row.m_mean <= row.m_ci_high
    t = ev.paired_test(df, "m", "baseline", "sonic")
    assert t["n"] == 30 and t["median_diff"] > 0.8 and t["p_value"] < 1e-4


def test_log_spectral_distance_properties():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(SR)
    assert ev.log_spectral_distance(ref, ref, SR) == pytest.approx(0.0, abs=1e-6)
    # phase-insensitive: a sign flip leaves the spectrum unchanged
    assert ev.log_spectral_distance(-ref, ref, SR) == pytest.approx(0.0, abs=1e-6)
    # a 2x gain is a constant 6.02 dB offset in every bin
    assert ev.log_spectral_distance(2 * ref, ref, SR) == pytest.approx(20 * np.log10(2), abs=1e-3)
    region = np.zeros(SR, dtype=bool)
    region[:100] = True
    assert np.isnan(ev.log_spectral_distance(ref, ref, SR, region))
