"""Reference-based metrics for synthetic mixtures with known sources.

Notation: x = mixture, s = true target, b = reference background (x = s + b),
y = edited output. E = union of ground-truth target intervals, Ē = complement.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon

Interval = tuple[float, float]
EPS = 1e-12
MIN_REGION_S = 0.25
SDR_CAP_DB = 60.0  # perfectly preserved regions would otherwise give ~inf and dominate means


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB (both signals zero-meaned)."""
    est = np.asarray(est, dtype=np.float64) - np.mean(est)
    ref = np.asarray(ref, dtype=np.float64) - np.mean(ref)
    alpha = (est @ ref) / (ref @ ref + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((target @ target + EPS) / (noise @ noise + EPS)))


def sdr(est: np.ndarray, ref: np.ndarray, cap: float = SDR_CAP_DB) -> float:
    """Scale-dependent SDR: 10 log10(||ref||^2 / ||est - ref||^2), capped at ``cap`` dB.

    Unlike SI-SDR this penalises gain changes, which matters for background
    preservation: subtracting a leaky estimate that is a scaled copy of the
    background is invisible to SI-SDR but not to SDR.
    """
    ref = np.asarray(ref, dtype=np.float64)
    err = np.asarray(est, dtype=np.float64) - ref
    return float(min(cap, 10 * np.log10((ref @ ref + EPS) / (err @ err + EPS))))


def region_mask(intervals: Sequence[Interval], n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for t0, t1 in intervals:
        m[max(0, int(round(t0 * sr))): min(n, int(round(t1 * sr)))] = True
    return m


def _region_sdr(y, b, region, sr) -> float:
    if region.sum() < MIN_REGION_S * sr:
        return float("nan")
    return sdr(y[region], b[region])


def target_suppression_db(y: np.ndarray, s: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """10 log10(||s_E||^2 / ||(y - b)_E||^2): how much target energy was removed in E."""
    if not region.any():
        return float("nan")
    resid = (y - b)[region]
    return float(10 * np.log10((np.sum(s[region] ** 2) + EPS) / (np.sum(resid ** 2) + EPS)))


def _frames(intervals: Sequence[Interval], n_frames: int, frame_rate: int) -> np.ndarray:
    f = np.zeros(n_frames, dtype=bool)
    for t0, t1 in intervals:
        f[int(round(t0 * frame_rate)): int(round(t1 * frame_rate))] = True
    return f


def frame_metrics(
    pred: Sequence[Interval], gt: Sequence[Interval], duration_s: float, frame_rate: int = 100
) -> dict[str, float]:
    n = int(round(duration_s * frame_rate))
    p, g = _frames(pred, n, frame_rate), _frames(gt, n, frame_rate)
    tp, fp, fn = np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else nan
    recall = tp / (tp + fn) if tp + fn else nan
    if tp + fp + fn == 0:
        iou, f1 = nan, nan
    else:
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"loc_precision": float(precision), "loc_recall": float(recall), "loc_f1": float(f1), "loc_iou": float(iou)}


def boundary_errors(pred: Sequence[Interval], gt: Sequence[Interval]) -> dict[str, float]:
    """Match each GT interval to the predicted interval with the largest overlap."""
    on, off, matched = [], [], 0
    for g0, g1 in gt:
        overlaps = [max(0.0, min(g1, p1) - max(g0, p0)) for p0, p1 in pred]
        if overlaps and max(overlaps) > 0:
            p0, p1 = pred[int(np.argmax(overlaps))]
            on.append(abs(p0 - g0))
            off.append(abs(p1 - g1))
            matched += 1
    nan = float("nan")
    return {
        "onset_err_s": float(np.mean(on)) if on else nan,
        "offset_err_s": float(np.mean(off)) if off else nan,
        "event_recall": matched / len(gt) if gt else nan,
    }


def edge_click_db(error: np.ndarray, mask: np.ndarray, sr: int, near_s: float = 0.02,
                  ring_s: float = 0.1, cutoff_hz: float = 4000.0) -> float:
    """High-frequency energy of the error right at mask transitions vs. its neighbourhood.

    A hard mask switch subtracts a step of the estimate and produces a broadband
    click, which shows up as a positive value. NaN when the mask has no transitions.
    """
    on = mask > 0
    full = mask >= 1 - 1e-9
    edges = np.concatenate([np.flatnonzero(np.diff(on.astype(np.int8))),
                            np.flatnonzero(np.diff(full.astype(np.int8)))])
    if len(edges) == 0:
        return float("nan")
    cutoff = min(cutoff_hz, 0.45 * sr)
    hp = sosfilt(butter(4, cutoff, btype="highpass", fs=sr, output="sos"), error) ** 2
    near = np.zeros(len(error), dtype=bool)
    ring = np.zeros(len(error), dtype=bool)
    k_near, k_ring = int(near_s * sr), int(ring_s * sr)
    for e in np.unique(edges):
        ring[max(0, e - k_ring): e + k_ring] = True
        near[max(0, e - k_near): e + k_near] = True
    ring &= ~near
    if not ring.any():
        return float("nan")
    return float(10 * np.log10((hp[near].mean() + EPS) / (hp[ring].mean() + EPS)))


def log_spectral_distance(y: np.ndarray, ref: np.ndarray, sr: int, region: np.ndarray | None = None,
                          n_fft: int = 1024, hop: int = 256) -> float:
    """Mean per-frame log-spectral distance in dB (phase-insensitive).

    Suited to generated audio, where waveform metrics punish phase differences
    even when the spectrum is right. Frames are kept if their centre lies in
    ``region`` (a boolean sample mask); NaN if fewer than ~0.25 s of frames remain.
    """
    from scipy.signal import stft

    _, _, Y = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    _, _, R = stft(ref, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    d = 10 * np.log10(np.abs(Y) ** 2 + 1e-10) - 10 * np.log10(np.abs(R) ** 2 + 1e-10)
    per_frame = np.sqrt(np.mean(d ** 2, axis=0))
    if region is not None:
        centres = np.arange(per_frame.shape[0]) * hop + n_fft // 2
        keep = region[np.clip(centres, 0, len(region) - 1)]
        if keep.sum() * hop < MIN_REGION_S * sr:
            return float("nan")
        per_frame = per_frame[keep]
    return float(per_frame.mean())


def evaluate_sample(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    gt_intervals: Sequence[Interval],
    sr: int,
    pred_intervals: Sequence[Interval] | None = None,
    mask: np.ndarray | None = None,
    frame_rate: int = 100,
) -> dict[str, float]:
    n = len(x)
    E = region_mask(gt_intervals, n, sr)
    out_reg = ~E
    r = {
        "sdr_bg": sdr(y, b),
        "sisdr_bg": min(SDR_CAP_DB, si_sdr(y, b)),
        "sisdr_mix": min(SDR_CAP_DB, si_sdr(x, b)),
        "sdr_bg_in": _region_sdr(y, b, E, sr),
        "sdr_bg_out": _region_sdr(y, b, out_reg, sr),
        "mse_in": float(np.mean((y[E] - b[E]) ** 2)) if E.any() else float("nan"),
        "mse_out": float(np.mean((y[out_reg] - b[out_reg]) ** 2)) if out_reg.any() else float("nan"),
        "target_suppression_db": target_suppression_db(y, s, b, E),
    }
    r["sisdri"] = r["sisdr_bg"] - r["sisdr_mix"]
    r["lsd_in"] = log_spectral_distance(y, b, sr, E)
    r["lsd_out"] = log_spectral_distance(y, b, sr, out_reg)
    if pred_intervals is not None:
        r.update(frame_metrics(pred_intervals, gt_intervals, n / sr, frame_rate))
        r.update(boundary_errors(pred_intervals, gt_intervals))
    if mask is not None:
        r["edge_click_db"] = edge_click_db(y - b, mask, sr)
    return r


def _bootstrap_ci(v: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(df: pd.DataFrame, metrics: Sequence[str], by: Sequence[str],
              n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean and 95% bootstrap CI per group; NaNs are excluded and counted."""
    rows = []
    for keys, g in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        for m in metrics:
            if m not in g:
                continue
            v = g[m].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            lo, hi = _bootstrap_ci(v, n_boot, seed)
            row.update({f"{m}_mean": float(v.mean()) if len(v) else float("nan"),
                        f"{m}_ci_low": lo, f"{m}_ci_high": hi, f"{m}_n": int(len(v))})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(df: pd.DataFrame, metric: str, a: str, b: str,
                key: str = "sample_id", method_col: str = "method") -> dict[str, float]:
    """Wilcoxon signed-rank test on per-sample differences (b - a)."""
    wide = df[df[method_col].isin([a, b])].pivot_table(index=key, columns=method_col, values=metric)
    wide = wide.dropna()
    nan = float("nan")
    if len(wide) < 2 or a not in wide or b not in wide:
        return {"metric": metric, "a": a, "b": b, "n": int(len(wide)), "mean_diff": nan, "median_diff": nan, "p_value": nan}
    d = (wide[b] - wide[a]).to_numpy()
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"metric": metric, "a": a, "b": b, "n": int(len(d)), "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)), "p_value": p}
