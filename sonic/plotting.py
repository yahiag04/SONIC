"""Figures: localization visualisation, method comparison, ablation curves."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_COLORS = {"baseline": "#7f7f7f", "sonic": "#1f77b4", "oracle": "#2ca02c"}


def plot_localization(
    x: np.ndarray,
    sr: int,
    frame_times: np.ndarray,
    frame_scores: np.ndarray,
    threshold: float,
    pred_intervals: Sequence[tuple[float, float]],
    gt_intervals: Sequence[tuple[float, float]],
    mask: np.ndarray | None = None,
    title: str = "",
    path: str | Path | None = None,
):
    t_end = len(x) / sr
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.4, 0.8]})
    ax = axes[0]
    ax.specgram(x, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", scale="dB", vmin=-120)
    ax.set_ylabel("Hz")
    ax.set_ylim(0, sr / 2)
    ax.set_title(title)

    ax = axes[1]
    ax.plot(frame_times, frame_scores, color="#1f77b4", lw=1.3, label="normalized CLAP score")
    ax.axhline(threshold, color="k", ls="--", lw=1, label=f"threshold = {threshold:g}")
    for t0, t1 in gt_intervals:
        ax.axvspan(t0, t1, color="#2ca02c", alpha=0.15)
    ax.set_ylabel("score")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    for t0, t1 in gt_intervals:
        ax.barh(1, t1 - t0, left=t0, height=0.6, color="#2ca02c")
    for t0, t1 in pred_intervals:
        ax.barh(0, t1 - t0, left=t0, height=0.6, color="#1f77b4")
    if mask is not None:
        tt = np.arange(len(mask)) / sr
        step = max(1, len(mask) // 4000)
        ax.plot(tt[::step], mask[::step] * 0.6 - 0.3, color="#d62728", lw=0.8, label="edit mask")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_yticks([0, 1], ["SONIC", "ground truth"])
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, t_end)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_method_bars(summary: pd.DataFrame, metrics: Sequence[str], path: str | Path,
                     group_col: str = "scenario"):
    """Grouped bars (mean ± 95% CI) of each metric per scenario and method."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False)
    groups = list(summary[group_col].unique())
    methods = [m for m in METHOD_COLORS if m in set(summary.method)]
    width = 0.8 / max(1, len(methods))
    for ax, metric in zip(axes[0], metrics):
        for j, m in enumerate(methods):
            sub = summary[summary.method == m].set_index(group_col).reindex(groups)
            mean = sub[f"{metric}_mean"].to_numpy()
            err = np.vstack([mean - sub[f"{metric}_ci_low"].to_numpy(), sub[f"{metric}_ci_high"].to_numpy() - mean])
            ax.bar(np.arange(len(groups)) + j * width, mean, width, yerr=err, capsize=3,
                   color=METHOD_COLORS[m], label=m)
        ax.set_xticks(np.arange(len(groups)) + width * (len(methods) - 1) / 2, groups, rotation=15)
        ax.set_title(metric)
        ax.axhline(0, color="k", lw=0.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ablation(df: pd.DataFrame, factor: str, metrics: Sequence[str], path: str | Path):
    """Mean ± 95% CI of metrics across the values of one ablation factor (dev split)."""
    sub = df[df.factor == factor].copy()
    numeric = factor != "mask"
    order = sorted(sub.value.unique(), key=float) if numeric else sorted(sub.value.unique())
    ref = df[df.factor == "reference"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        means, lo, hi = [], [], []
        for v in order:
            vals = sub.loc[sub.value == v, metric].dropna().to_numpy()
            means.append(vals.mean() if len(vals) else np.nan)
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            lo.append(1.96 * se)
            hi.append(1.96 * se)
        xs = np.arange(len(order))
        ax.errorbar(xs, means, yerr=[lo, hi], marker="o", capsize=3, color=METHOD_COLORS["sonic"], label="sonic")
        for m in ("baseline", "oracle"):
            if metric in ref and (ref.method == m).any():
                ax.axhline(ref.loc[ref.method == m, metric].mean(), color=METHOD_COLORS[m], ls="--", lw=1, label=m)
        ax.set_xticks(xs, order)
        ax.set_xlabel(factor)
        ax.set_title(metric)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
