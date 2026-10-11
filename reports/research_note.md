# SONIC: Temporal Localization for Instruction-Guided Sound Removal

*Research note. All numbers come from the Colab run whose outputs are committed in
`results/main/eval/` (held-out eval split, 60 clips) and `results/ablations/` (dev split,
30 clips). Values are means with 95% bootstrap confidence intervals; tests are paired
Wilcoxon signed-rank tests over clips.*

## 1. Abstract

Text-queried separators remove a described sound by subtracting their estimate of it, but
that estimate also contains leaked background, so global subtraction damages the background
even where the target is silent. SONIC adds explicit temporal localization: CLAP scores
overlapping 1 s windows against the target description (contrasted with generic background
prompts), the scores are turned into event intervals, and the AudioSep estimate is subtracted
only inside those intervals, with raised-cosine crossfades. All three conditions (global
baseline, SONIC, and an oracle mask from ground-truth boundaries) subtract the same AudioSep
output. On 60 held-out synthetic mixtures, SONIC raises out-of-event background SDR from 20.8
to 36.7 dB (p < 1e-10) and whole-clip SDR from 15.5 to 17.6 dB (p = 0.007). However, it
lowers in-event target suppression by 2.6 dB on average (p < 1e-4), because localization
misses parts of events. The hypothesis "better preservation **without** sacrificing editing
quality" is therefore only **partially supported**: it holds for single short events (no
measurable suppression loss) but not for intermittent events, where SONIC is worse than the
baseline overall. The oracle mask improves preservation without any suppression loss
(+5.4 dB whole-clip SDR), which identifies localization recall as the bottleneck.

## 2. Problem formulation

A mixture `x = s + b` contains a target `s`, named by an instruction, and a background `b`.
The goal is `y ≈ b`. A separator gives `ŝ = f(x, text) = s + e`, where `e` is the estimation
error. Global removal outputs `y = x − ŝ`, with error `−e` everywhere. Localized removal
outputs `y = x − m ⊙ ŝ` with a mask `m(t) ∈ [0, 1]`, with error `(1 − m)s − m e`. Where the
target is silent and `m = 0`, the error is exactly zero. Localization therefore trades
**avoided leakage** (`m = 0` where `s = 0`) against **missed target** (`m < 1` where `s ≠ 0`).

* **H1.** CLAP-based localization improves background preservation outside target events
  without significantly reducing target suppression inside them.
* **H2.** The SONIC–oracle gap measures the benefit lost to localization errors.

## 3. Related approaches

* **Language-queried source separation:** LASS-Net (Liu et al., 2022) and AudioSep (Liu et
  al., 2023). SONIC uses AudioSep unchanged as its separator.
* **Contrastive language–audio pretraining:** CLAP (Wu et al., 2023), which AudioSep uses
  as its query encoder. SONIC reuses this encoder for window-level localization.
* **Instruction-following generative editing:** AUDIT (Wang et al., 2023) regenerates audio
  with a latent diffusion model. SONIC instead edits by masked subtraction, so samples outside
  the mask stay bit-identical.

## 4. Methodology

**Pretrained components (not contributions):** the AudioSep ResUNet30 separator
(`audiosep_base_4M_steps.ckpt`) and its LAION-CLAP HTSAT-base encoder.

**Original engineering contributions:**

1. **Instruction parser.** A rule-based parser maps removal instructions to a target phrase
   and rejects anything else.
2. **Contrastive window score.** `cos(a_w, t_target) − max_k cos(a_w, t_bg,k)`, with four
   generic background prompts.
3. **Interval estimation.** Per-clip z-score; averaging all windows that cover each 10 ms
   frame; a threshold; a 0.25 s median filter; filling gaps < 0.25 s; dropping events
   < 0.1 s; padding by 0.1 s.
4. **Gated reconstruction.** `y = x − m ⊙ ŝ`, where `m = 1` inside the intervals, with
   raised-cosine ramps outside them. A soft sigmoid mask is tested as an ablation.
5. **Fair comparison by construction.** One cached AudioSep estimate per clip is shared by
   all methods.

The final configuration was selected on the dev split only: W = 1 s, hop 0.25 s, threshold
0.5 (z-units), crossfade 200 ms, hard mask.

## 5. Experimental setup

* **Data.** 10 s / 32 kHz additive mixtures from ESC-50 and LibriSpeech dev-clean, in three
  scenarios:
  * `dog_env`: dog barking over an environmental bed plus speech;
  * `short_event`: one short event over a continuous bed;
  * `intermittent`: 2–3 occurrences of one event.

  Event SNR is uniform in [−5, 5] dB. The dev split (ESC-50 folds 1–4, 10 clips per
  scenario) is used only for ablations and selection. The eval split (fold 5, 20 clips per
  scenario, disjoint speakers) was run once.
* **Metrics.** All are computed against the known background `b`:
  * scale-dependent SDR on the whole clip, outside events and inside events (`sdr_bg`,
    `sdr_bg_out`, `sdr_bg_in`, capped at 60 dB);
  * target suppression `10 log10(‖s_E‖²/‖(y−b)_E‖²)`;
  * frame-level localization P/R/F1/IoU (100 Hz) and boundary errors;
  * a high-frequency edge-click metric;
  * latency and peak GPU memory.

  Inside events, `sdr_bg_in` and suppression share the same residual, so their differences
  between methods are identical. CLAP similarity is reported only as an auxiliary signal.
* **Hardware.** Google Colab GPU runtime. Peak allocated GPU memory was about 2.3 GB, well
  within a T4's 16 GB. The GPU model was not recorded in this run's outputs; `evaluate.py`
  now writes `environment.json` for future runs.

## 6. Quantitative results (eval split, n = 60)

**Table 1. Overall.** SDR and suppression in dB.

| Method | sdr_bg | sdr_bg_out | sdr_bg_in | suppression | SI-SDRi |
|---|---|---|---|---|---|
| Baseline (global) | 15.48 [13.53, 17.53] | 20.78 [17.39, 24.21] | 13.12 [11.68, 14.57] | 13.74 [12.43, 15.32] | 7.72 [5.79, 9.70] |
| **SONIC** | **17.59** [15.57, 19.71] | **36.74** [33.30, 40.00] | 10.56 [9.01, 12.28] | 11.18 [9.52, 12.93] | **10.03** [8.35, 11.90] |
| Oracle mask | 20.93 [19.18, 22.65] | 39.33 [36.50, 42.15] | 13.12 [11.68, 14.57] | 13.74 [12.43, 15.32] | 13.41 [12.08, 14.95] |

**Paired tests, SONIC − baseline:**

| Metric | Mean difference | Median difference | p | Note |
|---|---|---|---|---|
| sdr_bg | +2.11 dB | +1.77 dB | 0.0072 | SONIC better on 44 of 60 clips |
| sdr_bg_out | +16.0 dB | | 1.6e-11 | |
| suppression | −2.56 dB | 0.00 dB | 1.4e-5 | loss > 1 dB on 23 of 60 clips |

**Oracle − baseline:** +5.45 dB `sdr_bg` (p = 1.6e-11), with no change in suppression.

**SONIC − oracle:** the gap is mostly *inside* events (suppression and `sdr_bg_in`
−2.56 dB). Outside events the remaining gap is not significant (+2.59 dB for the oracle,
p = 0.145).

**Table 2. By scenario,** SONIC − baseline (paired, n = 20 each).

| Scenario | Δ sdr_bg | Δ sdr_bg_out | Δ suppression | SONIC loc F1 / IoU | Event recall |
|---|---|---|---|---|---|
| dog_env | +5.09 (p = 0.004) | +19.6 (p = 1.9e-6) | −2.92 (median −0.51, p = 0.005) | 0.77 / 0.65 | 1.00 |
| short_event | +4.51 (p = 1.9e-6) | +14.9 (p = 1.9e-6) | +0.04 (p = 0.75, n.s.) | 0.61 / 0.46 | 1.00 |
| intermittent | **−3.26** (p = 0.033) | +13.4 (p = 1.9e-6) | −4.82 (p = 5e-4) | 0.67 / 0.52 | 0.85 |

**Localization behaviour by scenario:**
* `short_event`: frame recall is high (0.96) but precision is low (0.48). Over-long
  intervals are cheap, because subtracting the estimate a little beyond the event costs
  little, so suppression is fully kept.
* `intermittent`: recall is only 0.67, and 15% of events are missed entirely. Each missed
  event leaves the target in the output, which outweighs the preservation gain.
* Onset errors are 0.38–0.50 s, against 0.10 s for the oracle; the oracle's error equals its
  padding.

See `results/main/eval/figures/` and the three localization plots in `examples/`.

**Cost.** AudioSep takes 0.37 s per 10 s clip. CLAP localization at W = 1 s / H = 0.25 s
adds 0.84 s (37 windows, each repeat-padded to 10 s internally by CLAP), for 1.21 s in total
(3.3× the baseline). Peak GPU memory is 2346 MB vs 2270 MB.

**Discontinuities.** The edge-click metric shows no high-frequency excess at mask edges for
SONIC (−1.94 [−3.83, −0.35] dB) or the oracle (−2.55 [−3.98, −1.19] dB). No listening test
was run.

**Auxiliary CLAP metric.** The drop in CLAP similarity to the target text is 0.21 for the
baseline and 0.17 for SONIC. This is consistent with SONIC's lower suppression and is not
used as evidence of fidelity.

## 7. Ablation study (dev split, n = 30, one factor at a time)

| Factor | Values → `sdr_bg` (dB) | Observation |
|---|---|---|
| Threshold | −0.5: 13.91 · 0: 15.18 · **0.5: 15.62** · 1.0: 14.53 | Clear trade-off: higher thresholds raise `sdr_bg_out` (21.9 → 39.4) but lower `sdr_bg_in` (9.4 → 7.0). |
| Window | 0.5 s: 15.09 · **1 s: 15.62** · 2 s: 14.68 | 1 s also has the best localization F1 (0.72 vs 0.68 / 0.65). |
| Crossfade | 0 ms: 15.48 · 10 ms: 15.52 · 50 ms: 15.62 · **200 ms: 15.76** | Small, monotone gain, inside the CIs; the edge-click metric does not separate the settings (−0.67 to −1.04 dB, overlapping CIs). |
| Mask | **hard: 15.62** · soft: 14.34 | The sigmoid mask under-subtracts inside events (`sdr_bg_in` 7.67 vs 8.94). |

On dev, the baseline scores 12.80 dB and the oracle 16.99 dB. The selection protocol picked
the dev-best value per factor. Except for the crossfade, these values match the defaults,
and the differences between neighbouring values are mostly within the confidence intervals.

## 8. Limitations

* **Removal only; rule-based parser.**
* **Synthetic data.** Mixtures are additive and anechoic, with no real recordings and no
  clips where the target is absent. Per-clip z-scoring assumes the target is present.
* **CLAP and ESC-50 overlap.** The CLAP checkpoint was trained/evaluated with ESC-50, so
  localization may be optimistic. Even so, localization is the measured bottleneck.
* **Ground-truth boundaries** span whole trimmed clips, including pauses. This lowers SONIC's
  frame precision and caps the oracle's IoU below 1 (padding).
* **One run, one separator, one seed.** Results depend on how much AudioSep leaks.
* **Crossfade evidence is weak.** The click metric could not distinguish crossfade lengths,
  and no perceptual evaluation was done.

## 9. Future work

* **Improve recall on intermittent events.** Options: shorter hops; scoring against a
  separated-target track instead of the mixture; hysteresis thresholds; a minimum mask floor
  inside low-confidence regions.
* **Calibrated or learned frame-level detection** (e.g. a light text-conditioned SED head on
  frozen CLAP features).
* **Adaptive gating**: fall back to global removal when localization confidence is low,
  motivated by the intermittent scenario.
* **Real recordings, target-absent clips, listening tests;** amplification and replacement
  edits.

## Summary of claims

| Type | Statement | Evidence |
|---|---|---|
| Pretrained | AudioSep separator; LAION-CLAP encoder | third-party, used unchanged |
| Engineering | contrastive CLAP window scoring, interval post-processing, gated crossfaded subtraction, evaluation protocol, dataset generator | implemented; 88 unit tests |
| Validated | gating the same AudioSep estimate with CLAP intervals improves out-of-event preservation (+16.0 dB) and whole-clip SDR (+2.1 dB) | Table 1, p ≤ 0.007 |
| Validated | with ground-truth intervals, gating improves whole-clip SDR by 5.4 dB with no suppression loss | Table 1 |
| Not supported in general | "no loss of editing quality": suppression drops 2.6 dB on average, and SONIC is worse overall on intermittent events | Tables 1–2 |
| Supported for one scenario | no significant suppression loss for single short events | Table 2 |

## References

* X. Liu et al., "Separate What You Describe: Language-Queried Audio Source Separation,"
  Interspeech 2022.
* X. Liu et al., "Separate Anything You Describe," arXiv:2308.05037, 2023.
* Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, S. Dubnov, "Large-scale
  Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption
  Augmentation," ICASSP 2023.
* Y. Wang et al., "AUDIT: Audio Editing by Following Instructions with Latent Diffusion
  Models," NeurIPS 2023.
* K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM Multimedia 2015.
* V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on
  public domain audio books," ICASSP 2015.
* J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – half-baked or well done?,"
  ICASSP 2019.
