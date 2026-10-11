# SONIC: Temporal Localization for Instruction-Guided Sound Removal

*Research note, draft. Every quantitative claim below is a placeholder of the form
`[[RESULT PENDING: …]]` until it is replaced with numbers from
`results/main/eval/summary.md` and `results/ablations/summary.md`, produced by
`notebooks/sonic_colab.ipynb`. No result in this document has been estimated or invented.*

## 1. Abstract

Text-queried source separators can remove a described sound from a recording by
subtracting their estimate of it. Because the estimate also contains leaked background
energy, global subtraction damages the background everywhere, including where the target
is silent. We study whether **explicit temporal localization** limits this damage: SONIC
scores overlapping windows with CLAP text–audio similarity against the target description
(contrasted with generic background prompts), converts the scores into event intervals, and
subtracts the AudioSep estimate only inside those intervals, using raised-cosine crossfades.
On controlled synthetic mixtures built from ESC-50 and LibriSpeech, with known stems and
event boundaries, we compare global removal, SONIC, and an oracle mask under an identical
separation output. [[RESULT PENDING: one-sentence summary of the eval-split comparison,
e.g. change in out-of-event SDR and in-event suppression, with CIs and p-values.]]

## 2. Problem formulation

A mixture `x = s + b` contains a target sound `s`, described by an instruction, and a
background `b`. Given `x` and an instruction such as "Remove the dog barking", the goal is
an output `y ≈ b`. A separator yields `ŝ = f(x, text)`. The global edit is `y = x − ŝ`, and
the localized edit is `y = x − m ⊙ ŝ` with a temporal mask `m(t) ∈ [0, 1]`.

Writing `ŝ = s + e`, the global error `y − b = −e` spans the whole clip, while the localized
error is `(1 − m)s − m e`. Wherever the target is silent and `m = 0`, the error is exactly
zero. Localization therefore trades **missed target** (`m < 1` where `s ≠ 0`) against
**avoided leakage** (`m = 0` where `s = 0`).

**Hypothesis H1.** With a localization mask from CLAP window scores, background
preservation outside target events improves over global removal, while target suppression
inside events is not significantly reduced.

**Hypothesis H2.** The gap between SONIC and the oracle mask measures how much of the
attainable benefit is lost to localization errors.

## 3. Related approaches

* **Language-queried audio source separation.** LASS-Net (Liu et al., Interspeech 2022)
  and AudioSep (Liu et al., 2023) separate a sound described in text. SONIC uses AudioSep
  unchanged as its separator.
* **Contrastive language–audio pretraining.** CLAP (Wu et al., ICASSP 2023) embeds text
  and audio in a shared space; AudioSep uses it as its query encoder. SONIC reuses the same
  encoder for window-level localization.
* **Instruction-based audio editing with generative models.** AUDIT (Wang et al.,
  NeurIPS 2023) edits audio by following instructions with a latent diffusion model.
  Regenerating the whole signal can alter regions that should stay untouched. SONIC instead
  edits by masked subtraction, which leaves samples outside the mask exactly unchanged.

## 4. Proposed methodology

**Pretrained components (not contributions):** AudioSep (ResUNet30 separator,
`audiosep_base_4M_steps.ckpt`) and its LAION-CLAP HTSAT-base query encoder
(`music_speech_audioset_epoch_15_esc_89.98.pt`).

**Original engineering contributions:**

1. *Instruction parsing.* A rule-based parser extracts the target phrase from removal
   instructions. Unsupported instructions are rejected.
2. *Contrastive window scoring.* For windows `a_w` of length `W` and hop `H`:
   `score_w = cos(a_w, t_target) − max_k cos(a_w, t_bg,k)`, with generic background prompts
   (e.g. "ambient background noise", "people talking"). This reduces the bias of raw CLAP
   similarity toward whatever dominates the scene.
3. *Interval estimation.* Per-clip z-scoring; averaging all windows that cover each 10 ms
   frame; a threshold; median filtering; filling gaps < 0.25 s; dropping events < 0.1 s;
   padding by 0.1 s.
4. *Gated reconstruction.* `m(t)` equals 1 inside the intervals, with raised-cosine ramps
   placed outside them (the crossfade), and `y = x − m ⊙ ŝ`. Samples with `m = 0` are copied
   bit-exactly. A soft variant, a sigmoid of the frame score, is evaluated as an ablation.
5. *Fair comparison by construction.* Baseline, SONIC and the oracle subtract the **same
   cached** AudioSep estimate, so differences come only from the mask.

## 5. Experimental setup

* **Data.** 10 s, 32 kHz synthetic mixtures in three scenarios: `dog_env` (dog barking
  over an environmental bed plus speech), `short_event` (one short event over a continuous
  bed), and `intermittent` (2–3 occurrences of one event). Event SNR is uniform in [−5, 5]
  dB. The dev split (ESC-50 folds 1–4, 10 clips per scenario) is used only for ablations and
  parameter selection. The eval split (fold 5, 20 clips per scenario, disjoint LibriSpeech
  speakers) is used once, with the selected configuration.
* **Metrics.** Scale-dependent SDR of `y` against `b` on the whole clip, inside target
  regions and outside them (capped at 60 dB); SI-SDR and SI-SDR improvement; target
  suppression `10 log10(‖s_E‖²/‖(y−b)_E‖²)`; frame-level localization P/R/F1/IoU and boundary
  errors; a high-frequency click metric at mask edges; latency and peak GPU memory. CLAP
  similarity is reported only as an auxiliary signal. Aggregates are means with 95%
  bootstrap CIs. Method differences use paired Wilcoxon signed-rank tests.
* **Hardware.** [[RESULT PENDING: GPU model and memory reported by the notebook's GPU
  check cell; torch version.]]

## 6. Quantitative results

[[RESULT PENDING: table from `results/main/eval/summary.md`, "Reconstruction quality
(overall)": sdr_bg, sisdr_bg, sisdri, sdr_bg_out, sdr_bg_in, target_suppression_db for
baseline / SONIC / oracle.]]

[[RESULT PENDING: per-scenario table and figure `quality_by_scenario.png`.]]

[[RESULT PENDING: localization table (SONIC vs oracle) and `localization_by_scenario.png`.]]

[[RESULT PENDING: paired tests from `stats.csv`; state explicitly whether H1 is supported
for sdr_bg_out and whether target_suppression_db differs significantly.]]

[[RESULT PENDING: cost table, latency and peak GPU memory for baseline vs SONIC.]]

## 7. Ablation study (dev split)

One factor varied at a time around the default (W = 1 s, H = 0.25 s, threshold 0.5,
crossfade 50 ms, hard mask):

| Factor | Values |
|---|---|
| Window size W | 0.5, 1, 2 s (75% overlap) |
| Threshold (z-units) | −0.5, 0, 0.5, 1.0 |
| Crossfade | 0, 10, 50, 200 ms |
| Mask | hard, soft |

[[RESULT PENDING: `results/ablations/summary.md` and the four `ablation_*.png` figures;
selected configuration from `tuned_overrides.txt`; comment on the crossfade-vs-click
trade-off using edge_click_db.]]

## 8. Limitations

* Removal is the only edit type; the parser is rule-based.
* Synthetic, additive, anechoic mixtures; real recordings with reverberation and
  overlapping events are not evaluated.
* The CLAP checkpoint was trained/evaluated with ESC-50 data, so localization of ESC-50
  sources may be optimistic.
* Per-clip normalization presumes the target is present. Clips without the target are not
  evaluated.
* Ground-truth boundaries are those of the trimmed event clips and include internal pauses,
  which slightly penalizes precise detectors.
* A single separator is used. The benefit of gating depends on how much AudioSep leaks
  background, which may differ for other separators.

## 9. Future work

* Learnable or calibrated localization, e.g. thresholds calibrated across prompts, or a
  lightweight frame-level text-conditioned detector.
* Separating within localized segments (instead of gating a global estimate), and joint
  separation–localization.
* Amplification and replacement edits; real recordings with human listening tests
  alongside reference-based metrics.

## Summary of claims

| Type | Statement | Status |
|---|---|---|
| Pretrained | AudioSep separator; LAION-CLAP encoder | third-party, used unchanged |
| Engineering | contrastive CLAP window scoring, interval post-processing, gated crossfaded subtraction, evaluation protocol, dataset generator | implemented, unit-tested (with test doubles) |
| Hypothesis | H1, H2 above | [[RESULT PENDING]] |
| Validated finding | none yet | requires the Colab run |

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
