# SONIC — Design

Date: 2026-10-11
Source requirements: `SONIC_AGENT_SPEC.md` (project specification).

## 1. Goal and research question

Instruction-guided **selective sound removal**: given an audio clip and an
instruction such as *"Remove the dog barking while preserving everything
else"*, output the clip with the target removed.

Research question: *Does explicit temporal localization improve background
preservation without sacrificing target removal?*

Hypothesis (H1): text-conditioned separators leak background energy into the
target estimate, including in regions where the target is silent. Subtracting
the estimate only inside detected target intervals removes that damage
outside the intervals, while leaving removal quality inside the intervals
unchanged.

## 2. Execution constraints

- Real model inference and all reported experiments run on **Google Colab
  (T4)**, executed by the user. Locally only unit tests run (with clearly
  named fakes); no model weights are downloaded locally.
- Repository: `https://github.com/yahiag04/SONIC.git` (public). The notebook
  clones it without credentials.
- No model checkpoints or third-party audio are committed.
- Python: code targets Python ≥ 3.10 (spec says 3.11; current Colab images
  ship a newer 3.x, so we do not pin a minor version).

## 3. Pretrained components (not original contributions)

| Component | Use | License | Notes |
|---|---|---|---|
| AudioSep (`Audio-AGI/AudioSep`, ResUNet30, `audiosep_base_4M_steps.ckpt`) | Text-queried source separation | MIT (code); weights from HF `audo/AudioSep` | 32 kHz mono. Code is cloned at runtime, not vendored. |
| LAION CLAP HTSAT-base (`music_speech_audioset_epoch_15_esc_89.98.pt`) | AudioSep's query encoder; reused by SONIC for text–audio window similarity | CC0-1.0 (LAION-AI/CLAP repo) | Accessed through AudioSep's `CLAP_Encoder` (`_get_text_embed`, `_get_audio_embed`). Short windows are repeat-padded to 10 s internally. Requires `roberta-base` tokenizer from HF. |

Using AudioSep's own CLAP for localization means no extra weights, a single
shared embedding space, and lower peak memory.

Known caveat: CLAP was trained on ESC-50-adjacent data (the checkpoint name
reports ESC-50 accuracy). Localization numbers on ESC-50 sources may therefore
be optimistic; this is documented as a limitation.

## 4. Architecture

```
audio + instruction
  → instruction.parse()            → EditInstruction(action, target)
  → separation.separate(x, target) → ŝ  (AudioSep, once per clip, cached)
  → localization.localize(x, target, embedder) → intervals, scores
  → editing.{global_remove | localized_remove}(x, ŝ, mask)
  → y
```

### Modules (`sonic/`)

- **`audio.py`**: load/resample to 32 kHz mono (soundfile + torchaudio/scipy
  resample), save, peak and RMS normalization, `mix_at_snr`, clipping guard
  (rescale the whole mixture *and all stems* by the same factor so
  mixture = Σ stems still holds exactly), `frame_signal(x, win, hop)` →
  window start indices, `raised_cosine_ramp(n)`.
- **`instruction.py`**: rule-based parser producing
  `EditInstruction(action: Literal["remove"], target: str, raw: str)`.
  Verbs: remove, delete, eliminate, mute, erase, cut, get rid of, take out,
  suppress. Strips leading determiners ("the", "all", "any") and trailing
  clauses ("while …", "but keep …", "and keep …", "from the recording",
  punctuation). Unsupported instructions (e.g. "amplify", or no target)
  raise `UnsupportedInstructionError`. No LLM is used; this is a deliberate
  scope decision.
- **`separation.py`**:
  - `Separator` protocol: `separate(wav: np.ndarray, text: str) -> np.ndarray`.
  - `Embedder` protocol: `embed_text(texts) -> (N,D)`, `embed_audio(windows) -> (M,D)`
    (L2-normalized).
  - `AudioSepModel`: loads AudioSep once from a given repo dir + checkpoint
    dir (handles the relative-path requirement by temporarily `chdir`-ing
    into the AudioSep repo during construction), implements both protocols,
    optional chunked inference for clips > 10 s, batched audio embedding.
    Failure to load raises; there is no fallback.
- **`localization.py`**:
  - `window_scores(x, sr, target, embedder, win_s, hop_s, background_prompts)`:
    per-window score = cos(a, t_target) − max_k cos(a, t_bg_k).
  - Per-clip normalization: z-score (configurable: `zscore` | `minmax` | `none`).
  - `scores_to_frame_curve`: overlap-averaged score at a fixed frame rate
    (e.g. 100 Hz) so window size and frame resolution are decoupled.
  - Post-processing: threshold → median filter (odd kernel, in frames) →
    remove segments shorter than `min_event_s` → fill gaps shorter than
    `min_gap_s` → dilate by `pad_s` on both sides → clip to signal bounds.
  - Returns `LocalizationResult(intervals: list[(t0, t1)], frame_times,
    frame_scores, threshold)`.
  - `intervals_to_mask(intervals, n_samples, sr, crossfade_s)`: hard mask
    with raised-cosine ramps *outside* each interval (so the full interval
    has weight 1); overlapping ramps are merged with `max`.
  - `soft_mask(frame_scores, ..., temperature)`: sigmoid mask (ablation).
- **`editing.py`**:
  - `global_remove(x, s_hat) = x − s_hat`.
  - `masked_remove(x, s_hat, mask) = x − mask · s_hat`. Wherever `mask == 0`
    the output equals the input exactly.
- **`evaluation.py`**: metrics (Section 6), per-sample records, aggregation
  with bootstrap CIs, paired Wilcoxon test.
- **`pipeline.py`**: `edit(x, instruction, method, cfg, model, oracle_intervals=None)`
  where `method ∈ {baseline, sonic, oracle}`. Times each stage
  (`time.perf_counter` with `torch.cuda.synchronize` when on CUDA) and
  records `torch.cuda.max_memory_allocated`.
- **`config.py`**: load `configs/default.yaml` into a dataclass; CLI
  overrides via `key=value`.

### Fairness

The three methods share **one AudioSep estimate ŝ per sample** (same model,
same prompt, same input). They differ only in the mask: baseline uses
m ≡ 1, SONIC uses the CLAP mask, oracle uses the ground-truth intervals
with the same padding and crossfade. Estimates are cached to disk so ablations
re-use them.

## 5. Dataset (`scripts/prepare_dataset.py`)

- Sources, downloaded inside Colab:
  - ESC-50 (`karolpiczak/ESC-50`, CC BY-NC 3.0 per its README), 5 s clips, 44.1 kHz.
  - LibriSpeech dev-clean (CC BY 4.0) for speech.
- Splits: ESC-50 folds 1–4 → **dev** (tuning and ablations); fold 5 →
  **eval** (reported results only). LibriSpeech speakers are split
  disjointly the same way.
- Each sample is 10 s at 32 kHz:
  - Background bed: a continuous class (rain, wind, crackling fire, sea
    waves, crickets, engine, vacuum cleaner, washing machine), with up to three
    5 s clips concatenated with 50 ms crossfades.
  - Optional speech (scenario-dependent), RMS-scaled.
  - Target event(s): the event clip is trimmed to its active region
    (energy-based, top-dB trim) and placed at random non-overlapping onsets;
    the event SNR relative to background+speech is sampled uniformly in
    [−5, 5] dB.
- Scenarios (20 eval + 10 dev each):
  1. `dog_env`: target "dog barking"; background = environmental bed + speech.
  2. `short_event`: one short event (door knock, glass breaking, cough,
     clock alarm, can opening …) over a continuous bed, no speech.
  3. `intermittent`: the same event class occurring 2–3 times.
- Ground-truth intervals come from the trimmed event boundaries at the
  placed offsets.
- Stored per sample (`data/<split>/<id>/`): `mixture.wav`, `target.wav`,
  `background.wav` (the reference edited audio = everything except the
  target), `stems/*.wav`, `meta.json` (instruction, target prompt, intervals,
  SNR, source filenames, seed). A `manifest.csv` per split.
- Instruction templates vary in wording (e.g. "Remove the dog barking while
  preserving everything else", "Get rid of the knocking on the door",
  "Mute the coughing").
- Deterministic given `--seed`. Peak-normalized to ≤ 0.9 with consistent
  stem scaling. No audio is committed. `examples/` gets the three demo
  samples at runtime.

## 6. Evaluation (`sonic/evaluation.py`, `scripts/evaluate.py`)

Notation: x = mixture, s = true target, b = reference background (x = s + b),
y = output, E = event region (union of GT intervals), Ē = complement.

| Metric | Definition | Measures |
|---|---|---|
| `sdr_bg` | 10·log10(‖b‖²/‖y−b‖²), whole clip (scale-dependent) | overall quality; **ablation selection criterion** |
| `sisdr_bg` | SI-SDR(y, b) on the whole clip | overall quality, gain-insensitive |
| `sisdri` | SI-SDR(y, b) − SI-SDR(x, b) | improvement over doing nothing |
| `sdr_bg_out` / `mse_out` | SDR / MSE on Ē only | background preservation (H1 main) |
| `sdr_bg_in` | SDR on E only | quality where editing happens |
| `target_suppression_db` | 10·log10(‖s_E‖² / ‖(y−b)_E‖²) | removal effectiveness |
| `loc_precision/recall/f1`, `loc_iou` | frame-level vs GT at 100 Hz | localization |
| `onset_err_s`, `offset_err_s` | mean abs boundary error of matched intervals | localization |
| `boundary_hf_ratio_db` | high-pass (> 4 kHz) energy of (y − b) in ±20 ms around mask edges, relative to the same in the baseline | discontinuities |
| `latency_*_s`, `peak_gpu_mb` | per stage | cost |
| `clap_target_sim_drop` | cos(CLAP(x), t) − cos(CLAP(y), t) | auxiliary only |

Regional metrics are scale-dependent SDR, not SI-SDR: subtracting a leaky
estimate that is a scaled copy of the background (y = 0.95·b) is invisible to
SI-SDR but is exactly the damage H1 is about. All SDR/SI-SDR values are capped
at 60 dB so perfectly preserved regions do not dominate the means. Regions
shorter than 0.25 s are skipped (NaN, excluded from aggregation, counts
reported). (Revised during implementation; the first draft used regional SI-SDR.)

Aggregation: mean ± 95% bootstrap CI (1000 resamples, fixed seed) per
scenario and overall; paired Wilcoxon signed-rank (scipy) on per-sample
differences, SONIC vs baseline and oracle vs baseline, reporting the p-value
and the median difference. Results are written as `results/per_sample.csv`,
`results/summary.csv`, `results/summary.md`.

## 7. Experiments

- **Main (eval split):** baseline, SONIC (CLAP mask), oracle mask, all on
  the shared cached ŝ.
- **Ablations (dev split, cached ŝ, so cheap):**
  - window ∈ {0.5, 1.0, 2.0} s at 75% overlap
  - threshold ∈ {−0.5, 0, 0.5, 1.0} (z-score units)
  - crossfade ∈ {0, 10, 50, 200} ms
  - hard vs soft mask
  Each is varied one factor at a time around the default.
- **Protocol:** pick the dev-best setting by `sdr_bg` (tie-break:
  `loc_f1`), write it into `configs/default.yaml`, then run eval once.
- **Plots** (`results/figures/`): main bar chart with CIs, ablation line
  plots, and a localization figure per demo example (spectrogram; score
  curve with threshold; predicted vs GT intervals; mask).

## 8. Repository layout

The layout from spec §11 is followed, plus `sonic/pipeline.py`,
`sonic/config.py`, `docs/`, and `scripts/fetch_models.py` (clones AudioSep
at a pinned commit and downloads the checkpoints from HF into
`WEIGHTS_DIR`). `scripts/run_ablations.py` and `scripts/make_figures.py` are
added for experiments and plots.

Scripts:
- `prepare_dataset.py --data-root --seed`
- `fetch_models.py --models-dir`
- `run_baseline.py` / `run_sonic.py`: run one method over a split, or
  `--audio --instruction` for a single file; write outputs + per-sample JSON.
- `run_ablations.py`
- `evaluate.py`: compute metrics, tables, statistics.
- `make_figures.py`

## 9. Colab notebook

Steps: config cell (`REPO_URL`, `WORK_DIR`, `DATA_DIR`, `MODELS_DIR`,
`RESULTS_DIR`, optional Drive mount) → clone/pull → `pip install -r
requirements.txt` + AudioSep deps → GPU check → fetch models → prepare
dataset → demo: parse the instruction, run baseline + SONIC, `IPython.display.Audio`
for original/baseline/SONIC/reference, localization plot → full eval →
ablations → figures → zip results (copy to Drive if mounted).

## 10. Testing

pytest, runnable locally without model weights:
- `test_audio.py`: resample length, `mix_at_snr` achieves the target SNR,
  the clipping guard preserves mixture = Σ stems, framing indices.
- `test_instruction.py`: verb/target variants, determiners and trailing
  clauses, unsupported → error.
- `test_localization.py`: `FakeEmbedder` (an energy detector disguised as
  embeddings) on a synthetic signal with a known burst → correct interval;
  median filter, min-duration, gap fill, dilation; mask properties (values
  in [0,1], exactly 1 inside intervals, exactly 0 far outside, ramps
  monotone, crossfade = 0 gives a step).
- `test_editing.py`: `masked_remove` keeps samples with mask 0 bit-identical;
  mask ≡ 1 equals `global_remove`; perfect ŝ gives perfect removal.
- `test_evaluation.py`: SI-SDR of identical signals is very large, and is
  scale-invariant; suppression with a known residual; frame P/R/IoU on
  hand-built intervals.
- `test_dataset.py`: the generator on synthetic tone "stems" (no ESC-50
  needed) checks no clipping, mixture = target + background, and that the
  meta intervals match where the target is non-zero.

Fakes are named `Fake*` and live in `tests/`; nothing in `sonic/` or the
scripts uses them.

## 11. Honesty rules

- `reports/research_note.md` is written with explicit
  `[[RESULT PENDING: …]]` placeholders until real Colab results are
  returned; no numbers are invented.
- The README states which parts are pretrained (AudioSep, CLAP) and which
  are original (localization-gated editing, evaluation protocol, dataset
  generator).

## 12. Out of scope

Amplification and replacement (optional per spec, not started before the
primary experiments are complete), LLM-based instruction parsing, training
or fine-tuning any model.
