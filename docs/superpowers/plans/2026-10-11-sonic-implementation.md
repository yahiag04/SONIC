# SONIC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SONIC prototype described in
`docs/superpowers/specs/2026-10-11-sonic-design.md`: AudioSep-based
instruction-guided sound removal with CLAP temporal localization, a global
baseline, an oracle condition, an evaluation framework, a Colab notebook, and a
research note.

**Architecture:** Pure-numpy core modules (`audio`, `instruction`,
`localization`, `editing`, `evaluation`) that are testable without models; a
thin `separation` wrapper around AudioSep/CLAP that is imported only on Colab;
scripts that cache AudioSep estimates once per sample and apply baseline,
SONIC, and oracle masks to the same estimate.

**Tech Stack:** Python ≥3.10, numpy, scipy, soundfile, pandas, matplotlib,
PyYAML, pytest; on Colab: torch, torchaudio, AudioSep (cloned), transformers.

Local constraint: no model weights locally. Model code paths are verified on
Colab by the user.

---

## File map

| File | Responsibility |
|---|---|
| `sonic/audio.py` | I/O, resample (scipy `resample_poly`), normalization, SNR mixing, clipping guard, framing, ramps |
| `sonic/instruction.py` | `EditInstruction`, `parse_instruction`, `UnsupportedInstructionError` |
| `sonic/localization.py` | window scores, frame curve, post-processing, interval/mask utils, `localize` |
| `sonic/editing.py` | `global_remove`, `masked_remove` |
| `sonic/evaluation.py` | metrics, `evaluate_sample`, `aggregate`, `paired_test` |
| `sonic/separation.py` | `Separator`/`Embedder` protocols, `AudioSepModel` |
| `sonic/config.py` | `SonicConfig` dataclass + YAML load + overrides |
| `sonic/pipeline.py` | `build_mask`, `edit_with_estimate`, `edit` (end-to-end, timed) |
| `sonic/dataset.py` | scenario sampling + mixture synthesis (pure, testable) |
| `sonic/plotting.py` | localization figure, bar/ablation plots |
| `scripts/*.py` | CLI entry points (fetch models, prepare dataset, run methods, ablations, evaluate, figures) |
| `tests/*.py` | unit tests; `tests/fakes.py` holds `FakeEmbedder`, `FakeSeparator` |

## Tasks

### Task 0: Scaffold
- [ ] `pyproject.toml` (package `sonic`, deps, pytest config), `requirements.txt`, `requirements-colab.txt`, `.gitignore` (data/, models/, results/*, *.wav except none, .venv), `configs/default.yaml`, empty `sonic/__init__.py`.
- [ ] Local venv: `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'`.
- [ ] Commit.

### Task 1: `audio.py` (TDD)
Interface:
```python
SR = 32000
load_audio(path, sr=SR) -> np.ndarray  # mono float32
save_audio(path, x, sr=SR) -> None
resample(x, orig_sr, new_sr) -> np.ndarray
rms(x) -> float
mix_at_snr(signal, noise, snr_db) -> np.ndarray   # returns scaled `signal`
fit_peak(stems: dict[str, np.ndarray], peak=0.9) -> dict  # same gain for all, so sums stay consistent
frame_starts(n, win, hop) -> np.ndarray  # last window covers the end
raised_cosine(n) -> np.ndarray  # 0->1, length n
```
Tests: resample length ratio; `mix_at_snr` gives 10·log10(rms²) diff == snr within 1e-6; `fit_peak` keeps sum peak ≤ 0.9 and ratios; frame_starts covers the signal; raised_cosine endpoints and monotonicity; save/load round trip.

### Task 2: `instruction.py` (TDD)
Tests (parametrized): "Remove the dog barking while preserving everything else." → target "dog barking"; "Get rid of all the coughing" → "coughing"; "mute the siren from the recording" → "siren"; "Please delete the door knocking, but keep the rain" → "door knocking"; "Amplify the speech" → raises; "Remove" → raises.

### Task 3: `editing.py` (TDD)
Tests: mask 0 → output identical (`np.array_equal`); mask 1 equals global; perfect estimate gives exact background; shape mismatch raises.

### Task 4: `localization.py` (TDD)
Interface:
```python
Interval = tuple[float, float]
window_scores(x, sr, target, embedder, win_s, hop_s, background_prompts) -> (starts_s, scores)
normalize_scores(scores, method) -> np.ndarray
frame_curve(starts_s, scores, win_s, duration_s, frame_rate) -> (times, curve)  # overlap-averaged
binarize(curve, threshold) -> bool array
median_smooth(active, kernel) -> bool array
active_to_intervals(active, frame_rate) -> list[Interval]
postprocess(intervals, min_event_s, min_gap_s, pad_s, duration_s) -> list[Interval]
intervals_to_mask(intervals, n, sr, crossfade_s) -> np.ndarray  # ramps outside intervals
soft_mask(times, curve, n, sr, threshold, temperature) -> np.ndarray
localize(x, sr, target, embedder, cfg) -> LocalizationResult
```
Tests: FakeEmbedder (energy-based) on silence+burst finds the burst within one hop; median_smooth removes an isolated frame; gap fill merges; min_event drops short; pad clips to bounds; mask is 1 inside, 0 beyond crossfade, monotone ramps, crossfade 0 → step; soft mask in [0,1].

### Task 5: `evaluation.py` (TDD)
Interface: `si_sdr(est, ref)`, `region_mask_from_intervals`, `target_suppression_db`, `frame_metrics(pred, gt, duration, frame_rate)`, `boundary_errors`, `boundary_hf_energy`, `evaluate_sample(...) -> dict`, `aggregate(df, metrics, by) -> df`, `paired_test(df, metric, a, b) -> dict`.
Tests: si_sdr identical > 100 dB and scale invariant; known noise SNR; suppression of a 0.1-scaled residual = 20 dB; frame P/R/IoU on hand intervals; aggregate CI contains the mean.

### Task 6: `dataset.py` (TDD with synthetic stems)
Interface: `SCENARIOS`, `place_events(event, n_events, length, rng, min_gap)`, `synthesize_sample(bg, event, speech, n_events, snr_db, rng, sr, length_s) -> dict(mixture, target, background, stems, intervals)`, `trim_active(x, sr, top_db)`, `make_instruction(target_label, rng)`.
Tests: mixture == target + background exactly; peak ≤ 0.9; intervals match where target ≠ 0; events don't overlap; deterministic for a seed.

### Task 7: `config.py`, `separation.py`, `pipeline.py`
- Config dataclass with nested sections (`localization`, `editing`, `separation`, `paths`); `load_config(path, overrides)`.
- `AudioSepModel(audiosep_dir, checkpoint_dir, device, use_chunk)`; `separate`, `embed_text`, `embed_audio(windows, batch_size)`.
- `pipeline.build_mask(method, ...)`, `edit_with_estimate`, `edit` with timing.
- Tests: config load + override; pipeline with FakeSeparator/FakeEmbedder for all 3 methods (oracle needs intervals; sonic outside-mask samples untouched).

### Task 8: Scripts
`fetch_models.py` (clone AudioSep at a pinned commit, hf_hub_download the 2 checkpoints, symlink into `AudioSep/checkpoint/`), `prepare_dataset.py` (download ESC-50 zip + LibriSpeech dev-clean via URL, build splits), `run_separation.py` (cache ŝ + CLAP window scores per sample, with latency/memory), `run_baseline.py` / `run_sonic.py` (apply masks from cache, or single-file mode end-to-end), `run_ablations.py`, `evaluate.py`, `make_figures.py`, `make_examples.py`.
Smoke test locally: a `tests/test_scripts.py` that runs prepare on synthetic stems → separation with fakes via `--fake` is NOT allowed in scripts (honesty rule); instead test the script-level functions with fakes imported in the tests.

### Task 9: Notebook
`notebooks/sonic_colab.ipynb` generated by `scripts/build_notebook.py` (nbformat) so it stays reviewable.

### Task 10: README + research note (with placeholders), push to GitHub.

## Self-review
Spec coverage: modules (T1–T7), dataset (T6, T8), baseline/SONIC/oracle (T7–T8), metrics (T5), ablations (T8), plots (T8 + plotting), notebook (T9), report/README (T10). Out of scope: amplification.
