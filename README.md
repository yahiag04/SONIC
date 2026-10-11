# SONIC: Semantic and Temporal Control for Audio Editing

Instruction-guided **selective sound removal** with explicit **temporal localization**.

**🔊 [Listen to the demo](https://yahiag04.github.io/SONIC/)** · [Research note](reports/research_note.md) · [Colab](https://colab.research.google.com/github/yahiag04/SONIC/blob/main/notebooks/sonic_colab.ipynb)

> *"Remove the dog barking while preserving everything else."*
> → the dog is removed only where it barks; every other sample is left untouched.

**Research question.** Can explicit temporal localization improve background preservation
in instruction-guided audio editing without sacrificing editing quality?

**Status.** Implemented, unit-tested, and evaluated on Colab. Results are in
`results/main/eval/summary.md` (held-out eval split) and `results/ablations/summary.md`
(dev split). The write-up is in `reports/research_note.md`.

### Key results (eval split, 60 clips, mean [95% CI])

| Method | Whole-clip SDR | Background SDR outside events | Target suppression |
|---|---|---|---|
| Baseline (global removal) | 15.5 [13.5, 17.5] dB | 20.8 [17.4, 24.2] dB | 13.7 [12.4, 15.3] dB |
| **SONIC** (CLAP localization) | **17.6** [15.6, 19.7] dB | **36.7** [33.3, 40.0] dB | 11.2 [9.5, 12.9] dB |
| Oracle mask (ground-truth intervals) | 20.9 [19.2, 22.7] dB | 39.3 [36.5, 42.2] dB | 13.7 [12.4, 15.3] dB |

* Localization strongly improves background preservation (+16 dB outside events) and
  whole-clip SDR (+2.1 dB, p = 0.007).
* It is **not free**: localization misses lower target suppression by 2.6 dB on average
  (p < 1e-4). There is no loss for single short events, but on intermittent events SONIC is
  worse than the baseline overall.
* The oracle mask gains +5.4 dB with no suppression loss: localization recall is the
  bottleneck.
* Cost: 1.21 s vs 0.37 s per 10 s clip; about 2.3 GB peak GPU memory.

Demo audio and localization plots: `examples/` (three examples, not hand-picked).

[Open the notebook in Colab](https://colab.research.google.com/github/yahiag04/SONIC/blob/main/notebooks/sonic_colab.ipynb)

---

## What is pretrained and what is original

| | Component | Source |
|---|---|---|
| Pretrained (third-party) | **AudioSep** text-queried separation (ResUNet30, `audiosep_base_4M_steps.ckpt`) | Liu et al., 2023, [Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep), MIT |
| Pretrained (third-party) | **LAION-CLAP** HTSAT-base text/audio encoder (AudioSep's query encoder) | Wu et al., 2023, [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP), CC0-1.0 |
| Original engineering | CLAP-window **temporal localization** (contrastive target-vs-background score, per-clip normalization, post-processing) | this repo, `sonic/localization.py` |
| Original engineering | **Mask-gated removal** with raised-cosine crossfades; samples outside the mask are bit-identical to the input | `sonic/editing.py` |
| Original engineering | Controlled **dataset generator**, **evaluation protocol** (regional SDR, suppression, localization, click metric, paired statistics), cached fair comparison | `sonic/dataset*.py`, `sonic/evaluation.py`, `sonic/experiment.py` |
| Hypothesis (to be tested) | Localized subtraction improves preservation outside target events with no loss of suppression inside them | `reports/research_note.md` |

No model is trained or fine-tuned.

## Method

```
audio + instruction
  └─ instruction parser (rule-based)      "Remove the dog barking ..." → target = "dog barking"
  └─ AudioSep(x, target) → ŝ               one pass, shared by every method
  └─ CLAP window scores                    cos(a_w, t_target) − max_k cos(a_w, t_background_k)
       → z-score → frame curve → threshold → median filter → gap fill / min duration → pad
       → intervals
  └─ mask m(t) ∈ [0,1] with raised-cosine crossfades
  └─ y = x − m(t)·ŝ
```

Three conditions share the **same** AudioSep estimate ŝ and differ only in `m(t)`:

* **baseline**: `m ≡ 1` (global removal)
* **SONIC**: `m` from CLAP temporal localization
* **oracle**: `m` from ground-truth event boundaries (upper bound for localization)

## Repository layout

```
sonic/                  library (numpy core; torch only inside separation.py)
  audio.py              I/O, resampling, mixing, framing, ramps
  instruction.py        instruction → EditInstruction(action, target)
  localization.py       CLAP window scoring + interval post-processing + masks
  separation.py         AudioSep/CLAP wrapper (Separator + Embedder interfaces)
  editing.py            global_remove / masked_remove
  pipeline.py           end-to-end edit() for a single clip; baseline/sonic/oracle masks
  evaluation.py         SDR, SI-SDR, suppression, localization, click metric, bootstrap, Wilcoxon
  dataset.py            pure mixture synthesis;  dataset_builder.py: ESC-50/LibriSpeech → splits
  experiment.py         stage-1 caching and stage-2 evaluation
  ablation.py           one-factor-at-a-time ablations + dev-based selection
  plotting.py           localization / comparison / ablation figures
  config.py             YAML config with `section.key=value` CLI overrides
scripts/                fetch_models, prepare_dataset, run_separation, run_baseline, run_sonic,
                        run_ablations, evaluate, make_examples, build_notebook
notebooks/sonic_colab.ipynb
configs/default.yaml
tests/                  pytest suite (test doubles only; no model weights needed)
reports/research_note.md
docs/superpowers/       design spec and implementation plan
```

## Setup

### Local (unit tests only, CPU, no weights)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

### Google Colab (full pipeline)

Open `notebooks/sonic_colab.ipynb` on a **T4 GPU** runtime and run all cells. Only the first
configuration cell may need editing (paths, Drive persistence, `QUICK` smoke-test mode).

### Command line (any CUDA machine)

```bash
pip install -r requirements-colab.txt && pip install -e .
OV="paths.data_root=/data/sonic paths.models_dir=/models/sonic"

python scripts/fetch_models.py $OV                    # AudioSep @ pinned commit + HF checkpoints
python scripts/prepare_dataset.py $OV                 # ESC-50 + LibriSpeech → dev/eval mixtures
python scripts/run_separation.py --split dev  $OV     # stage 1 (GPU), cached
python scripts/run_separation.py --split eval $OV
python scripts/run_ablations.py $OV                   # dev only → results/ablations/tuned.yaml
python scripts/evaluate.py --split eval --config results/ablations/tuned.yaml --clap $OV
python scripts/make_examples.py --config results/ablations/tuned.yaml $OV

# single file, end to end
python scripts/run_sonic.py --audio clip.wav --instruction "Remove the dog barking" $OV
python scripts/run_baseline.py --audio clip.wav --instruction "Remove the dog barking" $OV
```

## Dataset

Synthetic 10 s mixtures at 32 kHz, generated deterministically (`dataset.seed`):

| Scenario | Target | Background |
|---|---|---|
| `dog_env` | dog barking (1 event, ≤ 3 s) | continuous environmental bed + speech |
| `short_event` | door knock, glass breaking, cough, can opening, sneeze, car horn, alarm (1 event, ≤ 2 s) | continuous bed |
| `intermittent` | dog, knock, cough, sneeze, car horn, cat, rooster (2–3 events, ≤ 1.5 s each) | continuous bed |

Beds: rain, wind, sea waves, crackling fire, crickets, engine, vacuum cleaner, washing machine
(three clips crossfaded). Event SNR ∈ [−5, 5] dB relative to the background; speech is
0–6 dB above the bed. ESC-50 folds 1–4 → **dev** (10 per scenario, for tuning and ablations);
fold 5 → **eval** (20 per scenario, reported). LibriSpeech speakers are split 80/20, disjoint.
Each sample stores the mixture, target, reference background (= ideal edited output), stems,
instruction and event boundaries. No audio is committed.

## Metrics

With x = mixture, s = target, b = reference background, y = output, E = ground-truth event
regions:

* **Background preservation**: `sdr_bg_out`, `mse_out`, i.e. SDR = 10·log10(‖b‖²/‖y−b‖²) on Ē.
  Scale-dependent on purpose: SI-SDR cannot see a scaled copy of the background being removed.
* **Editing quality**: `sdr_bg_in` (on E), `sdr_bg` / `sisdr_bg` / `sisdri` (whole clip).
* **Target suppression**: `target_suppression_db` = 10·log10(‖s_E‖² / ‖(y−b)_E‖²).
* **Localization**: frame-level precision / recall / F1 / IoU, onset/offset error, event recall.
* **Discontinuities**: `edge_click_db`, high-frequency error energy at mask edges vs. its neighbourhood.
* **Cost**: latency per stage, peak GPU memory.
* **Auxiliary only**: CLAP similarity drop to the target text (not a fidelity measure).

SDR values are capped at 60 dB. Results are reported per sample and aggregated (mean, 95%
bootstrap CI), with paired Wilcoxon tests between methods.

## Known limitations

* Removal only; amplification/replacement are not implemented.
* The instruction parser is rule-based: it handles removal phrasings, not arbitrary language.
* Localization recall is the bottleneck: missed event segments keep the target in the output
  (see the intermittent scenario in the research note).
* The CLAP checkpoint reports ESC-50 accuracy in its name, i.e. ESC-50 was part of its
  training/evaluation data, so localization on ESC-50 sources may be optimistic.
* Per-clip z-score normalization assumes the target is present; on clips without the target,
  some windows will still exceed the threshold.
* Ground-truth intervals span the trimmed event clip, including short pauses inside it.
* Synthetic mixtures are additive with no reverberation or real acoustic scenes.
* AudioSep and CLAP operate on 10 s segments; longer inputs rely on AudioSep's chunked mode,
  and CLAP crops inputs longer than 10 s at random.
* `sonic/separation.py` contains compatibility shims for current torch/transformers
  (see its docstring); they were written against the pinned AudioSep commit.

## References

* X. Liu et al., "Separate Anything You Describe," arXiv:2308.05037, 2023 (AudioSep).
* Y. Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and
  Keyword-to-Caption Augmentation," ICASSP 2023 (LAION-CLAP).
* K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM Multimedia 2015.
* V. Panayotov et al., "LibriSpeech: an ASR corpus based on public domain audio books," ICASSP 2015.
* J. Le Roux et al., "SDR – half-baked or well done?," ICASSP 2019.

## License

Code: MIT (see `LICENSE`). Third-party models and datasets keep their own licenses
(AudioSep: MIT; LAION-CLAP: CC0-1.0; ESC-50: CC BY-NC 3.0; LibriSpeech: CC BY 4.0).
Audio generated from ESC-50 is for non-commercial research use.
