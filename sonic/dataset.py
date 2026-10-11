"""Synthetic evaluation mixtures with known stems and event boundaries.

Pure functions only; file download and I/O live in ``scripts/prepare_dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonic.audio import fit_peak, raised_cosine, rms, scale_to_rms, scale_to_snr

# ESC-50 category -> natural-language prompt used as the separation/localization query.
TARGET_PROMPTS = {
    "dog": "dog barking",
    "door_wood_knock": "knocking on a door",
    "glass_breaking": "glass breaking",
    "coughing": "coughing",
    "can_opening": "can opening",
    "clock_alarm": "alarm clock ringing",
    "sneezing": "sneezing",
    "car_horn": "car horn honking",
    "crying_baby": "baby crying",
    "cat": "cat meowing",
    "rooster": "rooster crowing",
    "church_bells": "church bells ringing",
}

# Continuous ESC-50 categories used as background beds.
BED_CLASSES = ("rain", "wind", "sea_waves", "crackling_fire", "crickets", "engine", "vacuum_cleaner", "washing_machine")

INSTRUCTION_TEMPLATES = (
    "Remove the {p} while preserving everything else.",
    "Get rid of the {p}.",
    "Delete the {p} but keep the rest of the recording.",
    "Mute the {p}.",
    "Eliminate the {p} and keep the background.",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    targets: tuple[str, ...]
    n_events: tuple[int, int]  # inclusive range
    max_event_s: float
    with_speech: bool


SCENARIOS = {
    "dog_env": Scenario("dog_env", ("dog",), (1, 1), 3.0, True),
    "short_event": Scenario(
        "short_event",
        ("door_wood_knock", "glass_breaking", "coughing", "can_opening", "sneezing", "car_horn", "clock_alarm"),
        (1, 1), 2.0, False,
    ),
    "intermittent": Scenario(
        "intermittent",
        ("dog", "door_wood_knock", "coughing", "sneezing", "car_horn", "cat", "rooster"),
        (2, 3), 1.5, False,
    ),
}


def make_instruction(target_class: str, rng: np.random.Generator) -> tuple[str, str]:
    prompt = TARGET_PROMPTS[target_class]
    template = INSTRUCTION_TEMPLATES[int(rng.integers(len(INSTRUCTION_TEMPLATES)))]
    return template.format(p=prompt), prompt


def trim_active(x: np.ndarray, sr: int, top_db: float = 30.0, frame_s: float = 0.02) -> np.ndarray:
    """Trim leading/trailing parts quieter than (max frame energy - top_db)."""
    hop = max(1, int(frame_s * sr))
    n_frames = max(1, len(x) // hop)
    energy = np.array([np.mean(x[i * hop:(i + 1) * hop] ** 2) for i in range(n_frames)]) + 1e-20
    db = 10 * np.log10(energy)
    active = np.flatnonzero(db >= db.max() - top_db)
    return x[active[0] * hop: min(len(x), (active[-1] + 1) * hop)]


def concat_crossfade(clips: list[np.ndarray], length: int, sr: int, fade_s: float = 0.05) -> np.ndarray:
    """Concatenate clips (cycling through them) with crossfades until ``length`` samples."""
    if not clips:
        raise ValueError("need at least one clip")
    f = int(fade_s * sr)
    out = np.array(clips[0], dtype=np.float64)
    i = 1
    while len(out) < length:
        nxt = np.array(clips[i % len(clips)], dtype=np.float64)
        i += 1
        k = min(f, len(out), len(nxt))
        if k == 0:
            out = np.concatenate([out, nxt])
            continue
        up = raised_cosine(k)
        out = np.concatenate([out[:-k], out[-k:] * (1 - up) + nxt[:k] * up, nxt[k:]])
    return out[:length]


def place_events(event_lengths: list[int], length: int, rng: np.random.Generator,
                 min_gap: int, margin: int, max_tries: int = 1000) -> list[int]:
    """Random non-overlapping onsets, one per event and in the same order as ``event_lengths``,
    with at least ``min_gap`` samples between consecutive events."""
    for _ in range(max_tries):
        onsets = []
        ok = True
        for L in event_lengths:
            hi = length - margin - L
            if hi <= margin:
                ok = False
                break
            onsets.append(int(rng.integers(margin, hi)))
        if not ok:
            break
        spans = sorted(zip(onsets, event_lengths))
        if all(spans[j + 1][0] - (spans[j][0] + spans[j][1]) >= min_gap for j in range(len(spans) - 1)):
            return onsets
    raise ValueError("could not place events without overlap; shorten events or reduce their number")


def synthesize_sample(
    bed: np.ndarray,
    events: list[np.ndarray],
    snr_db: float,
    rng: np.random.Generator,
    sr: int,
    speech: np.ndarray | None = None,
    speech_to_bed_db: float = 3.0,
    length_s: float = 10.0,
    bed_rms: float = 0.05,
    peak: float = 0.9,
    min_gap_s: float = 0.5,
    margin_s: float = 0.25,
) -> dict:
    """Mix a bed (+ optional speech) with target events at random positions.

    ``snr_db`` is the RMS ratio of the (active) event clips to the background
    (bed + speech) over the whole clip. Returns stems whose sum is exactly the
    mixture, plus ground-truth intervals in seconds.
    """
    n = int(round(length_s * sr))
    if len(bed) < n:
        raise ValueError("bed shorter than requested length")
    bed = scale_to_rms(np.asarray(bed[:n], dtype=np.float64), bed_rms)
    stems = {"bed": bed}
    background = bed.copy()
    if speech is not None:
        sp = np.zeros(n)
        sp[: min(n, len(speech))] = speech[:n]
        sp = scale_to_snr(sp, bed, speech_to_bed_db)
        stems["speech"] = sp
        background = background + sp

    ref_level = rms(background)
    events = [scale_to_rms(np.asarray(e, dtype=np.float64), ref_level * 10 ** (snr_db / 20)) for e in events]
    onsets = place_events([len(e) for e in events], n, rng, int(min_gap_s * sr), int(margin_s * sr))
    target = np.zeros(n)
    intervals = []
    for o, e in zip(onsets, events):
        target[o:o + len(e)] += e
        intervals.append((o / sr, (o + len(e)) / sr))
    intervals.sort()
    stems["target"] = target

    stems, gain = fit_peak(stems, peak)
    target = stems["target"]
    background = sum(v for k, v in stems.items() if k != "target")
    stems = {k: v.astype(np.float32) for k, v in stems.items()}
    target32 = target.astype(np.float32)
    background32 = np.asarray(background, dtype=np.float32)
    return {
        "mixture": target32 + background32,
        "target": target32,
        "background": background32,
        "stems": stems,
        "intervals": intervals,
        "gain": gain,
    }
