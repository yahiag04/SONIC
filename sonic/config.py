"""Typed configuration loaded from YAML with ``section.key=value`` overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Sequence

import yaml

from sonic.localization import LocalizationParams

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass(frozen=True)
class PathsConfig:
    data_root: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    cache_dir: str = "results/cache"


@dataclass(frozen=True)
class SeparationConfig:
    device: str = "auto"
    use_chunk: bool = False


@dataclass(frozen=True)
class EditingConfig:
    crossfade_s: float = 0.05
    mask: str = "hard"  # hard | soft
    soft_temperature: float = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 1234
    length_s: float = 10.0
    n_eval_per_scenario: int = 20
    n_dev_per_scenario: int = 10
    snr_db_min: float = -5.0
    snr_db_max: float = 5.0
    speech_to_bed_db_min: float = 0.0
    speech_to_bed_db_max: float = 6.0


@dataclass(frozen=True)
class SonicConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    localization: LocalizationParams = field(default_factory=LocalizationParams)
    editing: EditingConfig = field(default_factory=EditingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["localization"]["background_prompts"] = list(d["localization"]["background_prompts"])
        return d

    def with_overrides(self, overrides: Sequence[str]) -> "SonicConfig":
        data = self.to_dict()
        _apply_overrides(data, overrides)
        return _from_dict(data)


_SECTION_CLASSES = {
    "paths": PathsConfig,
    "separation": SeparationConfig,
    "localization": LocalizationParams,
    "editing": EditingConfig,
    "dataset": DatasetConfig,
}


def _from_dict(data: dict) -> SonicConfig:
    unknown = set(data) - set(_SECTION_CLASSES)
    if unknown:
        raise KeyError(f"unknown config sections: {sorted(unknown)}")
    kwargs = {}
    for name, cls in _SECTION_CLASSES.items():
        section = dict(data.get(name) or {})
        valid = {f.name for f in fields(cls)}
        bad = set(section) - valid
        if bad:
            raise KeyError(f"unknown keys in [{name}]: {sorted(bad)}")
        if name == "localization" and "background_prompts" in section:
            section["background_prompts"] = tuple(section["background_prompts"])
        kwargs[name] = cls(**section)
    return SonicConfig(**kwargs)


def _apply_overrides(data: dict, overrides: Sequence[str]) -> None:
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        data.setdefault(section, {})[name] = yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> SonicConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    _apply_overrides(data, overrides)
    return _from_dict(data)


def replace_section(cfg: SonicConfig, section: str, **changes) -> SonicConfig:
    """Return a copy of ``cfg`` with fields of one section replaced (used by ablations)."""
    return replace(cfg, **{section: replace(getattr(cfg, section), **changes)})
