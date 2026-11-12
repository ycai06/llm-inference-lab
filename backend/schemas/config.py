"""Strict YAML configuration with an explicitly bounded experiment matrix."""

from __future__ import annotations

import itertools
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from backend.schemas.request_config import RequestConfig


class ConfigurationError(ValueError):
    """The experiment cannot be represented safely by this backend."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: Path
    binary: Path
    prompt_tokens: tuple[int, ...] = (128, 512, 1024)
    generation_tokens: tuple[int, ...] = (64, 128, 256)
    threads: tuple[int, ...] = (6,)
    gpu_layers: tuple[int, ...] = (0,)
    batch_sizes: tuple[int, ...] = (512,)
    ubatch_sizes: tuple[int, ...] = (128,)
    repetitions: int = 5
    timeout_s: float = 180
    shuffle_seed: int = 42
    flash_attention: str = "auto"
    mode: str = "synthetic_warm"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(model=str(self.model), binary=str(self.binary))
        return result

    def cases(self) -> list[dict[str, Any]]:
        workloads = [("prefill", n) for n in self.prompt_tokens] + [
            ("decode", n) for n in self.generation_tokens
        ]
        cases = [
            dict(
                kind=kind,
                tokens=n,
                threads=t,
                gpu_layers=g,
                batch_size=b,
                ubatch_size=u,
                flash_attention=self.flash_attention,
            )
            for (kind, n), t, g, b, u in itertools.product(
                workloads, self.threads, self.gpu_layers, self.batch_sizes, self.ubatch_sizes
            )
            if u <= b
        ]
        random.Random(self.shuffle_seed).shuffle(cases)
        return cases


def load_config(path: Path) -> ExperimentConfig | RequestConfig:
    try:
        source = path.read_text()
        node = yaml.compose(source, Loader=yaml.SafeLoader)
        if not isinstance(node, yaml.MappingNode):
            raise ConfigurationError("Configuration must be a YAML mapping")
        seen: set[str] = set()
        for key, _ in node.value:
            if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ConfigurationError("Configuration keys must be strings")
            if key.value in seen:
                raise ConfigurationError(f"Duplicate configuration key: {key.value}")
            seen.add(key.value)
        raw = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration must be a YAML mapping")
    if raw.get("mode") == "request_warm":
        from backend.schemas.request_config import parse_request_config

        return parse_request_config(path, raw)
    unknown = set(raw) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ConfigurationError(f"Unsupported configuration keys: {sorted(unknown, key=str)}")
    for field in ("model", "binary"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{field} must be a nonempty path")
        expanded = os.path.expandvars(value)
        if re.search(r"\$\{?\w+", expanded):
            raise ConfigurationError(f"Set the environment variable in {field}: {value}")
        candidate = Path(expanded).expanduser()
        raw[field] = (
            (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate
        )
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ConfigurationError("name must be a nonempty string")
    for field in (
        "prompt_tokens",
        "generation_tokens",
        "threads",
        "gpu_layers",
        "batch_sizes",
        "ubatch_sizes",
    ):
        values = raw.get(field, ExperimentConfig.__dataclass_fields__[field].default)
        if not isinstance(values, (list, tuple)) or any(type(x) is not int for x in values):
            raise ConfigurationError(f"{field} must be a list of integers")
        if len(values) != len(set(values)):
            raise ConfigurationError(f"{field} contains duplicate settings")
        if field not in ("prompt_tokens", "generation_tokens") and not values:
            raise ConfigurationError(f"{field} cannot be empty")
        minimum = -1 if field == "gpu_layers" else 1
        maximum = 4096 if field in ("threads", "gpu_layers") else 131072
        if any(not minimum <= x <= maximum for x in values):
            raise ConfigurationError(f"{field} must be between {minimum} and {maximum}")
        raw[field] = tuple(values)
    for field, lower, upper in (("repetitions", 1, 1000), ("shuffle_seed", 0, 2**32 - 1)):
        value = raw.get(field, ExperimentConfig.__dataclass_fields__[field].default)
        if type(value) is not int or not lower <= value <= upper:
            raise ConfigurationError(f"{field} must be an integer in [{lower}, {upper}]")
    timeout = raw.get("timeout_s", 180)
    if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
        raise ConfigurationError("timeout_s must be finite and in (0, 86400]")
    if raw.get("mode", "synthetic_warm") != "synthetic_warm":
        raise ConfigurationError("This backend supports only mode: synthetic_warm")
    if raw.get("flash_attention", "auto") not in ("on", "off", "auto"):
        raise ConfigurationError("flash_attention must be on, off, or auto (quote YAML strings)")
    config = ExperimentConfig(**raw)
    if not config.prompt_tokens and not config.generation_tokens:
        raise ConfigurationError("At least one workload is required")
    if any(u > b for u in config.ubatch_sizes for b in config.batch_sizes):
        raise ConfigurationError("Every ubatch_size must be <= every batch_size")
    count = (len(config.prompt_tokens) + len(config.generation_tokens)) * math.prod(
        len(values)
        for values in (config.threads, config.gpu_layers, config.batch_sizes, config.ubatch_sizes)
    )
    if count > 256:
        raise ConfigurationError("Experiment matrix exceeds 256 cases; split the suite")
    return config
