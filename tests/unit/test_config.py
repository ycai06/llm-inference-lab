from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import yaml

from backend.schemas.config import ConfigurationError, ExperimentConfig, load_config


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    values = {"name": "unit-only", "model": "model.gguf", "binary": "llama-bench"}
    values.update(overrides)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_defaults_and_relative_paths_are_config_relative(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    assert config.repetitions == 5
    assert config.model == tmp_path / "model.gguf"
    assert config.binary == tmp_path / "llama-bench"
    assert config.mode == "synthetic_warm"
    assert config.to_dict()["model"] == str(tmp_path / "model.gguf")
    assert len(config.cases()) == 6


def test_environment_paths_expand_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIOL_UNIT_MODEL", str(tmp_path / "external-model.gguf"))
    config = load_config(write_config(tmp_path, model="${LIOL_UNIT_MODEL}"))
    assert config.model == tmp_path / "external-model.gguf"
    assert not config.model.exists()


def test_missing_environment_variable_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIOL_UNIT_UNSET", raising=False)
    with pytest.raises(ConfigurationError, match="Set the environment variable in model"):
        load_config(write_config(tmp_path, model="${LIOL_UNIT_UNSET}"))


def test_matrix_is_complete_deterministic_and_does_not_change_global_rng(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            prompt_tokens=[128, 512],
            generation_tokens=[64],
            threads=[2, 4],
            gpu_layers=[0, -1],
            batch_sizes=[256, 512],
            ubatch_sizes=[64, 128],
        )
    )
    before = random.getstate()
    cases = config.cases()
    assert cases == config.cases()
    assert random.getstate() == before
    assert len(cases) == 48
    settings = {tuple(sorted(case.items())) for case in cases}
    assert len(settings) == 48
    assert {case["kind"] for case in cases} == {"prefill", "decode"}
    assert {case["gpu_layers"] for case in cases} == {0, -1}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"arbitrary_shell_args": ["--danger"]}, "Unsupported configuration keys"),
        ({"prompt_tokens": [128, 128]}, "duplicate"),
        ({"threads": [True]}, "list of integers"),
        ({"threads": [2.0]}, "list of integers"),
        ({"threads": []}, "cannot be empty"),
        ({"gpu_layers": [-2]}, "between"),
        ({"prompt_tokens": [0]}, "between"),
        ({"generation_tokens": [-1]}, "between"),
        ({"prompt_tokens": [], "generation_tokens": []}, "At least one workload"),
        ({"batch_sizes": [128], "ubatch_sizes": [256]}, "ubatch_size"),
        ({"batch_sizes": [64, 256], "ubatch_sizes": [128]}, "ubatch_size"),
        ({"repetitions": True}, "repetitions must be an integer"),
        ({"repetitions": 0}, "repetitions must be an integer"),
        ({"shuffle_seed": -1}, "shuffle_seed must be an integer"),
        ({"timeout_s": float("inf")}, "timeout_s must be finite"),
        ({"timeout_s": float("nan")}, "timeout_s must be finite"),
        ({"timeout_s": True}, "timeout_s must be finite"),
        ({"mode": "concurrent"}, "supports only mode"),
        ({"flash_attention": True}, "quote YAML strings"),
        ({"name": "  "}, "name must be a nonempty string"),
        ({"binary": ""}, "binary must be a nonempty path"),
    ],
)
def test_reject_invalid_configuration(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(tmp_path, **overrides))


def test_reject_matrix_above_limit(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="exceeds 256"):
        load_config(
            write_config(
                tmp_path,
                prompt_tokens=list(range(1, 129)),
                generation_tokens=[],
                threads=[1, 2, 3],
            )
        )


@pytest.mark.parametrize("text", ["- list\n- is not a mapping", "threads: [", "null"])
def test_malformed_yaml_and_nonmapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_config(tmp_path / "missing.yaml")


def test_reject_duplicate_root_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "name: first\nname: silently-overridden\nmodel: model.gguf\nbinary: llama-bench\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="[Dd]uplicate"):
        load_config(path)


def test_matrix_limit_is_checked_before_materializing_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_materialization(_: ExperimentConfig) -> list[dict[str, Any]]:
        pytest.fail("An oversized matrix was materialized before its cardinality check")

    monkeypatch.setattr(ExperimentConfig, "cases", forbidden_materialization)
    with pytest.raises(ConfigurationError, match="exceeds 256"):
        load_config(
            write_config(
                tmp_path,
                prompt_tokens=list(range(1, 129)),
                generation_tokens=[],
                threads=list(range(1, 129)),
                gpu_layers=list(range(128)),
            )
        )
