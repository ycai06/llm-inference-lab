"""Warm-request configuration validation without an inference engine or weights."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend.schemas.config import ConfigurationError
from backend.schemas.request_config import MAX_FIXTURE_BYTES, RequestConfig, parse_request_config


@pytest.fixture
def raw_config(tmp_path: Path) -> dict[str, Any]:
    fixture = tmp_path / "systems.txt"
    fixture.write_text("A processor runs instructions and observes memory ordering.\n")
    return {
        "name": "TEST ONLY request configuration",
        "model": "external.gguf",
        "binary": "llama-server",
        "fixtures": [fixture.name],
    }


def parse(tmp_path: Path, raw: dict[str, Any], **overrides: Any) -> RequestConfig:
    return parse_request_config(tmp_path / "experiment.yaml", raw | overrides)


def test_defaults_and_paths_without_model_or_engine_access(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    original = raw_config.copy()
    config = parse(tmp_path, raw_config)
    assert raw_config == original
    assert config.model == tmp_path / "external.gguf"
    assert config.binary == tmp_path / "llama-server"
    assert not config.model.exists()
    assert not config.binary.exists()
    assert config.fixtures == (tmp_path / "systems.txt",)
    assert config.mode == "request_warm"
    assert config.repetitions == 5
    assert config.warmups == 1
    assert config.cache_prompt is False
    assert config.ignore_eos is False
    assert config.seed == 42
    assert config.context_size == 2048
    assert len(config.cases()) == 9
    serialized = json.loads(json.dumps(config.to_dict()))
    assert serialized["fixtures"] == [str(tmp_path / "systems.txt")]
    assert serialized["binary"] == str(config.binary)


def test_expanded_paths_are_relative_to_configuration(
    tmp_path: Path, raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIOL_REQUEST_MODEL", "relative-model.gguf")
    monkeypatch.setenv("LIOL_REQUEST_FIXTURE", str(tmp_path / "systems.txt"))
    config = parse(
        tmp_path, raw_config, model="${LIOL_REQUEST_MODEL}", fixtures=["$LIOL_REQUEST_FIXTURE"]
    )
    assert config.model == tmp_path / "relative-model.gguf"
    assert config.fixtures == (tmp_path / "systems.txt",)


def test_unset_or_empty_environment_variable_is_actionable(
    tmp_path: Path, raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIOL_REQUEST_UNSET", raising=False)
    with pytest.raises(ConfigurationError, match="Set the environment variable in model"):
        parse(tmp_path, raw_config, model="${LIOL_REQUEST_UNSET}")
    monkeypatch.setenv("LIOL_REQUEST_EMPTY", "")
    with pytest.raises(ConfigurationError, match="expand to a nonempty path"):
        parse(tmp_path, raw_config, binary="${LIOL_REQUEST_EMPTY}")


def test_matrix_is_complete_seeded_and_does_not_modify_global_random_state(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    other = tmp_path / "architecture.txt"
    other.write_text("Cache behavior depends on access patterns.\n")
    config = parse(
        tmp_path,
        raw_config,
        fixtures=["systems.txt", "architecture.txt"],
        prompt_tokens=[128, 512],
        generation_tokens=[8, 16],
        threads=[4, 6],
        gpu_layers=[0, -1],
        ubatch_sizes=[64, 128],
    )
    state = random.getstate()
    cases = config.cases()
    assert config.cases() == cases
    assert random.getstate() == state
    assert len(cases) == 64
    assert len({tuple(sorted(case.items())) for case in cases}) == 64
    assert {case["kind"] for case in cases} == {"request"}
    assert {case["context_size"] for case in cases} == {2048}
    assert {case["fixture"] for case in cases} == {str(p) for p in config.fixtures}
    assert replace(config, shuffle_seed=21).cases() != cases


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": " \n"}, "name must be a nonempty string"),
        ({"name": 12}, "name must be a nonempty string"),
        ({"binary": ""}, "binary must be a nonempty path"),
        ({"model": False}, "model must be a nonempty path"),
        ({"fixtures": []}, "fixtures must be a nonempty list"),
        ({"fixtures": "systems.txt"}, "fixtures must be a nonempty list"),
        ({"fixtures": [False]}, "fixture must be a nonempty path"),
        ({"fixtures": ["systems.txt", "./systems.txt"]}, "duplicate resolved paths"),
        ({"prompt_tokens": []}, "prompt_tokens must be a nonempty list"),
        ({"generation_tokens": []}, "generation_tokens must be a nonempty list"),
        ({"threads": []}, "threads must be a nonempty list"),
        ({"gpu_layers": []}, "gpu_layers must be a nonempty list"),
        ({"batch_sizes": []}, "batch_sizes must be a nonempty list"),
        ({"ubatch_sizes": []}, "ubatch_sizes must be a nonempty list"),
        ({"prompt_tokens": [0]}, "prompt_tokens must be an integer"),
        ({"prompt_tokens": [True]}, "prompt_tokens must be an integer"),
        ({"generation_tokens": [4097]}, "generation_tokens must be an integer"),
        ({"generation_tokens": [2.0]}, "generation_tokens must be an integer"),
        ({"threads": [4097]}, "threads must be an integer"),
        ({"gpu_layers": [-2]}, "gpu_layers must be an integer"),
        ({"batch_sizes": [32769]}, "batch_sizes must be an integer"),
        ({"threads": [4, 4]}, "contains duplicate settings"),
        ({"batch_sizes": [64], "ubatch_sizes": [128]}, "Every ubatch_size"),
        ({"batch_sizes": [64, 256], "ubatch_sizes": [128]}, "Every ubatch_size"),
        ({"context_size": True}, "context_size must be an integer"),
        ({"context_size": 32769}, "context_size must be an integer"),
        ({"context_size": 1024}, "must fit context_size"),
        ({"repetitions": True}, "repetitions must be an integer"),
        ({"repetitions": 0}, "repetitions must be an integer"),
        ({"repetitions": 1001}, "repetitions must be an integer"),
        ({"warmups": 0}, "warmups must be an integer"),
        ({"warmups": 11}, "warmups must be an integer"),
        ({"seed": -1}, "seed must be an integer"),
        ({"seed": 2**32}, "seed must be an integer"),
        ({"shuffle_seed": False}, "shuffle_seed must be an integer"),
        ({"ignore_eos": 1}, "ignore_eos must be a boolean"),
        ({"cache_prompt": 0}, "cache_prompt must be a boolean"),
        ({"cache_prompt": True}, "cache_prompt must be false"),
        ({"mode": "concurrent"}, "Supported mode"),
        ({"flash_attention": True}, "quote YAML strings"),
        ({"flash_attention": "automatic"}, "flash_attention must be"),
        ({"arbitrary_shell_args": ["--host", "0.0.0.0"]}, "Unsupported configuration keys"),
    ],
)
def test_reject_invalid_configuration(
    tmp_path: Path,
    raw_config: dict[str, Any],
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        parse(tmp_path, raw_config, **overrides)


@pytest.mark.parametrize("field", ["timeout_s", "startup_timeout_s"])
@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), 86401, "10"])
def test_reject_invalid_timeouts(
    tmp_path: Path, raw_config: dict[str, Any], field: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError, match=f"{field} must be finite"):
        parse(tmp_path, raw_config, **{field: value})


def test_context_reserves_one_position_and_accepts_exact_safe_boundary(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    config = parse(tmp_path, raw_config, prompt_tokens=[16], generation_tokens=[8], context_size=25)
    assert config.context_size == 25
    with pytest.raises(ConfigurationError, match="must fit context_size"):
        parse(tmp_path, raw_config, prompt_tokens=[16], generation_tokens=[8], context_size=24)


def test_missing_directory_nonutf8_and_empty_fixtures(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError, match="missing or not a regular file"):
        parse(tmp_path, raw_config, fixtures=["absent.txt"])
    with pytest.raises(ConfigurationError, match="missing or not a regular file"):
        parse(tmp_path, raw_config, fixtures=[str(tmp_path)])
    broken = tmp_path / "invalid-encoding.txt"
    broken.write_bytes(b"\xff\xfe\x80")
    with pytest.raises(ConfigurationError, match="Cannot read UTF-8 fixture"):
        parse(tmp_path, raw_config, fixtures=[broken.name])
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n\t")
    with pytest.raises(ConfigurationError, match="nonempty text"):
        parse(tmp_path, raw_config, fixtures=[empty.name])


def test_fixture_size_read_is_bounded(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    fixture = tmp_path / "systems.txt"
    fixture.write_bytes(b"a" * MAX_FIXTURE_BYTES)
    assert parse(tmp_path, raw_config).fixtures == (fixture,)
    fixture.write_bytes(b"a" * (MAX_FIXTURE_BYTES + 1))
    with pytest.raises(ConfigurationError, match="exceeds 262144 bytes"):
        parse(tmp_path, raw_config)


def test_fixture_aliases_are_duplicate_paths(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    alias = tmp_path / "alias.txt"
    alias.symlink_to(tmp_path / "systems.txt")
    with pytest.raises(ConfigurationError, match="duplicate resolved paths"):
        parse(tmp_path, raw_config, fixtures=["systems.txt", alias.name])


def test_matrix_limit_is_checked_before_materializing_cases(
    tmp_path: Path, raw_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_: RequestConfig) -> list[dict[str, Any]]:
        pytest.fail("Oversized request matrix was materialized")

    monkeypatch.setattr(RequestConfig, "cases", forbidden)
    with pytest.raises(ConfigurationError, match="exceeds 128 request cases"):
        parse(
            tmp_path,
            raw_config,
            prompt_tokens=list(range(1, 129)),
            generation_tokens=[8, 16],
            threads=[1, 2],
        )


def test_direct_cases_call_also_bounds_materialization(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    config = parse(tmp_path, raw_config)
    oversized = replace(config, prompt_tokens=tuple(range(1, 129)), threads=(1, 2))
    with pytest.raises(ConfigurationError, match="exceeds 128 request cases"):
        oversized.cases()
