from __future__ import annotations

import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend.engines.llama_bench import BenchCase, LlamaBench, parse_samples

HELP = """usage: llama-bench
 -m, --model <filename>
 -r, --repetitions <n>
 -o, --output <csv|json|jsonl|md|sql>
 -p, --n-prompt <n>
 -n, --n-gen <n>
 -t, --threads <n>
 -ngl, --n-gpu-layers <n>
 -b, --batch-size <n>
 -ub, --ubatch-size <n>
 -fa, --flash-attn <on|off|auto>
 -dev, --device <dev0/dev1/...>
 -nkvo, --no-kv-offload <0|1>
 -nopo, --no-op-offload <0|1>
 --list-devices
 --offline
 -v, --verbose
"""


@pytest.fixture
def case() -> BenchCase:
    return BenchCase("prefill", 128, 2, 0, 128, 64)


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "test-only.gguf"
    # A fake engine never loads weights; this is only a path/header fixture.
    path.write_bytes(b"GGUFtest-fixture")
    return path


def fake_engine(
    tmp_path: Path, *, help_text: str = HELP, devices: str = "MTL0: Test Metal Device"
) -> Path:
    path = tmp_path / "fake-llama-bench"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"help_text = {help_text!r}\n"
        f"devices = {devices!r}\n"
        "args = sys.argv[1:]\n"
        "if args == ['--help']: print(help_text)\n"
        "elif args == ['--version']: print('test-fixture-version')\n"
        "elif args == ['--list-devices']: print(devices)\n"
        "else: print(json.dumps(args))\n"
    )
    path.chmod(0o755)
    return path


def row(case: BenchCase) -> dict[str, Any]:
    return {
        "n_prompt": case.tokens if case.kind == "prefill" else 0,
        "n_gen": case.tokens if case.kind == "decode" else 0,
        "n_threads": case.threads,
        "n_gpu_layers": case.gpu_layers,
        "n_batch": case.batch_size,
        "n_ubatch": case.ubatch_size,
        "n_depth": 0,
        "flash_attn": {"auto": -1, "off": 0, "on": 1}[case.flash_attention],
        "no_kv_offload": case.gpu_layers == 0,
        "no_op_offload": int(case.gpu_layers == 0),
        "devices": "none" if case.gpu_layers == 0 else "MTL0",
        "build_commit": "test-only-commit",
        "samples_ns": [1_000_000_000, 2_000_000_000],
        "samples_ts": [case.tokens, case.tokens / 2],
    }


def test_probe_reads_actual_binary_outputs(tmp_path: Path) -> None:
    metadata = LlamaBench(fake_engine(tmp_path)).probe()
    assert metadata["version"] == "test-fixture-version"
    assert metadata["help"] == HELP + "\n"
    assert "MTL0" in metadata["devices"]
    assert len(metadata["binary_sha256"]) == 64


def test_cpu_command_disables_all_offload(tmp_path: Path, model: Path, case: BenchCase) -> None:
    result = LlamaBench(fake_engine(tmp_path)).run(model, case, 5, 3)
    args = json.loads(result.stdout)
    for flag, expected in {
        "-ngl": "0",
        "-dev": "none",
        "-nkvo": "1",
        "-nopo": "1",
        "-p": "128",
        "-n": "0",
        "-r": "5",
        "-o": "json",
    }.items():
        assert args[args.index(flag) + 1] == expected
    assert "--no-warmup" not in args
    assert "--offline" in args
    assert "-v" in args


def test_decode_command_does_not_also_run_prefill(
    tmp_path: Path, model: Path, case: BenchCase
) -> None:
    result = LlamaBench(fake_engine(tmp_path)).run(model, replace(case, kind="decode"), 5, 3)
    args = json.loads(result.stdout)
    assert args[args.index("-p") + 1] == "0"
    assert args[args.index("-n") + 1] == "128"


def test_metal_requires_real_named_device(tmp_path: Path, model: Path, case: BenchCase) -> None:
    engine = LlamaBench(fake_engine(tmp_path, devices="BLAS: CPU accelerator"))
    with pytest.raises(ValueError, match="no MTL device"):
        engine.run(model, replace(case, gpu_layers=99), 5, 3)


def test_metal_explicitly_selects_reported_device(
    tmp_path: Path, model: Path, case: BenchCase
) -> None:
    result = LlamaBench(fake_engine(tmp_path)).run(model, replace(case, gpu_layers=99), 5, 3)
    args = json.loads(result.stdout)
    assert args[args.index("-dev") + 1] == "MTL0"
    assert args[args.index("-ngl") + 1] == "99"
    assert args[args.index("-nkvo") + 1] == "0"


def test_missing_model(tmp_path: Path, case: BenchCase) -> None:
    with pytest.raises(ValueError, match="Model file does not exist"):
        LlamaBench(fake_engine(tmp_path)).run(tmp_path / "missing.gguf", case, 5, 3)


def test_invalid_model_magic(tmp_path: Path, case: BenchCase) -> None:
    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a GGUF"):
        LlamaBench(fake_engine(tmp_path)).run(invalid, case, 5, 3)


def test_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not executable"):
        LlamaBench(tmp_path / "missing")


def test_rejects_unsupported_switch(tmp_path: Path, model: Path, case: BenchCase) -> None:
    engine = LlamaBench(fake_engine(tmp_path, help_text=HELP.replace("-nopo,", "-old,")))
    with pytest.raises(ValueError, match="-nopo"):
        engine.run(model, case, 5, 3)


def test_cancellation_reaches_managed_process(tmp_path: Path, model: Path, case: BenchCase) -> None:
    engine = LlamaBench(fake_engine(tmp_path))
    engine.probe()
    cancel = threading.Event()
    cancel.set()
    assert engine.run(model, case, 5, 3, cancel).status == "canceled"


def test_parse_preserves_metadata_and_individual_trials(case: BenchCase) -> None:
    original = row(case)
    samples, metadata = parse_samples(json.dumps([original]), case, 2)
    assert metadata == original
    assert [sample["trial_index"] for sample in samples] == [0, 1]
    assert samples[0] == {
        "trial_index": 0,
        "duration_ns": 1_000_000_000,
        "tokens_per_second": 128,
        "tokens": 128,
    }
    assert "ttft_ms" not in samples[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_prompt", 127),
        ("n_gen", 1),
        ("n_threads", 3),
        ("n_gpu_layers", 1),
        ("n_batch", 64),
        ("n_ubatch", 32),
    ],
)
def test_parse_rejects_parameter_mismatch(case: BenchCase, field: str, value: int) -> None:
    original = row(case)
    original[field] = value
    with pytest.raises(ValueError, match=field):
        parse_samples(json.dumps([original]), case, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_depth", 32),
        ("flash_attn", 0),
        ("no_kv_offload", False),
        ("no_op_offload", 0),
        ("devices", "MTL0"),
    ],
)
def test_parse_rejects_backend_control_mismatch(
    case: BenchCase,
    field: str,
    value: object,
) -> None:
    original = row(case)
    original[field] = value
    with pytest.raises(ValueError):
        parse_samples(json.dumps([original]), case, 2)


@pytest.mark.parametrize("flash_attention", ["auto", "off", "on"])
def test_parse_accepts_verified_flash_enum(case: BenchCase, flash_attention: str) -> None:
    updated = replace(case, flash_attention=flash_attention)
    assert len(parse_samples(json.dumps([row(updated)]), updated, 2)[0]) == 2


def test_parse_metal_requires_actual_device_metadata(case: BenchCase) -> None:
    updated = replace(case, gpu_layers=-1)
    original = row(updated)
    assert len(parse_samples(json.dumps([original]), updated, 2)[0]) == 2
    original["devices"] = "auto"
    with pytest.raises(ValueError, match="explicit MTL device"):
        parse_samples(json.dumps([original]), updated, 2)


@pytest.mark.parametrize("duration", [0, -1, 1.5, True, float("nan")])
def test_parse_rejects_invalid_duration(case: BenchCase, duration: object) -> None:
    original = row(case)
    original["samples_ns"][0] = duration
    with pytest.raises(ValueError, match="positive integer"):
        parse_samples(json.dumps([original]), case, 2)


@pytest.mark.parametrize("rate", [0, -1, True, float("nan"), float("inf"), 42])
def test_parse_rejects_invalid_or_inconsistent_rate(case: BenchCase, rate: object) -> None:
    original = row(case)
    original["samples_ts"][0] = rate
    with pytest.raises(ValueError):
        parse_samples(json.dumps([original]), case, 2)


def test_parse_rejects_sample_count_mismatch(case: BenchCase) -> None:
    with pytest.raises(ValueError, match="sample count"):
        parse_samples(json.dumps([row(case)]), case, 5)


@pytest.mark.parametrize("payload", ["not-json", "[]", "{}", "[{}, {}]"])
def test_parse_rejects_bad_shape(case: BenchCase, payload: str) -> None:
    with pytest.raises(ValueError):
        parse_samples(payload, case, 2)
