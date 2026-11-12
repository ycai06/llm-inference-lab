"""Request plot/table behavior uses test-only observations, never production data."""

from pathlib import Path
from typing import Any

import pytest

from backend.analysis.request_report import report_request


def experiment(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "TEST ONLY",
        "name": "TEST ONLY requests",
        "status": "completed",
        "config": {"mode": "request_warm"},
        "cases": [
            {
                "id": "case",
                "status": "completed",
                "error": None,
                "config": {
                    "prompt_tokens": 128,
                    "generation_tokens": 8,
                    "threads": 4,
                    "gpu_layers": -1,
                },
                "request_samples": samples,
            }
        ],
    }


def test_request_report_separates_units_and_unavailable_metrics(tmp_path: Path) -> None:
    samples = [
        {
            "request_latency_ms": 400.0 + i,
            "ttft_ms": 100.0 + i,
            "prefill_tokens_per_second": 1000.0 + i,
            "decode_tokens_per_second": 20.0 + i,
            "mean_inter_token_latency_ms": None,
            "median_inter_token_latency_ms": None,
            "generated_tokens": 8,
            "early_stop": False,
            "warnings": ["TEST ONLY coalesced event"],
        }
        for i in range(5)
    ]
    path = tmp_path / "request.md"
    report_request(experiment(samples), path)
    text = path.read_text()
    assert "| Request latency | ms | 5 | 402.00" in text
    assert "| Native decode | tokens/s | 5 | 22.00" in text
    assert "| Mean observed token interval | ms | 0 | unavailable" in text
    assert "Early stops: 0" in text
    assert "TEST ONLY coalesced event" in text
    assert "not measured GPU VRAM" in text
    assert path.with_suffix(".png").exists()
    svg = path.with_suffix(".svg").read_text()
    assert "Client-observed latency" in svg
    assert "Native server throughput" in svg


def test_failed_request_report_has_no_plot(tmp_path: Path) -> None:
    data = experiment([])
    data["cases"][0].update(status="timeout", error="TEST ONLY timeout")
    path = tmp_path / "failed.md"
    report_request(data, path)
    assert "TEST ONLY timeout" in path.read_text()
    assert not path.with_suffix(".svg").exists()


def test_request_report_requires_request_mode(tmp_path: Path) -> None:
    data = experiment([])
    data["config"]["mode"] = "synthetic_warm"
    with pytest.raises(ValueError, match="require mode"):
        report_request(data, tmp_path / "invalid.md")


def test_one_request_no_tokens_has_no_confidence_interval(tmp_path: Path) -> None:
    path = tmp_path / "zero.md"
    report_request(
        experiment(
            [
                {
                    "request_latency_ms": 2.0,
                    "ttft_ms": None,
                    "prefill_tokens_per_second": None,
                    "decode_tokens_per_second": None,
                    "mean_inter_token_latency_ms": None,
                    "median_inter_token_latency_ms": None,
                    "generated_tokens": 0,
                    "early_stop": True,
                    "warnings": [],
                }
            ]
        ),
        path,
    )
    assert "Early stops: 1" in path.read_text()
    assert "| TTFT | ms | 0 | unavailable" in path.read_text()
    assert "| Request latency | ms | 1 | 2.00 | 2.00 | unavailable" in path.read_text()
