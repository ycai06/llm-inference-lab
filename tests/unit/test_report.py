"""A report must reflect only observed samples and explicit failure states."""

from pathlib import Path

from backend.analysis.report import report


def test_report_empty_failure_has_no_chart(tmp_path: Path) -> None:
    path = tmp_path / "failure.md"
    report(
        {
            "id": "test",
            "name": "Failure",
            "status": "failed",
            "cases": [
                {
                    "id": "case",
                    "config": {},
                    "samples": [],
                    "status": "timeout",
                    "error": "Timed out",
                }
            ],
        },
        path,
    )
    assert "timeout" in path.read_text()
    assert not path.with_suffix(".svg").exists()


def test_report_with_single_sample_has_no_interval(tmp_path: Path) -> None:
    path = tmp_path / "single.md"
    report(
        {
            "id": "test",
            "name": "Single",
            "status": "completed",
            "cases": [
                {
                    "config": {
                        "kind": "prefill",
                        "tokens": 8,
                        "threads": 1,
                        "gpu_layers": 0,
                        "batch_size": 32,
                        "ubatch_size": 16,
                    },
                    "samples": [{"tokens_per_second": 20.0}],
                }
            ],
        },
        path,
    )
    assert "20.00" in path.read_text()
    assert "unavailable" in path.read_text()
    assert "20.00" in path.with_suffix(".svg").read_text()
