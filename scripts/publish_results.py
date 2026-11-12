"""Publish compact measured samples without local model/binary paths or raw logs.

Run from the repository: uv run python scripts/publish_results.py EXPERIMENT_ID STEM
The complete local export remains the authoritative provenance record.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from backend.analysis.report import report
from backend.storage.database import Store


def portable(value: Any, paths: list[str]) -> Any:
    """Replace configured local paths even inside nested native response/raw JSON fields."""
    if isinstance(value, dict):
        return {key: portable(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, paths) for item in value]
    if isinstance(value, str):
        for path in sorted(paths, key=len, reverse=True):
            value = value.replace(path, Path(path).name)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id")
    parser.add_argument("stem", type=Path, help="Output stem, e.g. reports/baseline")
    parser.add_argument(
        "--db", type=Path, default=Path(os.environ.get("LIOL_DB", "data/results/liol.sqlite3"))
    )
    args = parser.parse_args()
    original = Store(args.db).get_experiment(args.id)
    config = {k: v for k, v in original["config"].items() if k not in ("binary", "model")}
    if "fixtures" in config:
        config["fixtures"] = [Path(p).name for p in config["fixtures"]]
    metadata = original["metadata"]
    model = metadata["model"]
    public = {
        "id": original["id"],
        "name": original["name"],
        "status": original["status"],
        "created_at": original["created_at"],
        "finished_at": original["finished_at"],
        "config": config,
        "metadata": {
            "environment": metadata["environment"],
            "model": {k: model[k] for k in ("sha256", "size_bytes", "gguf_version", "metadata")},
            "engine": {k: metadata["engine"][k] for k in ("version", "binary_sha256")},
            "timing_source": metadata["timing_source"],
            "warmup_policy": metadata["warmup_policy"],
            "kv_policy": metadata["kv_policy"],
            "order": metadata["order"],
            "scope": "Compact measured data; complete local raw export retains full provenance",
        },
        "cases": [],
    }
    for case in original["cases"]:
        process = case["process"] or {}
        case_config = dict(case["config"])
        if "fixture" in case_config:
            case_config["fixture"] = Path(case_config["fixture"]).name
        resources = process.get("server", process)
        public["cases"].append(
            {
                "id": case["id"],
                "status": case["status"],
                "error": case["error"],
                "config": case_config,
                "samples": case["samples"],
                "request_samples": case.get("request_samples", []),
                "process": {
                    k: resources.get(k)
                    for k in (
                        "sampled_peak_rss_bytes",
                        "sampled_cpu_seconds",
                        "resource_samples",
                        "wall_ns",
                        "startup_duration_ms",
                    )
                },
            }
        )
        if "fixture" in process:
            public["cases"][-1]["fixture"] = {
                k: v for k, v in process["fixture"].items() if k != "path"
            }
    for key in ("tokenizer", "sampling", "unavailable_metrics", "workload", "mode"):
        if key in metadata:
            public["metadata"][key] = metadata[key]
    args.stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        original["config"]["model"],
        original["config"]["binary"],
        *original["config"].get("fixtures", []),
    ]
    public = portable(public, paths)
    args.stem.with_suffix(".json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    report(public, args.stem.with_suffix(".md"))


if __name__ == "__main__":
    main()
