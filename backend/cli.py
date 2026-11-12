"""Local CLI entry point. Models and engines are always explicitly supplied."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from backend.analysis.report import report
from backend.benchmarks.environment import environment
from backend.benchmarks.request_runner import run_request_experiment
from backend.benchmarks.runner import run_experiment
from backend.engines.llama_bench import LlamaBench
from backend.engines.llama_server import LlamaServer
from backend.schemas.config import load_config
from backend.schemas.request_config import RequestConfig
from backend.storage.database import Store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LIOL: reproducible local inference microbenchmarks"
    )
    parser.add_argument(
        "--db", type=Path, default=Path(os.environ.get("LIOL_DB", "data/results/liol.sqlite3"))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Read hardware and binary capabilities")
    doctor.add_argument("--binary", type=Path)
    doctor.add_argument("--engine", choices=("bench", "server"), default="bench")
    for command in ("plan", "run"):
        commands.add_parser(command).add_argument("config", type=Path)
    commands.add_parser("list")
    for command in ("show", "export", "report"):
        sub = commands.add_parser(command)
        sub.add_argument("id")
        if command in ("export", "report"):
            sub.add_argument("--output", type=Path, required=True)
        if command == "export":
            sub.add_argument("--format", choices=("csv", "json"), default="json")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            adapter = LlamaServer if args.engine == "server" else LlamaBench
            result = {
                "environment": environment(),
                "engine": adapter(args.binary).probe() if args.binary else None,
            }
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.command in ("plan", "run"):
            config = load_config(args.config)
            if args.command == "plan":
                print(json.dumps({"config": config.to_dict(), "cases": config.cases()}, indent=2))
                return 0
            store = Store(args.db)
            experiment_id = (
                run_request_experiment(config, LlamaServer(config.binary), store)
                if isinstance(config, RequestConfig)
                else run_experiment(config, LlamaBench(config.binary), store)
            )
            experiment = store.get_experiment(experiment_id)
            directory = args.db.parent / experiment_id
            store.export_json(experiment_id, directory / "experiment.json")
            store.export_csv(experiment_id, directory / "samples.csv")
            report(experiment, directory / "report.md")
            print(
                json.dumps(
                    {
                        "id": experiment_id,
                        "status": experiment["status"],
                        "results": str(directory.resolve()),
                    },
                    indent=2,
                )
            )
            return (
                0
                if experiment["status"] == "completed"
                else (130 if experiment["status"] == "canceled" else 1)
            )
        store = Store(args.db)
        if args.command == "list":
            print(json.dumps(store.list_experiments(), indent=2))
        elif args.command == "show":
            print(json.dumps(store.get_experiment(args.id), indent=2))
        elif args.command == "report":
            report(store.get_experiment(args.id), args.output)
        elif args.format == "csv":
            store.export_csv(args.id, args.output)
        else:
            store.export_json(args.id, args.output)
        return 0
    except InterruptedError as exc:
        print(f"liol: {exc}", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, sqlite3.Error) as exc:
        print(f"liol: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
