"""The orchestrator depends on this contract, rather than process internals."""

from pathlib import Path
from threading import Event
from typing import Any, Protocol

from backend.engines.llama_bench import BenchCase
from backend.engines.process import ProcessResult


class BenchmarkEngine(Protocol):
    def probe(self) -> dict[str, Any]: ...

    def run(
        self,
        model: Path,
        case: BenchCase,
        repetitions: int,
        timeout_s: float,
        cancel: Event | None = None,
    ) -> ProcessResult: ...

    def parse(
        self, stdout: str, case: BenchCase, repetitions: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...
