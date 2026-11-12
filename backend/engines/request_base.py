"""Lifecycle and request interface for tokenized text-completion engines."""

from __future__ import annotations

from threading import Event
from types import TracebackType
from typing import Any, Protocol, Self

from backend.engines.llama_server import ServerSettings


class RequestSession(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def tokenize(
        self,
        text: str,
        add_special: bool = True,
        parse_special: bool = False,
        timeout_s: float = 30,
    ) -> list[int]: ...

    def complete(
        self,
        prompt_tokens: list[int],
        generation_tokens: int,
        seed: int,
        ignore_eos: bool,
        cache_prompt: bool,
        timeout_s: float,
        cancel: Event,
        trial_index: int = 0,
    ) -> dict[str, Any]: ...


class RequestEngine(Protocol):
    def probe(self) -> dict[str, Any]: ...

    def session(self, settings: ServerSettings, cancel: Event) -> RequestSession: ...
