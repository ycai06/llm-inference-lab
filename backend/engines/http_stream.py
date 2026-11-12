"""Bounded localhost HTTP requests and SSE frames with receive-batch timestamps."""

from __future__ import annotations

import codecs
import http.client
import json
import math
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_EVENT_CHARS = 2 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value: {value}")


def strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


@dataclass(frozen=True)
class SSEEvent:
    data: Any
    arrival_ns: int
    raw_data: str
    event: str | None = None


class SSEDecoder:
    """All frames completed in one read share its receive timestamp."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._data: list[str] = []
        self._event: str | None = None
        self._event_size = 0

    def feed(self, chunk: bytes, arrival_ns: int) -> list[SSEEvent]:
        try:
            self._buffer += self._decoder.decode(chunk)
        except UnicodeDecodeError as exc:
            raise ValueError("SSE stream is not valid UTF-8") from exc
        if len(self._buffer) > MAX_EVENT_CHARS:
            raise ValueError("SSE line exceeds size limit")
        events = []
        while match := re.search(r"[\r\n]", self._buffer):
            pos = match.start()
            if self._buffer[pos] == "\r" and pos + 1 == len(self._buffer):
                break
            separator = 2 if self._buffer[pos : pos + 2] == "\r\n" else 1
            line = self._buffer[:pos]
            self._buffer = self._buffer[pos + separator :]
            if not line:
                if self._data:
                    raw = "\n".join(self._data)
                    data = "[DONE]" if raw == "[DONE]" else strict_json(raw)
                    events.append(SSEEvent(data, arrival_ns, raw, self._event))
                self._data, self._event, self._event_size = [], None, 0
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                self._data.append(value)
                self._event_size += len(value)
                if self._event_size > MAX_EVENT_CHARS:
                    raise ValueError("SSE event exceeds size limit")
            elif field == "event":
                self._event = value
        return events

    def finish(self) -> None:
        try:
            remaining = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("Truncated UTF-8 in SSE stream") from exc
        if remaining or self._buffer.strip() or self._data:
            raise ValueError("Truncated SSE event")


@dataclass(frozen=True)
class StreamResponse:
    events: list[SSEEvent]
    started_ns: int
    completed_ns: int


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        super().__init__(f"Local server returned HTTP {status}: {body[:1000]}")


class _RequestGuard:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        timeout_s: float,
        cancel: threading.Event,
        alive: Callable[[], bool] | None,
    ) -> None:
        self.connection = connection
        self.deadline = time.monotonic() + timeout_s
        self.cancel = cancel
        self.alive = alive
        self.done = threading.Event()
        self.reason: str | None = None
        self.socket: socket.socket | None = None
        self.thread = threading.Thread(target=self._watch, daemon=True, name="liol-http-deadline")

    def _watch(self) -> None:
        while not self.done.wait(0.01):
            if self.cancel.is_set():
                self.reason = "canceled"
            elif time.monotonic() >= self.deadline:
                self.reason = "timeout"
            elif self.alive is not None and not self.alive():
                self.reason = "exited"
            if self.reason is not None:
                sock = self.socket or self.connection.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                # Only interrupt the socket here. HTTPConnection.close() also mutates
                # HTTPResponse state and can race with the reader's chunk decoder.
                # The request thread closes both objects in finish()/finally.
                return

    def start(self) -> None:
        self.check()
        self.thread.start()

    def check(self) -> None:
        if self.cancel.is_set() or self.reason == "canceled":
            raise InterruptedError("Local request canceled")
        if self.reason == "timeout" or time.monotonic() >= self.deadline:
            raise TimeoutError("Local request exceeded its overall deadline")
        if self.reason == "exited" or (self.alive is not None and not self.alive()):
            raise RuntimeError("Managed server exited during request")

    def finish(self) -> None:
        self.done.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1)
        self.connection.close()


class LocalHTTPClient:
    """Connect directly to an owned IPv4 loopback server; no proxy discovery."""

    def __init__(self, port: int, api_key: str, alive: Callable[[], bool] | None = None) -> None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid local server port")
        self.port, self._api_key, self.alive = port, api_key, alive
        self.last_events: list[SSEEvent] = []

    def _open(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_s: float,
        cancel: threading.Event,
    ) -> tuple[http.client.HTTPResponse, _RequestGuard, int]:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("HTTP timeout must be positive and finite")
        if not path.startswith("/") or path.startswith("//") or "\r" in path or "\n" in path:
            raise ValueError("HTTP path must be local and absolute")
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout_s)
        guard = _RequestGuard(connection, timeout_s, cancel, self.alive)
        started = time.perf_counter_ns()
        try:
            guard.start()
            connection.connect()
            guard.socket = connection.sock
            guard.check()
            connection.request(
                method,
                path,
                body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response = connection.getresponse()
            guard.check()
            if response.status != 200:
                error = response.read(MAX_BODY_BYTES + 1)
                guard.check()
                raise HTTPStatusError(response.status, error.decode("utf-8", errors="replace"))
            return response, guard, started
        except BaseException:
            try:
                guard.check()
            finally:
                guard.finish()
            raise

    def json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_s: float,
        cancel: threading.Event,
    ) -> dict[str, Any]:
        response, guard, _ = self._open(method, path, payload, timeout_s, cancel)
        try:
            data = response.read(MAX_BODY_BYTES + 1)
            guard.check()
            if len(data) > MAX_BODY_BYTES:
                raise ValueError("JSON response exceeds size limit")
            result = strict_json(data.decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object from the local server")
            return result
        except (OSError, http.client.HTTPException):
            guard.check()
            raise
        finally:
            guard.finish()
            response.close()

    def stream(
        self, path: str, payload: dict[str, Any], timeout_s: float, cancel: threading.Event
    ) -> StreamResponse:
        self.last_events = []
        response, guard, started = self._open("POST", path, payload, timeout_s, cancel)
        decoder, size = SSEDecoder(), 0
        try:
            if not response.getheader("Content-Type", "").lower().startswith("text/event-stream"):
                raise ValueError("Expected text/event-stream from the local server")
            while True:
                chunk = response.read1(64 * 1024)
                arrived = time.perf_counter_ns()
                guard.check()
                if not chunk:
                    decoder.finish()
                    raise ValueError("SSE stream ended without a terminal completion event")
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    raise ValueError("SSE response exceeds size limit")
                for event in decoder.feed(chunk, arrived):
                    self.last_events.append(event)
                    if isinstance(event.data, dict) and "error" in event.data:
                        raise RuntimeError(f"Local completion failed: {event.data['error']}")
                    if isinstance(event.data, dict) and event.data.get("stop") is True:
                        return StreamResponse(list(self.last_events), started, arrived)
        except (OSError, http.client.HTTPException):
            guard.check()
            raise
        finally:
            guard.finish()
            response.close()
