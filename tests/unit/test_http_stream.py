"""Protocol framing and cancellation against isolated test-only loopback servers."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from backend.engines.http_stream import LocalHTTPClient, SSEDecoder


def test_sse_fragmentation_null_comments_utf8_and_receive_batches() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b": ping\r\ndata: nu", 100) == []
    events = decoder.feed(b'll\r\n\r\ndata: {"content":"\xc3', 200)
    assert len(events) == 1 and events[0].data is None
    events = decoder.feed(b'\xa9"}\n\ndata: {"stop":true}\n\n', 300)
    assert len(events) == 2
    assert events[0].data == {"content": "é"}
    assert events[0].arrival_ns == events[1].arrival_ns == 300
    decoder.finish()


@pytest.mark.parametrize("data", [b"data: {broken}\n\n", b"data: NaN\n\n", b"\xff\n\n"])
def test_sse_rejects_malformed_nonfinite_and_invalid_unicode(data: bytes) -> None:
    with pytest.raises(ValueError):
        SSEDecoder().feed(data, 100)


def test_sse_rejects_truncated_frame() -> None:
    decoder = SSEDecoder()
    decoder.feed(b'data: {"stop":true}', 100)
    with pytest.raises(ValueError, match="Truncated"):
        decoder.finish()


@pytest.mark.parametrize("cancel_request", [True, False])
def test_http_close_response_body_is_interruptible_after_headers(cancel_request: bool) -> None:
    headers_sent = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            headers_sent.set()
            release.wait(3)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cancel = threading.Event()
    timer = None
    try:
        if cancel_request:

            def stop_after_headers() -> None:
                assert headers_sent.wait(2)
                cancel.set()

            timer = threading.Thread(target=stop_after_headers, daemon=True)
            timer.start()
        client = LocalHTTPClient(server.server_port, "TEST_ONLY_KEY")
        started = time.monotonic()
        expected = InterruptedError if cancel_request else TimeoutError
        with pytest.raises(expected):
            client.stream("/completion", {}, 2 if cancel_request else 0.15, cancel)
        assert time.monotonic() - started < 1
    finally:
        release.set()
        if timer:
            timer.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_multiline_data() -> None:
    events = SSEDecoder().feed(b'event: message\ndata: {"a":\ndata: 1}\n\n', 100)
    assert json.loads(events[0].raw_data) == {"a": 1}


def test_cancel_during_keepalive_chunk_decoder_has_no_cross_thread_close() -> None:
    first_sent = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            body = b'data: {"stop":false,"tokens":[1]}\n\n'
            self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
            self.wfile.flush()
            first_sent.set()
            release.wait(3)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cancel = threading.Event()

    def cancel_reader() -> None:
        assert first_sent.wait(2)
        time.sleep(0.03)
        cancel.set()

    timer = threading.Thread(target=cancel_reader, daemon=True)
    timer.start()
    client = LocalHTTPClient(server.server_port, "TEST_ONLY_KEY")
    try:
        with pytest.raises(InterruptedError):
            client.stream("/completion", {}, 2, cancel)
        assert len(client.last_events) == 1
    finally:
        release.set()
        timer.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
