#!/usr/bin/env python3
"""TEST ONLY localhost server fixture; never used as a production inference backend."""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HELP = """TEST ONLY llama-server fixture
 -m, --model FILE
 --host HOST
 --port PORT
 -t, --threads N
 -tb, --threads-batch N
 -ngl, --gpu-layers N
 -dev, --device NAME
 -c, --ctx-size N
 -b, --batch-size N
 -ub, --ubatch-size N
 -fa, --flash-attn [on|off|auto]
 --fit [on|off]
 --parallel N
 --no-cache-prompt
 --cache-reuse N
 --cache-ram N
 --no-cache-idle-slots
 --no-context-shift
 --no-ui
 --offline
 --perf
 --api-key-file FILE
 --no-kv-offload
 --no-op-offload
 --list-devices
 --no-slots
 --no-cont-batching
"""


def main() -> int:
    args = sys.argv[1:]
    if "--help" in args:
        print(HELP)
        return 0
    if "--version" in args:
        print("version: TEST ONLY 0.9.0-dev (build 12182, commit fd27b2034)")
        return 0
    if "--list-devices" in args:
        print("Available devices:\n  MTL0: TEST ONLY synthetic Metal device")
        return 0

    def value(flag: str, short: str | None = None) -> str:
        selected = flag if flag in args else short
        if selected is None or selected not in args:
            raise ValueError(f"TEST ONLY fixture requires {flag}")
        return args[args.index(selected) + 1]

    behavior = os.environ.get("LIOL_SERVER_TEST_BEHAVIOR", "success")
    if behavior == "startup_failure":
        print("TEST ONLY deliberate startup failure", file=sys.stderr, flush=True)
        return 9
    host = value("--host")
    if host != "127.0.0.1":
        raise ValueError("TEST ONLY fixture must bind to 127.0.0.1")
    model = str(Path(value("--model", "-m")).resolve())
    context = int(value("--ctx-size", "-c"))
    key = Path(value("--api-key-file")).read_text().strip()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def json_response(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {key}":
                return True
            self.json_response(401, {"error": "TEST ONLY authentication required"})
            return False

        def do_GET(self) -> None:
            if self.path == "/health":
                self.json_response(200, {"status": "ok"})
            elif not self.authenticated():
                return
            elif self.path == "/props":
                self.json_response(
                    200,
                    {
                        "model_path": model,
                        "total_slots": 1,
                        "default_generation_settings": {"n_ctx": context},
                        "build_info": "TEST ONLY b12182-fd27b2034",
                    },
                )
            else:
                self.json_response(404, {"error": "TEST ONLY unknown endpoint"})

        def do_POST(self) -> None:
            if not self.authenticated():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_048_576:
                self.json_response(400, {"error": "TEST ONLY invalid request length"})
                return
            request = json.loads(self.rfile.read(length))
            if self.path == "/tokenize":
                assert request["add_special"] is True
                assert request["parse_special"] is False
                count = 3 if behavior == "short_fixture" else len(request["content"].split()) + 1
                self.json_response(200, {"tokens": list(range(count))})
                return
            if self.path != "/completion":
                self.json_response(404, {"error": "TEST ONLY unknown endpoint"})
                return
            marker = os.environ.get("LIOL_SERVER_TEST_REQUEST_READY")
            if marker:
                Path(marker).write_text(str(os.getpid()))
            if behavior == "failure":
                self.json_response(500, {"error": "TEST ONLY deliberate completion failure"})
                return
            if behavior == "wait":
                time.sleep(30)
            assert request["stream"] is True
            assert request["return_tokens"] is True
            assert request["cache_prompt"] is False
            assert request["temperature"] == 0
            assert isinstance(request["prompt"], list)
            prompt_count = len(request["prompt"])
            target = request["n_predict"]
            generated = min(2, target) if behavior == "early_eos" else target
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def event(data: Any) -> None:
                body = f"data: {json.dumps(data)}\n\n".encode()
                self.wfile.write(body)
                self.wfile.flush()

            try:
                self.wfile.write(b": TEST ONLY keep-alive\n\n")
                self.wfile.flush()
                event(None)  # Headers/progress must not become the first generated-token event.
                time.sleep(0.02)
                if behavior == "malformed":
                    self.wfile.write(b"data: {TEST ONLY invalid JSON\n\n")
                    self.wfile.flush()
                    return
                for index in range(generated):
                    event(
                        {
                            "content": f" token{index}",
                            "tokens": [200 + index],
                            "stop": False,
                            "tokens_predicted": index + 1,
                            "tokens_evaluated": prompt_count,
                        }
                    )
                    time.sleep(0.005)
                cache_n = 1 if behavior == "cache_reuse" else 0
                processed = prompt_count - cache_n
                predicted_ms = max(0, generated - 1) * 5.0
                event(
                    {
                        "stop": True,
                        "content": "",
                        "tokens": [],
                        "tokens_predicted": generated,
                        "tokens_evaluated": prompt_count,
                        "tokens_cached": prompt_count + max(0, generated - 1),
                        "truncated": False,
                        "stop_type": "eos" if generated < target else "limit",
                        "generation_settings": {
                            "n_predict": target,
                            "seed": request["seed"],
                            "temperature": 0.0,
                            "ignore_eos": request["ignore_eos"],
                            "samplers": ["temperature"],
                        },
                        "timings": {
                            "cache_n": cache_n,
                            "prompt_n": processed,
                            "prompt_ms": 2.0,
                            "prompt_per_second": processed * 500.0,
                            "predicted_n": generated,
                            "predicted_ms": predicted_ms,
                            "predicted_per_second": 200.0 if generated >= 2 else 0.0,
                        },
                    }
                )
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((host, int(value("--port"))), Handler)
    server.daemon_threads = True
    print(
        f"TEST ONLY server is listening on http://127.0.0.1:{server.server_port}",
        file=sys.stderr,
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
